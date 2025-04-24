import datetime
import functools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nodc_calculations.calculate import (
    dissolved_inorganic_nitrogen,
    oxygen,
    oxygen_saturation,
)

from nodc_statistics.calculate_parameter import get_prio_par
from nodc_statistics.regions import read_geo_info_file, sea_basin_for_position

statistics_directory = Path(__file__).parent / "data" / "statistics"
STATISTIC_FILES = {
    path.stem: path for path in statistics_directory.glob("*") if path.is_file()
}

SETTING_FILE = {
    "settings": Path(__file__).parent / "data" / "settings.json",
}

AREA_TAG_FILE = (
    Path(__file__).parent / "data" / "pos_area_tag_SVAR2022_HELCOM_OSPAR_vs2.csv"
)

SAVE_KWARGS = {"sep": "\t", "encoding": "utf-8", "index": False, "float_format": "%.2f"}

GPKG_FILE = Path.home() / "SVAR2022_HELCOM_OSPAR_vs2.gpkg"


def create_depth_bins(standard_depths):
    """
    Converts a list of standard depths into a dictionary with midpoints as keys
    and [min, max] depth intervals as values.
    """
    depth_dict = {}
    for i, midpoint in enumerate(standard_depths):
        if i == 0:
            min_depth = 0
        else:
            min_depth = (standard_depths[i - 1] + midpoint) / 2

        if i == len(standard_depths) - 1:
            max_depth = midpoint + (midpoint - min_depth)  # Extend last bin
        else:
            max_depth = (midpoint + standard_depths[i + 1]) / 2

        depth_dict[midpoint] = [min_depth, max_depth]

    return depth_dict


def create_file_path_dict(directory):
    directory = Path(__file__).parent / "data"
    #
    file_path_dict = {}
    p = Path(directory)
    for file in p.glob("*"):
        if file.is_file():
            file_path_dict[file.name] = str(file.resolve())
    return file_path_dict


@functools.cache
def nan_float(value: str):
    try:
        return float(value)
    except ValueError:
        return np.nan


class DataHandler:
    """
    DataHandler(data_path = "/sharkdata.txt") -> DataHandler object
    Reads data from sharkweb downloaded to given path.
    Maps sampling depths that are outside standard depths to standard depths
    Adds columns:
        - timestamp
        - month
        - year
        - salt
        - temp
        - doxy
        - depth (DEPH mapped to standard depths)
    self.data contaings a pandas dataframe with the data
    """

    def __init__(self, data_path):
        super().__init__()
        with SETTING_FILE["settings"].open("r", encoding="utf-8") as file:
            self.settings = json.load(file)
        self.data = self._read_shark_data(data_path)
        self._invalid_flags = {"S", "B", "E", 3, 4}
        # Extract parameter column names by finding columns with matching 'Q_' prefix
        quality_flag_columns = [col for col in self.data.columns if col.startswith("Q_")]
        for col in quality_flag_columns:
            self.data[col] = self.data[col].astype(str)

        self._parameters = [
            col[2:] for col in quality_flag_columns if col[2:] in self.data.columns
        ]
        self._valid_data = None
        self._match_sea_basins()
        # Convert standard_depths to depth_intervals
        depth_intervals = {
            key: create_depth_bins(values)
            for key, values in self.settings["standard_depths"].items()
        }
        self.data = self.assign_depth_intervals(self.data, depth_intervals)
        self._add_parameters(self.data)

    def _read_shark_data(self, filepath: str):
        """read text file from sharkweb"""
        file_path = Path(filepath)
        print(f"reading file {file_path}\n... ... ...")
        df = pd.read_csv(open(file_path, encoding="utf-8"), sep="\t")
        df["timestamp"] = df["SDATE"].apply(pd.Timestamp)
        df["month"] = df["timestamp"].apply(lambda x: x.month)
        df["year"] = df["timestamp"].apply(lambda x: x.year)

        return df

    @property
    def invalid_flags(self):
        return self._invalid_flags

    @invalid_flags.setter
    def invalid_flags(self, value):
        # You may want to validate the input here
        if not isinstance(value, (set, list, tuple)):
            raise ValueError("Invalid flags must be a set, list, or tuple.")
        self._invalid_flags = set(value)

    @property
    def valid_data(self):
        return self._remove_invalid_data()

    def _remove_invalid_data(self):
        valid_data = self.data.copy()
        # Set values to np.nan where the quality flag is invalid
        # The where function keeps the original value if the condition is True,
        # and sets it to np.nan if the condition is False. That is the reason for the ~
        for param in self._parameters:
            valid_data[param] = valid_data[param].where(
                ~valid_data[f"Q_{param}"].isin(self._invalid_flags), np.nan
            )

        return valid_data

    def _add_parameters(self, data):
        df = oxygen(data)
        data.loc[:, "salt"] = data.apply(
            lambda row: get_prio_par(
                row.SALT_CTD, row.SALT_BTL, row.Q_SALT_CTD, row.Q_SALT_BTL
            ),
            axis=1,
        )
        data.loc[:, "temp"] = data.apply(
            lambda row: get_prio_par(
                row.TEMP_CTD, row.TEMP_BTL, row.Q_TEMP_CTD, row.Q_TEMP_BTL
            ),
            axis=1,
        )

        data["doxy"] = df["oxygen"].copy()
        data["Q_doxy"] = df["Q_DOXY_BTL"].copy()

        dissolved_inorganic_nitrogen(data)

        _, _, o2sat_data = oxygen_saturation(data)
        print(o2sat_data.head())
        data["oxygen_saturation"] = o2sat_data["oxygen_saturation"].copy()

    def _match_sea_basins(self):
        self._geo_info = read_geo_info_file(filepath=GPKG_FILE)

        print("Matching sea basins...")
        # Assuming df is your DataFrame and sea_basin_for_position is the function
        # to apply
        # Step 1: Extract unique combinations of LONGI and LATIT
        unique_positions = self.data.copy()[["LONGI_DD", "LATIT_DD"]].drop_duplicates()
        # Step 2: Apply the function to each unique combination
        unique_positions["sea_basin"] = unique_positions.apply(
            lambda row: sea_basin_for_position(
                row["LONGI_DD"], row["LATIT_DD"], self._geo_info
            ),
            axis=1,
        )
        # Step 3: Map the results back to the original DataFrame
        self.data = self.data.merge(
            unique_positions, on=["LONGI_DD", "LATIT_DD"], how="left"
        )
        print("Matching sea basins finished")

    def assign_depth_intervals(
        self, data: pd.DataFrame, depth_intervals: dict
    ) -> pd.DataFrame:
        """
        Assigns 'depth_interval' and 'depth' (midpoint) columns to a DataFrame.

        Parameters:
        - data (pd.DataFrame):
            Input DataFrame with columns "DEPH" and optional "sea_basin".

        Returns:
        - pd.DataFrame:
            Updated DataFrame.
        """
        # Ensure new columns exist
        data["depth"] = data["DEPH"].copy()
        data["depth_interval"] = data["DEPH"].astype(str) + "_" + data["DEPH"].astype(str)

        for sea_basin in data["sea_basin"].dropna().unique():  # Ignore NaN sea_basins
            intervals = depth_intervals.get(sea_basin, depth_intervals["default"])
            for midpoint, (min_depth, max_depth) in intervals.items():
                boolean = (
                    (data["sea_basin"] == sea_basin)
                    & (data["DEPH"] >= min_depth)
                    & (data["DEPH"] <= max_depth)
                )
                data.loc[boolean, "depth"] = midpoint
                data.loc[boolean, "depth_interval"] = f"{min_depth}_{max_depth}"

        # Apply default intervals where 'sea_basin' is missing
        for midpoint, (min_depth, max_depth) in depth_intervals["default"].items():
            mask = data["sea_basin"].isna() & data["DEPH"].between(min_depth, max_depth)
            data.loc[mask, "depth"] = midpoint
            data.loc[mask, "depth_interval"] = f"{min_depth}_{max_depth}"

        return data


class CalculateStatistics:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        with SETTING_FILE["settings"].open("r", encoding="utf-8") as file:
            self.settings = json.load(file)

        # aggregera alla data på station, djup, år, månad
        self.agg_data = self._agg_station_data()

    def _agg_station_data(self):
        # Definiera kolumner att gruppera på
        # Skulle grupperat på REG_ID men fiskestationer saknar REG_ID och de vill vi ha
        # Använder STATN för alla så länge.
        # Alternativ: Använda REG_ID när det finns och fylla ut med STATN när det saknas?
        group_cols = ["STATN", "depth", "depth_interval", "year", "month", "sea_basin"]

        # Skapa dictionary för aggregering
        agg_dict = dict.fromkeys(self.settings["statistic_parameters"], "mean")

        # Utför gruppering och aggregering
        grouped = self.data.groupby(group_cols).agg(agg_dict).reset_index()

        return grouped

    def profile_statistics(self, save=True):
        """
        Calculate statistics for standard depths by month and sea basin
        Saves results to statistic library
        """
        # Definiera aggregeringsfunktioner och deras namn
        agg_funcs = {
            "mean": "mean",
            "std": "std",
            "count": "count",
            "max": "max",
            "min": "min",
            "95p": lambda x: np.percentile(x, 95),
            "5p": lambda x: np.percentile(x, 5),
        }

        # Definiera kolumner att gruppera på
        group_cols = ["depth", "depth_interval", "month", "sea_basin"]

        # Skapa dictionary för aggregering
        agg_dict = {
            param: list(agg_funcs.values())
            for param in self.settings["statistic_parameters"]
        }

        # Utför gruppering och aggregering
        grouped = self.agg_data.groupby(group_cols).agg(agg_dict)

        # Byt namn på kolumnerna
        # matcha namnen som groupby tilldelat mot namnen som vi satt i agg_funcs
        names = {
            col[1]: agg[0]
            for col, agg in zip(grouped.columns[0 : len(agg_funcs)], agg_funcs.items())
        }
        new_columns = []
        for col in grouped.columns:
            param, func = col
            suffix = names.get(func, func)  # Hämta suffix från names dictionary
            new_columns.append(f"{param}:{suffix}")

        grouped.columns = new_columns
        # Återställ index för att få en platt DataFrame
        grouped = grouped.reset_index().round(3)
        # set a limit for minimum number of values to return statistics
        threshold = 15
        # Iterate through parameters and apply the condition
        for param in self.settings["statistic_parameters"]:
            # Identify all columns related to this parameter
            related_cols = [col for col in grouped.columns if col.split(":")[0] == param]
            # Identify the ':count' column
            count_col = f"{param}:count"

            # Apply the condition and set all related columns to np.nan
            if count_col in grouped.columns and (grouped[count_col] <= threshold).any():
                grouped.loc[grouped[count_col] <= threshold, related_cols] = np.nan

        # Remove rows where all non-group columns are NaN
        grouped.dropna(
            subset=[col for col in grouped.columns if col not in group_cols],
            how="all",
            inplace=True,
        )

        # Check if the DataFrame is empty before saving
        if save and not grouped.empty:
            self._save_statistic_files(grouped)

    def _save_statistic_files(self, data, column_name="sea_basin", file_format="csv"):
        # Grupper DataFrame efter `area_tag`
        grouped = data.groupby(column_name)

        # Spara varje grupp i en separat fil
        for area_tag, group in grouped:
            # Skapa ett filnamn baserat på area_tag
            # FIX: kolla varför det ligger med ett \n i "Eastern Gotland Basin Swedish
            # Coastal waters"
            clean_area_tag = area_tag.replace("\n", "").strip()
            file_name = f"{clean_area_tag}.{file_format}"
            # Spara gruppen till en fil
            group.to_csv(
                Path(__file__).parent / "data/statistics" / file_name, **SAVE_KWARGS
            )


@functools.cache
def get_profile_statistics_for_parameter_and_position(
    parameter, longitude, latitude, point_in_time: datetime.datetime
):
    sea_basin = sea_basin_for_position(longitude, latitude)
    return get_profile_statistics_for_parameter_and_sea_basin(
        parameter, sea_basin, point_in_time
    )


@functools.cache
def get_profile_statistics_for_parameter_and_sea_basin(
    parameter: str, sea_basin: str, point_in_time: datetime.datetime
):
    try:
        statistic_path = STATISTIC_FILES[sea_basin]
    except KeyError:
        print(f"no basin named {sea_basin} {parameter}")
        return {
            "mean": [np.nan],
            "lower_limit": [np.nan],
            "upper_limit": [np.nan],
            "depth": [np.nan],
            "min_depth": [np.nan],
            "max_depth": [np.nan],
        }

    # Read the CSV file into a DataFrame
    df = pd.read_csv(statistic_path, delimiter="\t")

    # Filter rows based on the month
    filtered_df = df[df["month"].astype(int) == point_in_time.month]
    try:
        filtered_df[f"{parameter}:mean"]
        filtered_df[f"{parameter}:std"]
    except KeyError:
        print(f"no {parameter} for {sea_basin} ")
        return {
            "mean": [np.nan],
            "lower_limit": [np.nan],
            "upper_limit": [np.nan],
            "depth": [np.nan],
            "min_depth": [np.nan],
            "max_depth": [np.nan],
        }

    # Extract relevant columns, handling missing values if necessary
    mean_values = filtered_df[f"{parameter}:mean"].apply(nan_float).tolist()
    std_values = filtered_df[f"{parameter}:std"].apply(nan_float).tolist()
    depth = filtered_df["depth"].apply(nan_float).tolist()
    # Split the "depth_interval" column into two separate columns
    min_max_df = filtered_df["depth_interval"].str.split("_", expand=True)
    min_depth = min_max_df[0].tolist()
    max_depth = min_max_df[1].tolist()

    # filter the lists to remove depths without statistics
    filtered_data = [
        (mean, std, depth, min_depth, max_depth)
        for mean, std, depth in zip(mean_values, std_values, depth)
        if not (np.isnan(mean) or np.isnan(std))
    ]

    # Unzip filtered data back into two separate lists
    mean_values, std_values, depth, min_depth, max_depth = (
        map(list, zip(*filtered_data))
        if filtered_data
        else ([np.nan], [np.nan], [np.nan], [np.nan], [np.nan])
    )

    return {
        "mean": mean_values,
        "lower_limit": [
            round(mean_value - std_value, 2)
            for mean_value, std_value in zip(mean_values, std_values)
        ],
        "upper_limit": [
            round(mean_value + std_value, 2)
            for mean_value, std_value in zip(mean_values, std_values)
        ],
        "depth": depth,
        "min_depth": min_depth,
        "max_depth": max_depth,
    }


# Load the JSON files
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def create_sharktoolbox_json_from_directory(data_directory):
    get_stb_basin_names = {str(no): f"TYPOMR_KOD_{no!s}" for no in range(25)}
    # Add the special cases for 1 and 12 with 's' and 'n'
    for base_no in [1, 12]:
        get_stb_basin_names[f"{base_no}s"] = f"TYPOMR_KOD_{base_no}s"
        get_stb_basin_names[f"{base_no}n"] = f"TYPOMR_KOD_{base_no}n"
    get_stb_basin_names.update(
        {
            "Bothnian Bay": "BASIN_NR_1",
            "The Quark": "BASIN_NR_2",
            "Bothnian Sea": "BASIN_NR_3",
            "Åland Sea": "BASIN_NR_4",
            "Gulf of Finland": "BASIN_NR_6",
            "Northern Baltic Proper": "BASIN_NR_7",
            "Gulf of Riga": "BASIN_NR_10",
            "Western Gotland Basin": "BASIN_NR_8",
            "Eastern Gotland Basin": "BASIN_NR_9",
            "Gdansk Basin": "BASIN_NR_11",
            "Bornholm Basin": "BASIN_NR_12",
            "Arkona Basin": "BASIN_NR_13",
            "Great Belt": "BASIN_NR_14",
            "Kattegat": "BASIN_NR_16",
            "Skagerrak": "BASIN_NR_17",
        }
    )

    # Initialize config dictionary
    config_data = {}

    # Define file directory
    data_dir = Path(data_directory)

    # Loop over all CSV files
    for file_path in data_dir.glob("*.csv"):
        sea_area = file_path.stem  # Extract sea area from filename
        df = pd.read_csv(file_path, sep="\t", encoding="utf8")
        df.rename(
            columns={
                "depth": "DEPH_intrp",  # or replace how it is read in stb
                "din:mean": "DIN:mean",
                "salt:mean": "SALT:mean",
                "temp:mean": "TEMP:mean",
                "oxygen_saturation:mean": "DOXY_SAT:mean",
                "doxy:mean": "DOXY:mean",
                "din:std": "DIN:std",
                "salt:std": "SALT:std",
                "temp:std": "TEMP:std",
                "doxy:std": "DOXY:std",
            },
            inplace=True,
        )
        # config_data[get_stb_basin_names.get(sea_area, sea_area)] = {}
        # Extract unique parameter names
        param_names = [
            col for col in df.columns if any(sub in col for sub in ["std", "mean"])
        ]

        profile = {}
        for month, group in df.groupby("month"):
            # Replace NaN with None and ensure the values are converted to lists
            profile[str(month)] = {
                param: group[param]
                .where(pd.notna(group[param]), None)
                .tolist()  # Convert to list
                for param in [*param_names, "DEPH_intrp"]
            }

        config_data[get_stb_basin_names.get(sea_area, sea_area)] = {"profile": profile}

    def replace_nan_with_none(d):
        """Recursively replace NaN values with None in the dictionary."""
        if isinstance(d, dict):
            return {k: replace_nan_with_none(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [replace_nan_with_none(item) for item in d]
        return None if pd.isna(d) else d

    # Replace NaN with None in the entire config_data dictionary
    # config_data = replace_nan_with_none(config_data)

    return config_data


SVAR2022_basin_names = {
    "BASIN_NR_1": "Bothnian Bay",
    "BASIN_NR_2": "The Quark",
    "BASIN_NR_3": "Bothnian Sea",
    "BASIN_NR_4": "Åland Sea",
    "BASIN_NR_6": "Gulf of Finland",
    "BASIN_NR_7": "Northern Baltic Proper",
    "BASIN_NR_10": "Gulf of Riga",
    "BASIN_NR_8": "Western Gotland Basin",
    "BASIN_NR_9": "Eastern Gotland Basin",
    "BASIN_NR_11": "Gdansk Basin",
    "BASIN_NR_12": "Bornholm Basin",
    "BASIN_NR_13": "Arkona Basin",
    "BASIN_NR_14": "Great Belt",
    "BASIN_NR_16": "Kattegat",
    "BASIN_NR_17": "Skagerrak",
    "TYPOMR_KOD_1n": "Västkusten inre norra",
    "TYPOMR_KOD_1s": "Västkusten inre södra",
    "TYPOMR_KOD_2": "Västkusten fjordar",
    "TYPOMR_KOD_3": "Västkusten yttre Skagerrak",
    "TYPOMR_KOD_4": "Västkusten yttre Kattegatt",
    "TYPOMR_KOD_5": "halland",
    "TYPOMR_KOD_6": "Öresund",
    "TYPOMR_KOD_7": "Skåne",
    "TYPOMR_KOD_8": "Blekinge, Kalmar inre",
    "TYPOMR_KOD_9": "Blekinge, Kalmar yttre",
    "TYPOMR_KOD_10": "Gotland sydöst, Öland öster",
    "TYPOMR_KOD_11": "Gotland väster, norr",
    "TYPOMR_KOD_12s": "Östergötland, Sthlm södra",
    "TYPOMR_KOD_12n": "Östergötland, Sthlm norra",
    "TYPOMR_KOD_13": "Östergötland intre",
    "TYPOMR_KOD_14": "Östergötland yttre",
    "TYPOMR_KOD_15": "Sthlm skärgård yttre",
    "TYPOMR_KOD_16": "Södra bottehavet, inre",
    "TYPOMR_KOD_17": "Södra bottehavet, yttre",
    "TYPOMR_KOD_18": "Höga kusten inre",
    "TYPOMR_KOD_19": "Höga kusten yttre",
    "TYPOMR_KOD_20": "Kvarken inre",
    "TYPOMR_KOD_21": "Kvarken yttre",
    "TYPOMR_KOD_22": "Bottenviken inre",
    "TYPOMR_KOD_23": "Bottenviken yttre",
    "TYPOMR_KOD_24": "Sthlm inre skärgård",
}


# Function to create plots
def plot_comparison(original_data, new_data, sea_basin, months):
    parameters = ["SALT", "TEMP", "DOXY", "PHOS", "DIN", "SIO3-SI"]

    # Iterate over the parameters and create plots
    for month in months:
        try:
            original = original_data[sea_basin]["profile"].get(str(month), {})
        except KeyError:
            print(f"original data missing in {sea_basin}")
            continue
        try:
            new = new_data[sea_basin]["profile"].get(str(month), {})
        except KeyError:
            print(f"new data missing in {sea_basin}")
            continue

        fig, axs = plt.subplots(2, 3, figsize=(11.69, 8.27))
        axs = axs.flatten()
        for i, param in enumerate(parameters):
            ax = axs[i]
            # Get the data for the original and new datasets
            depth_original = original.get("DEPH", [np.nan])
            depth_new = new.get("DEPH_intrp", [np.nan])
            try:
                original_mean = original.get(
                    f"{param}:mean", len(depth_original) * [np.nan]
                )
                original_std = original.get(
                    f"{param}:std", len(depth_original) * [np.nan]
                )
            except KeyError:
                print(f"original data missing in month {month} {param}")
            try:
                new_mean = new.get(f"{param}:mean", len(depth_new) * [np.nan])
                new_std = new.get(f"{param}:std", len(depth_new) * [np.nan])
            except KeyError:
                print(f"new data missing in {sea_basin}, month {month} {param}")

            # if not original_mean or not new_mean:
            #     continue  # Skip if there's no data for the parameter

            # Plot the original data (solid line)
            ax.plot(
                original_mean,
                depth_original,
                color="black",
                label="Original",
                linewidth=2,
            )
            ax.fill_betweenx(
                depth_original,
                np.array(original_mean) - 2 * np.array(original_std),
                np.array(original_mean) + 2 * np.array(original_std),
                color="grey",
                alpha=0.3,
            )
            ax.plot(
                np.array(original_mean) - 2 * np.array(original_std),
                depth_original,
                color="black",
                linestyle="-",
                linewidth=1,
            )  # Border of grey area
            ax.plot(
                np.array(original_mean) + 2 * np.array(original_std),
                depth_original,
                color="black",
                linestyle="-",
                linewidth=1,
            )  # Border of grey area

            # Plot the new data (dashed line)
            ax.plot(
                new_mean,
                depth_new,
                color="black",
                linestyle="--",
                label="new",
                linewidth=2,
            )
            ax.fill_betweenx(
                depth_new,
                np.array(new_mean) - 2 * np.array(new_std),
                np.array(new_mean) + 2 * np.array(new_std),
                color="grey",
                alpha=0.3,
            )
            ax.plot(
                np.array(new_mean) - 2 * np.array(new_std),
                depth_new,
                color="black",
                linestyle="--",
                linewidth=1,
            )  # Border of grey area
            ax.plot(
                np.array(new_mean) + 2 * np.array(new_std),
                depth_new,
                color="black",
                linestyle="--",
                linewidth=1,
            )  # Border of grey area

            ax.set_title(f"{param}")
            ax.set_ylabel("Depth (m)")
            ax.set_xlabel(param)
            ax.legend()
            ax.yaxis.set_inverted(True)
        fig.suptitle(f"{SVAR2022_basin_names.get(sea_basin, sea_basin)} - Month {month}")
        # Adjust layout for better spacing
        plt.tight_layout()

        # Save the figure as a PNG file (one plot per sea_basin and month)
        save_path = (
            Path("..")
            / "figures"
            / f"{SVAR2022_basin_names.get(sea_basin, sea_basin)}_month_{month}_comparison.png"  # noqa: E501
        )
        fig.savefig(save_path, dpi=300)
        plt.close(fig)  # Close the figure to free memory


if __name__ == "__main__":
    ## create profile statistics ###
    data = DataHandler("C:/LenaV/code/data/sharkweb_data_1991-2020_for_statistics.txt")

    valid_data = data.valid_data

    statistics = CalculateStatistics(valid_data)
    statistics.profile_statistics()

    ## return statisitics
    get_profile_statistics_for_parameter_and_position(
        "TEMP_CTD", 10.759, 58.3050, datetime.datetime(2024, 5, 16)
    )
    get_profile_statistics_for_parameter_and_sea_basin(
        "TEMP_CTD", "Kattegat", datetime.datetime(2024, 5, 16)
    )

    ## create a json-file for sharktoolboc
    directory_path = (
        "C:/LenaV/code/w_qc-tool/nodc-statistics/src/nodc_statistics/data/statistics"
    )
    sharktoolbox_format = create_sharktoolbox_json_from_directory(
        data_directory=directory_path
    )
    # Write to a JSON file
    with open(
        "C:/LenaV/code/w_sharktoolbox/Sharktoolbox/data/statistics/basin_statistics.json",
        "w",
    ) as f:
        json.dump(sharktoolbox_format, f, indent=4)  # indent=4 makes it pretty-printed

    ## Plot statistics
    # Load original and new data from JSON files
    original_data = load_json(
        "C:/LenaV/code/w_sharktoolbox/SharkToolbox/data/statistics/basin_statistics_1991-2020.json"
    )
    new_data = load_json(
        "C:/LenaV/code/w_sharktoolbox/SharkToolbox/data/statistics/basin_statistics.json"
    )

    # Sea basins to compare
    sea_basins = list(original_data.keys())

    # Loop over each sea basin and plot comparisons for all months
    for sea_basin in sea_basins:
        months = list(
            original_data[sea_basin]["profile"].keys()
        )  # Get months for the sea basin
        plot_comparison(original_data, new_data, sea_basin, months)
