import datetime
import functools
import json
from pathlib import Path

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
            )  # noqa: E501

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
    }


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
