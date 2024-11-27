import datetime
import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from nodc_calculations.calculate import dissolved_inorganic_nitrogen, oxygen_saturation

from nodc_statistics import calculate_parameter, regions

statistics_directory = Path(__file__).parent / "data" / "statistics"
STATISTIC_FILES = {
    path.stem: path for path in statistics_directory.glob("*") if path.is_file()
}

SETTING_FILE = {
    "settings": Path(__file__).parent / "data" / "settings.json",
}

AREA_TAG_FILE = Path(__file__).parent / "data" / "pos_area_tag_1991_2020.csv"

SAVE_KWARGS = {"sep": "\t", "encoding": "utf-8", "index": False, "float_format": "%.2f"}


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
        calculate_parameter.map_to_standard_depth(
            data=self.data, standard_depths=self.settings["standard_depths"]
        )
        self._invalid_flags = {"S", "B", "E", 3, 4}
        # Extract parameter column names by finding columns with matching 'Q_' prefix
        quality_flag_columns = [col for col in self.data.columns if col.startswith('Q_')]
        for col in quality_flag_columns:
            self.data[col] = self.data[col].astype(str)

        self._parameters = [col[2:] for col in quality_flag_columns if col[2:] in self.data.columns]
        self._valid_data = None
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
        for param in self._parameters:
            valid_data[param] = valid_data[param].where(~valid_data[f"Q_{param}"].isin(self._invalid_flags), np.nan)

        return valid_data

    def _add_parameters(self, data):
        data.loc[:, "doxy"] = data.apply(
            lambda row: calculate_parameter.get_prio_par_oxy(
                row.DOXY_BTL, row.DOXY_CTD, row.Q_DOXY_BTL, row.Q_DOXY_CTD
            ),
            axis=1,
        )

        data.loc[:, "salt"] = data.apply(
            lambda row: calculate_parameter.get_prio_par(
                row.SALT_CTD, row.SALT_BTL, row.Q_SALT_CTD, row.Q_SALT_BTL
            ),
            axis=1,
        )
        data.loc[:, "temp"] = data.apply(
            lambda row: calculate_parameter.get_prio_par(
                row.TEMP_CTD, row.TEMP_BTL, row.Q_TEMP_CTD, row.Q_TEMP_BTL
            ),
            axis=1,
        )

        din_data = dissolved_inorganic_nitrogen(
            data.rename(
                columns={
                    "Q_DOXY_BTL": "Q_doxy",
                }
            )
        )
        data["din"] = din_data["din"].copy()

        _, _, o2sat_data = oxygen_saturation(
            data.rename(
                columns={
                    "Q_DOXY_BTL": "Q_doxy",
                }
            )
        )
        print(o2sat_data.head())
        data["oxygen_saturation"] = o2sat_data["oxygen_saturation"].copy()

class CalculateStatistics:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        with SETTING_FILE["settings"].open("r", encoding="utf-8") as file:
            self.settings = json.load(file)
        self.area_tags_df = pd.read_csv(open(AREA_TAG_FILE, encoding="utf-8"), sep="\t")
        self._set_up_areas()

        # aggregera alla data på station, djup, år, månad
        self.agg_data = self._agg_station_data()

    def _agg_station_data(self):
        # Definiera kolumner att gruppera på
        group_cols = ["STATN", "depth", "year", "month", "area_tag"]

        # Skapa dictionary för aggregering
        agg_dict = dict.fromkeys(self.settings["statistic_parameters"], "mean")

        # Utför gruppering och aggregering
        grouped = self.data.groupby(group_cols).agg(agg_dict).reset_index()

        return grouped

    def _set_up_areas(self):
        self.data["pos_string"] = self.data.apply(
            lambda row: "_".join([str(row.LONGI_DD), str(row.LATIT_DD)]), axis=1
        )
        # self.data["area_tag"] = self.data.apply(
        #     lambda row: "_".join([str(row[tag]) for tag in area_tags]), axis=1
        # )
        self._map_areas_to_pos_str()

        self.areas = self.data["area_tag"].unique()

    def _map_areas_to_pos_str(self):
        # Välj endast kolumnerna 'pos_string' och 'area_tag' från area_tags_df
        area_tags_filtered = self.area_tags_df[["pos_string", "area_tag"]]

        # Slå ihop dataframes baserat på kolumnen 'pos_string'
        self.data = pd.merge(
            self.data,
            area_tags_filtered,
            on="pos_string",
            how="left",
            suffixes=("_old", ""),
        )

    def _filter_area(self, area):
        return self.data["area_tag"] == area

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
        group_cols = ["depth", "month", "area_tag"]

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

        if save:
            self._save_statistic_files(grouped)

    def _save_statistic_files(self, data, column_name="area_tag", file_format="csv"):
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
    sea_basin = regions.sea_basin_for_position(longitude, latitude)
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
        }

    # Extract relevant columns, handling missing values if necessary
    mean_values = filtered_df[f"{parameter}:mean"].apply(nan_float).tolist()
    std_values = filtered_df[f"{parameter}:std"].apply(nan_float).tolist()
    depth = filtered_df["depth"].apply(nan_float).tolist()

    for i, values in enumerate(zip(
        mean_values,
        std_values,
    )):
        # Check if any value is np.nan
        if any(np.isnan(value) for value in values):
            # Set all values at this index to np.nan
            mean_values[i] = np.nan
            std_values[i] = np.nan

    return {
        "mean": mean_values,
        "lower_limit": [
            mean_value - std_value
            for mean_value, std_value in zip(mean_values, std_values)
        ],
        "upper_limit": [
            mean_value + std_value
            for mean_value, std_value in zip(mean_values, std_values)
        ],
        "depth": depth,
    }


if __name__ == "__main__":
    data = DataHandler("C:/LenaV/code/data/sharkweb_data_1991-2020_for_statistics.txt"
    )
    valid_data = data.valid_data

    # geo_info = regions.read_geo_info_file(Path.home() / "SVAR2022_HELCOM_OSPAR_vs2.gpkg")
    # area_tags = regions.get_area_tags(df=data.data, geo_info=geo_info)
    # area_tags.drop_duplicates(inplace=True)
    # area_tags["pos_string"] = (
    #     area_tags["LONGI_DD"].astype(str) + "_" + area_tags["LATIT_DD"].astype(str)
    # )
    # area_tags.to_csv(
    #     "src/nodc_statistics/data/pos_area_tag_1991_2020.csv",
    #     sep="\t",
    #     index=False,
    #     encoding="utf-8",
    # )
    # print(area_tags.head())

    statistics = CalculateStatistics(valid_data)
    statistics.profile_statistics()

    get_profile_statistics_for_parameter_and_position(
        "TEMP_CTD", 10.759, 58.3050, datetime.datetime(2024, 5, 16)
    )
