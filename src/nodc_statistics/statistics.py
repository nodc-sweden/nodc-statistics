import csv
import datetime
from pathlib import Path

from nodc_statistics import regions

STATISTIC_FILES = {
    "Skagerrak": Path(__file__).parent / "data" / "skagerrak.csv",
    "Kattegat": Path(__file__).parent / "data" / "kattegat.csv",
    "Baltic Sea": Path(__file__).parent / "data" / "baltic_sea.csv"
}


def get_profile_statistics_for_parameter_and_position(
    parameter, x_position, y_position, point_in_time: datetime.datetime
):
    sea_basin = regions.sea_basin_for_position(x_position, y_position)
    statistic_path = STATISTIC_FILES[sea_basin]

    with statistic_path.open() as csv_file:
        reader = csv.DictReader(csv_file, delimiter="\t")
        mean_values = []
        std_values = []
        depth = []
        for row in reader:
            if int(row["MONTH"]) == point_in_time.month:
                mean_values.append(float(row[f"{parameter}:mean"]))
                std_values.append(float(row[f"{parameter}:std"]))
                depth.append(float(row["DEPH"]))

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
