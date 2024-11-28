from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest

from nodc_statistics import regions

"""
För att köra tester:
pytest
eller
pytest {sökväga till specifik testfil}
"""


@pytest.mark.parametrize(
    "given_latitude, given_longitude, expected_sea_basin",
    (
        (58.3050, 10.759, "Skagerrak"),
        (57.19, 11.65, "Kattegat"),
        (55.87, 12.74837, "The Sound"),
        (65.27, 23.40, "Bothnian Bay"),
        (59.93, 27.43, "Gulf of Finland"),
        (55.02, 13.3008, "Arkona Basin"),
        (58.41, 15.61, None),  # Land position
        (5657.20, 1122.54, None),  # Bad position format
    ),
)
def test_get_correct_sea_basin_for_position(
    given_latitude, given_longitude, expected_sea_basin
):
    sea_basin = regions.sea_basin_for_position(given_longitude, given_latitude)
    assert sea_basin == expected_sea_basin


@pytest.mark.parametrize(
    "given_path", ((Path.home() / "SVAR2022_HELCOM_OSPAR_vs2.gpkg",))
)  # noqa: E501
def test_read_geopackage_to_geodataframe(given_path):
    geo_info = regions.read_geo_info_file(given_path)

    assert isinstance(geo_info, gpd.GeoDataFrame)


def test_sea_basin_for_position_missing_file():
    # Mock AREA_TAG_FILE DataFrame
    mock_area_tag_df = pd.DataFrame(
        {
            "pos_string": ["18.76333_59.345", "11.65_57.19"],
            "area_tag": ["TYPOMR_KOD_12n", "Kattegat"],
        }
    )

    # Mock longitude and latitude that matches the first row
    longitude = 11.65
    latitude = 57.19
    expected_area_tag = "Kattegat"

    # Mock GPKG_FILE.exists() to return False
    with (
        patch("nodc_statistics.regions.Path.exists", return_value=False),
        patch("pandas.read_csv", return_value=mock_area_tag_df),
    ):
        # Call the function
        result = regions.sea_basin_for_position(longitude, latitude)

    # Assert the result matches the expected value
    assert result == expected_area_tag


def test_sea_basin_for_position_missing_position():
    # Mock AREA_TAG_FILE DataFrame
    mock_area_tag_df = pd.DataFrame(
        {
            "pos_string": ["18.76333_59.345", "11.65_57.19"],
            "area_tag": ["TYPOMR_KOD_12n", "Kattegat"],
        }
    )

    # Mock longitude and latitude that matches the first row
    longitude = 12.65
    latitude = 57.19
    expected_area_tag = None

    # Mock GPKG_FILE.exists() to return False
    with (
        patch("nodc_statistics.regions.Path.exists", return_value=False),
        patch("pandas.read_csv", return_value=mock_area_tag_df),
    ):
        # Call the function
        result = regions.sea_basin_for_position(longitude, latitude)

    # Assert the result matches the expected value
    assert result == expected_area_tag
