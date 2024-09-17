import os

import geopandas as gpd
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
    ),
)
def test_get_correct_sea_basin_for_position(
    given_latitude, given_longitude, expected_sea_basin
):
    sea_basin = regions.sea_basin_for_position(given_longitude, given_latitude)
    assert sea_basin == expected_sea_basin

@pytest.mark.parametrize(
    "given_path", ((os.environ["QCTOOL_GEOPACKAGE"],))
    )
def test_read_geopackage_to_geodataframe(given_path):

    geo_info = regions.read_geo_info_file(given_path)

    assert isinstance(geo_info, gpd.GeoDataFrame)
