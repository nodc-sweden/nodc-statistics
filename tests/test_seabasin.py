import pytest

from nodc_statistics import regions


@pytest.mark.parametrize(
    "given_latitude, given_longitude, expected_sea_basin",
    (
        (58.3050, 10.759, "Skagerrak"),
        (57.19, 11.65, "Kattegat"),
        (55.87, 12.74837, "Kattegat"),
        (65.27, 23.40, "Baltic Sea"),
        (59.93, 27.43, "Baltic Sea"),
        (55.02, 13.3008, "Baltic Sea"),
    ),
)
def test_get_correct_sea_basin_for_position(
    given_latitude, given_longitude, expected_sea_basin
):
    sea_basin = regions.sea_basin_for_position(given_longitude, given_latitude)
    assert sea_basin == expected_sea_basin
