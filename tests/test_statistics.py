import datetime

import numpy as np
import pytest

from nodc_statistics import statistic


def test_get_profile_statistics_for_parameter_returns_dictionary():
    # Given a parameter
    given_parameter = "DOXY_CTD"

    # Given a position
    given_longitude = 11.619563447195599
    given_latitude = 57.63218414832167

    # Given a datetime
    given_datetime = datetime.datetime(2024, 5, 16, 9, 32)

    # When retrieving statistics
    statistics_object = statistic.get_profile_statistics_for_parameter_and_position(
        given_parameter, given_longitude, given_latitude, given_datetime
    )

    # Then a dictionary is returned
    assert isinstance(statistics_object, dict)


def test_get_profile_statistics_for_nonexistent_sea_basin():
    given_sea_basin = "incorrect_basin_name"

    given_parameter = "DOXY_CTD"

    # Given a datetime
    given_datetime = datetime.datetime(2024, 5, 16, 9, 32)

    # When retrieving statistics
    statistics_object = statistic.get_profile_statistics_for_parameter_and_sea_basin(
        given_parameter, given_sea_basin, given_datetime
    )

    # Then a dictionary is returned
    assert isinstance(statistics_object, dict)


def test_get_profile_statistics_for_nonexistent_parameter():
    given_sea_basin = "Skagerrak"

    given_parameter = "unknown_param"

    # Given a datetime
    given_datetime = datetime.datetime(2024, 5, 16, 9, 32)

    # When retrieving statistics
    statistics_object = statistic.get_profile_statistics_for_parameter_and_sea_basin(
        given_parameter, given_sea_basin, given_datetime
    )

    # Then a dictionary is returned
    assert isinstance(statistics_object, dict)


def test_profile_statistics_object_has_mean_lower_limit_and_upper_limit():
    # Given a parameter
    given_parameter = "DOXY_CTD"

    # Given a position
    given_longitude = 11.619563447195599
    given_latitude = 57.63218414832167

    # Given a datetime
    given_datetime = datetime.datetime(2024, 5, 16, 9, 32)

    # When retrieving statistics
    statistics_object = statistic.get_profile_statistics_for_parameter_and_position(
        given_parameter, given_longitude, given_latitude, given_datetime
    )

    # Then the object has a list of mean values
    assert "mean" in statistics_object
    assert isinstance(statistics_object["mean"], list)

    # And the object has a  list of lower limit values
    assert "lower_limit" in statistics_object
    assert isinstance(statistics_object["lower_limit"], list)

    # And the object has an  list of upper limit values
    assert "upper_limit" in statistics_object
    assert isinstance(statistics_object["upper_limit"], list)

    assert "depth" in statistics_object
    assert isinstance(statistics_object["depth"], list)


@pytest.mark.parametrize(
    "given_parameter, given_latitude, given_longitude, given_datetime, expected_range",
    (
        ("TEMP_CTD", 58.3050, 10.759, datetime.datetime(2024, 5, 16), (-2, 30)),
        ("DOXY_CTD", 58.3050, 10.759, datetime.datetime(2024, 5, 16), (2, 11)),
    ),
)
def test_mean_values_in_expected_range(
    given_parameter, given_latitude, given_longitude, given_datetime, expected_range
):
    statistics_object = statistic.get_profile_statistics_for_parameter_and_position(
        given_parameter, given_longitude, given_latitude, given_datetime
    )
    lower_bound, upper_bound = expected_range
    assert statistics_object["mean"]
    assert all(
        lower_bound < value < upper_bound
        for value in statistics_object["mean"]
        if ~np.isnan(value)
    )


@pytest.mark.parametrize(
    "given_parameter, given_latitude, given_longitude, given_datetime",
    (
        ("TEMP_CTD", 57.8305, 10.5685, datetime.datetime(2024, 5, 16)),
        ("DOXY_CTD", 57.8305, 10.5685, datetime.datetime(2024, 5, 16)),
    ),
)
def test_mean_is_between_lower_and_upper_limit(
    given_parameter, given_latitude, given_longitude, given_datetime
):
    statistics_object = statistic.get_profile_statistics_for_parameter_and_position(
        given_parameter, given_longitude, given_latitude, given_datetime
    )
    assert statistics_object["mean"]
    assert statistics_object["lower_limit"]
    assert statistics_object["upper_limit"]

    for lower_limit, mean_value, upper_limit in zip(
        statistics_object["lower_limit"],
        statistics_object["mean"],
        statistics_object["upper_limit"],
    ):
        if not np.isnan(mean_value):
            assert lower_limit <= mean_value <= upper_limit, (
                f"Mean value {mean_value} is not between lower limit {lower_limit} and "
                f"upper limit {upper_limit}"
            )


@pytest.mark.parametrize(
    "given_latitude, given_longitude",
    (
        (58.3050, 10.759),
        (57.19, 11.65),
    ),
)
def test_depth_value_is_increasing(given_latitude, given_longitude):
    given_parameter = "DOXY_CTD"
    given_datetime = datetime.datetime(2024, 5, 15, 16, 16)

    # When getting statistics
    statistics_object = statistic.get_profile_statistics_for_parameter_and_position(
        given_parameter, given_longitude, given_latitude, given_datetime
    )

    # Then there are depth values
    assert statistics_object["depth"]

    # And the depth values are all unique
    assert len(statistics_object["depth"]) == len(set(statistics_object["depth"]))

    # And the values are increasing
    assert statistics_object["depth"] == sorted(statistics_object["depth"])


@pytest.mark.parametrize(
    "given_parameter",
    (
        "TEMP_CTD",
        "DOXY_CTD",
    ),
)
def test_positions_in_different_sea_basins_have_different_statistics(given_parameter):
    skagerrak_latitude, skagerrak_longitude = (58.3050, 10.759)
    kattegat_latitude, kattegat_longitude = (57.19, 11.65)

    given_point_in_time = datetime.datetime(2024, 2, 16, 15, 2)

    skagerrak_statistics = statistic.get_profile_statistics_for_parameter_and_position(
        given_parameter, skagerrak_longitude, skagerrak_latitude, given_point_in_time
    )

    kattegat_statistics = statistic.get_profile_statistics_for_parameter_and_position(
        given_parameter, kattegat_longitude, kattegat_latitude, given_point_in_time
    )

    assert all(
        skagerrak_value != kattegat_value
        for skagerrak_value, kattegat_value in zip(
            skagerrak_statistics["mean"], kattegat_statistics["mean"]
        )
    )


def test_mean_surface_temperature_is_lower_in_february_than_in_august():
    february_statistics = statistic.get_profile_statistics_for_parameter_and_position(
        "TEMP_CTD", 10.759, 58.3050, datetime.datetime(2024, 2, 1, 12, 12)
    )

    august_statistics = statistic.get_profile_statistics_for_parameter_and_position(
        "TEMP_CTD", 10.759, 58.3050, datetime.datetime(2023, 8, 1, 12, 12)
    )
    print(february_statistics)
    february_surface_temperature = february_statistics["mean"][0]
    august_surface_temperature = august_statistics["mean"][0]

    assert february_surface_temperature < august_surface_temperature
