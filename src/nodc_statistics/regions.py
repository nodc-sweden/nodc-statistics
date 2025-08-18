import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import geopandas as gpd
import pandas as pd

GEOLAYERS_AREATAG = {
    "SVAR2022_typomrkust_lagad": "TYPOMRKUST",
    "ospar_subregions_20160418_3857_lagad": "area_tag",
    "helcom_subbasins_with_coastal_and_offshore_division_2022_level3_lagad": "level_34",
}

AREA_TAG_FILE = (
    Path(__file__).parent / "data" / "pos_area_tag_SVAR2022_HELCOM_OSPAR_vs2.csv"
)

GPKG_FILE = Path.home() / "SVAR2022_HELCOM_OSPAR_vs2.gpkg"


def sea_basins_for_positions(
    positions: Sequence[Tuple[float, float]], geo_info: Optional[gpd.GeoDataFrame] = None
) -> dict[str, list]:
    """
    Assign sea basin (area_tag) for a list of positions.

    Parameters
    ----------
    positions : list of (lon, lat) tuples
    geo_info : GeoDataFrame with polygons (must contain "area_tag" column).
               If None, falls back to text file lookup.

    Returns
    -------
    dict with keys: "LONGI_DD", "LATIT_DD", "sea_basin"
    """
    longis, latits, basins = [], [], []

    # validate positions
    valid_positions = [
        (lon, lat) for lon, lat in positions if -180 <= lon <= 180 and -90 <= lat <= 90
    ]
    if not valid_positions:
        return {"LONGI_DD": [], "LATIT_DD": [], "sea_basin": []}

    df = pd.DataFrame(valid_positions, columns=["LONGI_DD", "LATIT_DD"])

    # load geo_info if missing
    if not isinstance(geo_info, gpd.GeoDataFrame) and GPKG_FILE.exists():
        geo_info = read_geo_info_file(GPKG_FILE)

    if isinstance(geo_info, gpd.GeoDataFrame):
        points = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["LONGI_DD"], df["LATIT_DD"]),
            crs="EPSG:4326",
        )
        if geo_info.crs != points.crs:
            points = points.to_crs(geo_info.crs)

        joined = gpd.sjoin(points, geo_info, how="left", predicate="within")

        longis = joined["LONGI_DD"].tolist()
        latits = joined["LATIT_DD"].tolist()
        basins = joined["area_tag"].where(~joined["area_tag"].isna(), None).tolist()

    else:
        # fallback textfile lookup
        area_tag_df = pd.read_csv(AREA_TAG_FILE, sep="\t", encoding="utf-8").set_index(
            "pos_string"
        )
        pos_strings = [f"{lon}_{lat}" for lon, lat in valid_positions]
        matches = area_tag_df.reindex(pos_strings)["area_tag"]

        longis = [lon for lon, _ in valid_positions]
        latits = [lat for _, lat in valid_positions]
        basins = [val if pd.notna(val) else None for val in matches.tolist()]

    return {"LONGI_DD": longis, "LATIT_DD": latits, "sea_basin": basins}


def sea_basin_for_position(
    longitude: float, latitude: float, geo_info: Optional[gpd.GeoDataFrame] = None
) -> Optional[str]:
    seabasin = sea_basins_for_positions([(longitude, latitude)], geo_info=geo_info).get(
        "sea_basin", []
    )
    if not seabasin:  # empty list check
        return None
    return seabasin[0]


def read_geo_info_file(filepath: Path):
    """
    read a geopackage or shapefile to a geodataframe
    """

    file_path = Path(filepath)

    # Läs in specifika lager från filen
    layers = []
    for layer, area_tag in GEOLAYERS_AREATAG.items():
        print(area_tag)
        # Läs in varje lager och döp om vald kolumn till "area_tag"
        t0 = time.perf_counter()
        gdf = gpd.read_file(file_path, layer=layer)
        t1 = time.perf_counter()
        print(f"Read file took ({t1-t0:.3f} .s)")
        gdf = gdf.rename(columns={area_tag: "area_tag"})
        layers.append(gdf)

    # Kombinera lagren till en enda GeoDataFrame
    geo_info = pd.concat(layers, ignore_index=True)

    return geo_info


def update_area_tag_file(coordinates_filepath):
    data = pd.read_csv(coordinates_filepath, sep=",", encoding="utf8")
    geo_info = read_geo_info_file(Path.home() / "SVAR2022_HELCOM_OSPAR_vs2.gpkg")  # noqa: E501
    # Call new dict-returning API
    result = sea_basins_for_positions(
        positions=list(zip(data["LONGI_DD"], data["LATIT_DD"])), geo_info=geo_info
    )

    # Convert dict → pandas (or polars, depending on context)
    area_tags = pd.DataFrame(result)

    area_tags.drop_duplicates(inplace=True)
    area_tags["pos_string"] = (
        area_tags["LONGI_DD"].astype(str) + "_" + area_tags["LATIT_DD"].astype(str)
    )
    area_tags.to_csv(
        "src/nodc_statistics/data/pos_area_tag_SVAR2022_HELCOM_OSPAR_vs2.csv",
        sep="\t",
        index=False,
        encoding="utf-8",
    )
    print(area_tags.head())


if __name__ == "__main__":
    update_area_tag_file("C:/LenaV/code/data/coordinates.txt")
