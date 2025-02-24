import time
from pathlib import Path

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


# @functools.cache
# cache needs all argumetns to be hashable, GeoDataFrame is not
def sea_basin_for_position(longitude, latitude, geo_info=None):
    if not all((-180 <= longitude <= 180, -90 <= latitude <= 90)):
        return None
    point = pd.DataFrame({"LONGI_DD": [longitude], "LATIT_DD": [latitude]})
    if not isinstance(geo_info, gpd.GeoDataFrame) and GPKG_FILE.exists:
        geo_info = read_geo_info_file(GPKG_FILE)

    if isinstance(geo_info, gpd.GeoDataFrame):
        print("reading again in regions.sea_basin_for_position")
        geo_info = read_geo_info_file(GPKG_FILE)
        area_tag_df = get_area_tags(df=point, geo_info=geo_info)
        if len(area_tag_df["area_tag"].values) > 1:
            print(f'too many area_tag results {area_tag_df["area_tag"]}')
        value = area_tag_df["area_tag"].values[0] or None
    else:
        print("no geo_info, using area_tag textfile instead")
        area_tag_df = pd.read_csv(AREA_TAG_FILE, sep="\t", encoding="utf-8")
        # Create pos_string from input coordinates
        pos_string = f"{longitude}_{latitude}"

        # Look for a match in the DataFrame
        match = area_tag_df.loc[area_tag_df["pos_string"] == pos_string, "area_tag"]

        # Return the matched area_tag or None
        value = match.iloc[0] if not match.empty else None
        if value is None:
            print(f"no information for longitude: {longitude}, latitude: {latitude}")

    return value if not pd.isna(value) else None


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


def get_area_tags(df, geo_info: gpd.GeoDataFrame):
    """
    Hitta rätt "area_tag" för varje punkt i DataFrame df genom en rumslig join med
    geo_info.
    Returnera en DataFrame med kolumnerna "area_tag", "LONGI_DD" och "LATIT_DD".
    """
    # Skapa en geopandas GeoDataFrame med punkter från df
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df["LONGI_DD"], df["LATIT_DD"]), crs="EPSG:4326"
    )

    # Kontrollera om CRS för geo_info matchar punkternas CRS
    if geo_info.crs != points.crs:
        # Transformera punkternas CRS till matchande CRS
        points = points.to_crs(geo_info.crs)

    # Använd geopandas sjoin för att göra en rumslig join mellan punkterna och polygonerna
    # i geo_info. Med inner kommer endast de punkter som har en match i geo_info med
    joined = gpd.sjoin(points, geo_info, how="inner", predicate="within")

    # Lägg till "LONGI_DD" och "LATIT_DD" från points till joined
    joined["LONGI_DD"] = df["LONGI_DD"]
    joined["LATIT_DD"] = df["LATIT_DD"]

    # Kontrollera om det finns flera matchningar för samma punkt
    duplicate_check = (
        joined.groupby(["LONGI_DD", "LATIT_DD"])
        .agg(
            count_areatags=("area_tag", pd.Series.nunique),
            unique_areatags=("area_tag", "unique"),
        )
        .reset_index()
    )
    multiple_matches = duplicate_check[duplicate_check["count_areatags"] > 1]

    if not multiple_matches.empty:
        print("Varning: Följande punkter har flera matchande polygoner:")
        print(multiple_matches)
        raise ValueError("punkt matchar mot flera områden")
        # Hantera flera matchningar, t.ex. genom att välja den första matchningen
        joined = joined.drop_duplicates(subset=["LONGI_DD", "LATIT_DD"], keep="first")

    # Välj önskade kolumner och returnera som en DataFrame
    if not joined.empty:
        result_df = joined[["area_tag", "LONGI_DD", "LATIT_DD"]].copy()
        return result_df
    else:
        # Om inga matchningar hittades, returnera en tom DataFrame med rätt kolumner
        return pd.DataFrame(columns=["area_tag", "LONGI_DD", "LATIT_DD"])


def update_area_tag_file(coordinates_filepath):
    data = pd.read_csv(coordinates_filepath)
    data.columns = ["LONGI_DD", "LATIT_DD"]
    geo_info = read_geo_info_file(Path.home() / "SVAR2022_HELCOM_OSPAR_vs2.gpkg")  # noqa: E501
    area_tags = get_area_tags(df=data, geo_info=geo_info)
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
    # update_area_tag_file("C:/LenaV/code/data/coordinates.txt")
    read_geo_info_file(GPKG_FILE)
