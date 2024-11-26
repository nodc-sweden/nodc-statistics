import os
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

GEOLAYERS_AREATAG = {
    "SVAR2022_typomrkust_lagad": "TYPOMRKUST",
    "ospar_subregions_20160418_3857_lagad": "area_tag",
    "helcom_subbasins_with_coastal_and_offshore_division_2022_level3_lagad": "level_34",
}

AREA_TAG_FILE = Path(__file__).parent / "data" / "pos_area_tag_1991_2020.csv"

# @functools.cache
# cache needs all argumetns to be hashable, GeoDataFrame is not
def sea_basin_for_position(longitude, latitude, geo_info=None):
    if not all((-180 <= longitude <= 180, -90 <= latitude <= 90)):
        return None
    point = pd.DataFrame({"LONGI_DD": [longitude], "LATIT_DD": [latitude]})

    if not isinstance(geo_info, gpd.GeoDataFrame):
        print("reading again in regions.sea_basin_for_position")
        geo_info = read_geo_info_file(Path.home() / "SVAR2022_HELCOM_OSPAR_vs2.gpkg")
    area_tag_df = get_area_tags(df=point, geo_info=geo_info)

    if len(area_tag_df["area_tag"].values) > 1:
        print(f'too many area_tag results {area_tag_df["area_tag"]}')
    value = area_tag_df["area_tag"].values[0] or None
    return value if not pd.isna(value) else None


def read_geo_info_file(filepath: str):
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


def save_area_tag_files(data, column_name="area_tag", file_format="csv"):
    # Gruppér DataFrame efter `area_tag`
    grouped = data.groupby(column_name)

    # Spara varje grupp i en separat fil
    for area_tag, group in grouped:
        # Skapa ett filnamn baserat på area_tag
        file_name = f"{area_tag}.{file_format}"

        # Spara gruppen till en fil
        group.to_csv(file_name, index=False)


if __name__ == "__main__":
    save_kwargs = {
        "sep": "\t",
        "encoding": "utf-8",
        "index": False,
        "float_format": "%.2f",
    }

    df = pd.read_csv(
        open(
            "src/nodc_statistics/data/"
            "sharkweb_all_data_1991-2020_Physical and Chemical_1991-2020.csv",
            encoding="utf-8",
        ),
        sep="\t",
    )

    area_tags = get_area_tags(df)
    area_tags.drop_duplicates(inplace=True)
    area_tags["pos_string"] = (
        area_tags["LONGI_DD"].astype(str) + "_" + area_tags["LATIT_DD"].astype(str)
    )

    # area_tags.to_csv(
    #     "src/nodc_statistics/data/pos_area_tag_1991_2020.csv",
    #     sep="\t",
    #     index=False,
    #     encoding="utf-8",
    # )
    print(area_tags.head())
