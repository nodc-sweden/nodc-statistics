def sea_basin_for_position(longitude, latitude):
    if longitude > 12.75:
        return "Baltic Sea"
    else:
        if latitude < 58:
            return "Kattegat"
        else:
            return "Skagerrak"
