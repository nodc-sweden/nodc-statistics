import numpy as np


def get_prio_par_oxy(BTL, CTD, q_prio1_par="", q_prio2_par=""):
    if not np.isnan(BTL) and q_prio1_par not in ["B"]:
        return BTL
    elif not np.isnan(CTD) and q_prio2_par not in ["B"]:
        if CTD < 0.2:
            return np.nan
        else:
            return CTD
    else:
        return np.nan


def get_prio_par(prio1_par, prio2_par, q_prio1_par="", q_prio2_par=""):
    if not np.isnan(prio1_par) and q_prio1_par not in ["B"]:
        return prio1_par
    elif q_prio2_par not in ["B"]:
        return prio2_par
    else:
        return np.nan


def get_allowed_depth_interval(deph, standard_depths):
    """
    Sätt spann för djup nära standarddjupen.
    Mindre spann i ytan och större spann ju djupare vi kommer
    """
    # TODO: use basin specific standard depths (Skagerrak and Kattegat differs from the
    #  Baltic)

    # grunda djup
    if deph < 20:
        if deph < 2.5:
            # ytvatten: returnera intervall 0-djup+2.5
            return 0, deph + 2.5
        else:
            # ej ytvatten: returnera alla mätningar i ett intervall +/- 2.5 m (5 m spann)
            return deph - 2.499, deph + 2.5
    elif deph < 100:
        # intermediära djup returnera alla mätning +/- 5 m (10 m spann)
        return deph - 4.999, deph + 5
    elif deph < 200:
        # djupvatten returnera alla mätning +/- 12.5 m (25 m spann)
        return deph - 12.499, deph + 12.5
    else:
        # djupdjupvatten returnera alla mätning +/- 50 m (100 m spann)
        return deph - 49.999, deph + 50


def map_to_standard_depth(data, standard_depths: dict):
    """
    maps sample depths in DEPH column to standard depths as defined by
    get_allowed_depth_interval
    mapped standard depths are stored in column depth
    """

    data["depth"] = data["DEPH"]
    for depth in standard_depths:
        start, stop = get_allowed_depth_interval(depth, standard_depths=standard_depths)
        boolean = (data["DEPH"] >= start) & (data["DEPH"] <= stop)
        data.loc[boolean, "depth"] = depth

    return data
