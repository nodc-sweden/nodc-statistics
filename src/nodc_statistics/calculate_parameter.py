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
