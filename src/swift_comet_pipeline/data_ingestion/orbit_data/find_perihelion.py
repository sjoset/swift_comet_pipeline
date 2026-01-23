from astropy.time import Time
from astroquery.jplhorizons import Horizons

from swift_comet_pipeline.pipeline.product_system.product_facade import Products
from swift_comet_pipeline.scp_types.primitive import *


# TODO: clean up, document, and decouple from scp
def find_perihelia(scp: Products) -> list[OrbitPerihelion] | None:
    """
    Query JPL Horizons (via astroquery) for perihelion(s) within [t_start, t_end],
    returning a list of OrbitPerihelion or None on failure.

    Behavior
    --------
    - Aggregation is MEDIAN for both Tp and q within groups of repeated Tp values
      (grouped by `unique_tol_days` in JD).
    - Returns [] if no perihelion lies inside the window (unless `return_nearest_if_empty`
      is True, in which case the single nearest perihelion to the interval is returned).
    - Returns None on any failure (e.g., network/astroquery error, unexpected schema).

    Returns None on failure; else a (possibly empty) list of perihelia.
    """

    obs_log_df = scp.load_obs_log()
    if obs_log_df is None:
        return None

    time_start = Time(np.min(obs_log_df.MID_TIME))
    time_end = Time(np.max(obs_log_df.MID_TIME))
    horizons_id = scp.cfg.jpl_horizons_id
    return_nearest_if_empty = True
    perihelion_time_column = "Tp_jd"

    # print(f"{time_start=}\t{time_end=}")

    step = "1d"
    unique_tol_days: float = 1

    if time_end < time_start:
        # TODO: remove or log this
        print("End of window is before the start!")
        return None

    obj = Horizons(
        id=horizons_id,
        location=None,
        epochs={"start": time_start.utc.isot, "stop": time_end.utc.isot, "step": step},  # type: ignore
    )
    tbl = obj.elements()  # type: ignore
    if len(tbl) == 0:
        return []

    # # --- Extract Tp (JD TDB) ---
    # if "Tp_jd" in tbl.colnames:
    #     tp_jd_all = np.asarray(tbl["Tp_jd"], dtype=float)
    # elif "Tp" in tbl.colnames:
    #     tp_jd_all = np.asarray(tbl["Tp"], dtype=float)
    # elif "Tp_str" in tbl.colnames:
    #     tp_jd_all = Time(np.asarray(tbl["Tp_str"], dtype=str), scale="utc").tdb.jd  # type: ignore
    # else:
    #     return None

    if perihelion_time_column in tbl.colnames:
        tp_jd_all = np.asarray(tbl[perihelion_time_column], dtype=float)
    else:
        return None

    # --- Extract q (AU) ---
    if "q" not in tbl.colnames:
        return None  # schema failure
    q_all = np.asarray(tbl["q"], dtype=float)

    # --- Sort and group near-identical Tp values; aggregate with MEDIAN ---
    order = np.argsort(tp_jd_all)
    tp_sorted, q_sorted = tp_jd_all[order], q_all[order]

    groups_tp, groups_q = [], []
    if tp_sorted.size:
        run_tp, run_q = [tp_sorted[0]], [q_sorted[0]]
        for jd, qv in zip(tp_sorted[1:], q_sorted[1:]):
            if abs(jd - run_tp[-1]) <= unique_tol_days:
                run_tp.append(jd)
                run_q.append(qv)
            else:
                groups_tp.append(float(np.median(run_tp)))
                groups_q.append(float(np.median(run_q)))
                run_tp, run_q = [jd], [qv]
        groups_tp.append(float(np.median(run_tp)))
        groups_q.append(float(np.median(run_q)))

    tp_unique_jd = np.asarray(groups_tp)
    q_by_group = np.asarray(groups_q)
    if tp_unique_jd.size == 0:
        return []

    t0_tdb, t1_tdb = time_start.tdb, time_end.tdb

    # --- Select those whose Tp lies inside the window ---
    inside = (tp_unique_jd >= t0_tdb.jd) & (tp_unique_jd <= t1_tdb.jd)  # type: ignore
    tp_in, q_in = tp_unique_jd[inside], q_by_group[inside]

    def _pack(jd_arr: np.ndarray, q_arr: np.ndarray) -> list[OrbitPerihelion]:
        times_tdb = [Time(jd, format="jd", scale="utc") for jd in jd_arr]
        times_out = [t.utc for t in times_tdb]
        return [
            OrbitPerihelion(t_perihelion=Time(t), rh_au=float(q))
            for t, q in zip(times_out, q_arr)
        ]

    if tp_in.size:
        return _pack(tp_in, q_in)

    # --- If none inside, optionally return the closest perihelion to the interval ---
    if return_nearest_if_empty:
        a, b = t0_tdb.jd, t1_tdb.jd  # type: ignore
        dist = np.where(
            tp_unique_jd < a,
            a - tp_unique_jd,
            np.where(tp_unique_jd > b, tp_unique_jd - b, 0.0),
        )
        i = int(np.argmin(dist))
        return _pack(tp_unique_jd[i : i + 1], q_by_group[i : i + 1])

    return []


# TODO: remove
# def find_perihelion_old(
#     scp: Products,
#     t_start_search: Time | None = None,
#     t_end_search: Time | None = None,
# ) -> list[OrbitPerihelion] | None:
#
#     # either they're both None,
#     # or both not None
#     assert (t_start_search == t_end_search) or (
#         t_start_search is not None and t_end_search is not None
#     )
#
#     comet_df = scp.load_comet_orbit_data()
#
#     comet_df["DATE_OBS"] = comet_df["datetime_jd"].apply(lambda x: Time(x, format="jd"))
#
#     # filter the dataframe to the time limits specified
#     if t_start_search is not None:
#         t_start_mask = comet_df.DATE_OBS > t_start_search
#         t_end_mask = comet_df.DATE_OBS < t_end_search
#         t_mask = np.logical_and(t_start_mask, t_end_mask)
#         comet_df = comet_df[t_mask]
#     else:
#         comet_df = comet_df
#
#     # TODO: find multiple minima in comet_df.range and return a list of OrbitPerihelion based on this
#     range_min_idx = np.argmin(comet_df.range)
#     light_min_idx = np.argmin(comet_df.lighttime)
#
#     # TODO: what if the actual perihelion is not in the time range of our dataframe?
#
#     assert range_min_idx == light_min_idx
#
#     # for now, return the minimum we found instead of searching for a bunch of local minima
#     return [
#         OrbitPerihelion(
#             t_perihelion=Time(comet_df.iloc[range_min_idx].DATE_OBS),
#             r_h=comet_df.iloc[range_min_idx].range * u.AU,  # type: ignore
#         )
#     ]
