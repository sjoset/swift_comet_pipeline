from astropy.time import Time, TimeDelta
from astroquery.jplhorizons import Horizons

from swift_comet_pipeline.scp_types.primitive import *


# # TODO: remove old code
# def find_perihelia_old(scp: Products) -> list[OrbitPerihelion] | None:
#     obs_log_df = scp.load_obs_log()
#     if obs_log_df is None:
#         return None
#
#     time_start: Time = Time(np.min(obs_log_df.MID_TIME))
#     time_stop: Time = Time(np.max(obs_log_df.MID_TIME))
#     horizons_id = scp.cfg.jpl_horizons_id
#
#     if time_stop < time_start:
#         print("End of perihelion time window is before the start! Aborting.")
#         exit(1)
#
#     time_window_size = TimeDelta(time_stop - time_start)
#     time_ref = time_start + time_window_size / 2
#
#     horizons = Horizons(
#         id=horizons_id, id_type="designation", location=None, epochs=time_ref.tdb.jd  # type: ignore
#     )
#
#     elements = horizons.elements(closest_apparition=True, no_fragments=True)  # type: ignore
#     response_row = elements[0]
#
#     perihelion_time_column = "Tp"
#     perihelion_time_column_fallback = "Tp_jd"
#     tp_jd = (
#         response_row[perihelion_time_column]
#         if perihelion_time_column in response_row.colnames
#         else response_row[perihelion_time_column_fallback]
#     )
#     t_perihelion = Time(float(tp_jd), format="jd", scale="tdb")
#
#     perihelion_rh_au_column = "q"
#     rh_au = response_row[perihelion_rh_au_column]
#
#     return [OrbitPerihelion(t_perihelion=t_perihelion, rh_au=rh_au)]


# helper function for closest_perihelion_to_window()
def _infer_period(orbital_elements_row) -> u.Quantity | None:
    semimajor_axis_column = "a"
    if semimajor_axis_column in orbital_elements_row.colnames:
        semimajor_axis_au = float(orbital_elements_row[semimajor_axis_column])
        if np.isfinite(semimajor_axis_au) and semimajor_axis_au > 0:
            return (semimajor_axis_au**1.5) * u.year  # type: ignore
        else:
            return None

    # if semimajor axis is not included, look for the period in other columns
    # TODO:
    # assumes these columns are expressed in days (include in function documentation)
    possible_period_column_names = ["P", "period", "P_days", "P_d"]
    for p_colname in possible_period_column_names:
        if p_colname in orbital_elements_row.colnames:
            period_days = float(orbital_elements_row[p_colname])
            if np.isfinite(period_days) and period_days > 0:
                return period_days * u.day  # type: ignore

    return None


def _window_distance_from_perihelion(
    t_perihelion: Time, time_start: Time, time_stop: Time
) -> TimeDelta:

    tp_jd, start_jd, stop_jd = t_perihelion.tdb.jd, time_start.tdb.jd, time_stop.tdb.jd  # type: ignore
    if start_jd <= tp_jd <= stop_jd:
        return TimeDelta(0 * u.s)

    if abs(tp_jd - start_jd) < abs(tp_jd - stop_jd):
        delta_t = tp_jd - start_jd
    else:
        delta_t = tp_jd - stop_jd

    return TimeDelta(val=delta_t * u.day)  # type: ignore


# helper function for find_perihelia_new
def closest_perihelion_to_window(
    horizons_id: str, time_start: Time, time_stop: Time
) -> OrbitPerihelion | None:

    # midpoint of window
    time_window_size = (time_stop - time_start).to_value(u.day)  # type: ignore
    time_mid = time_start + (time_window_size * u.day) / 2  # type: ignore

    # query
    horizons = Horizons(
        id=horizons_id, id_type="designation", location=None, epochs=time_mid.tdb.jd
    )
    elements = horizons.elements(closest_apparition=True, no_fragments=True)  # type: ignore
    response_row = elements[0]

    # pull out perihelion time and heliocentric distance
    perihelion_time_column = "Tp"
    perihelion_time_column_fallback = "Tp_jd"
    tp_jd = (
        response_row[perihelion_time_column]
        if perihelion_time_column in response_row.colnames
        else response_row[perihelion_time_column_fallback]
    )
    t_perihelion = Time(float(tp_jd), format="jd", scale="tdb")

    perihelion_rh_au_column = "q"
    rh_au = response_row[perihelion_rh_au_column]

    initial_perihelion_solution = OrbitPerihelion(
        t_perihelion=t_perihelion, rh_au=rh_au
    )

    orbital_period = _infer_period(orbital_elements_row=response_row)
    # if we don't have an orbital period, just take Horizons answer: it should be the only solution
    if orbital_period is None:
        return initial_perihelion_solution

    # express the time difference between our observations and the initial perihelion result as a number of periods
    t_prime = int(
        np.rint(
            ((time_mid.tdb.jd - initial_perihelion_solution.t_perihelion.tdb.jd) * u.s)  # type: ignore
            / orbital_period
        )
    )

    candidate_list = []
    # search forward and backward in time by this many periods to see if this is a better solution
    number_of_search_periods = 2
    possible_perihelia_times = np.linspace(
        start=t_prime - number_of_search_periods,
        stop=t_prime + number_of_search_periods,
        num=2 * number_of_search_periods + 1,
        endpoint=True,
    )
    for k in possible_perihelia_times:
        new_time = Time(
            initial_perihelion_solution.t_perihelion.tdb.jd + t_prime * k,  # type: ignore
            format="jd",
            scale="tdb",
        )
        d = _window_distance_from_perihelion(
            t_perihelion=new_time, time_start=time_start, time_stop=time_stop
        )
        new_candidate = (
            d,
            k,
            OrbitPerihelion(
                t_perihelion=new_time, rh_au=initial_perihelion_solution.rh_au
            ),
        )
        candidate_list.append(new_candidate)

    # look for the minimum distance stored in element 0 of the tuple
    # if that is the same, we compare the second elements and choose by smaller |k|, the number of periods shifted
    best_candidate = min(candidate_list, key=lambda x: (x[0], abs(x[1])))

    # pull out the OrbitPerihelion in the third element of the tuple
    best_orbit_perihelion = best_candidate[2]
    return best_orbit_perihelion


def refine_perihelion_with_ephemerides(
    horizons_id: str,
    time_start: Time,
    time_stop: Time,
    search_time_step_size: str = "1h",
) -> OrbitPerihelion | None:
    epochs_dict = {
        "start": time_start.iso,
        "stop": time_stop.iso,
        "step": search_time_step_size,
    }
    horizons = Horizons(
        id=horizons_id, id_type="designation", location=None, epochs=epochs_dict
    )
    eph = horizons.ephemerides(closest_apparition=True, no_fragments=True)  # type: ignore

    r_column_name = "r"
    time_column_name = "datetime_jd"

    rs = np.asarray(eph[r_column_name], dtype=float)
    ts = np.asarray(eph[time_column_name], dtype=float)

    min_r_index = np.nanargmin(rs)

    r_perihelion = rs[min_r_index]
    t_perihelion = Time(ts[min_r_index], format="jd", scale="utc")

    return OrbitPerihelion(t_perihelion=t_perihelion, rh_au=r_perihelion)


def find_perihelion(
    horizons_id: str, time_start: Time, time_stop: Time
) -> OrbitPerihelion | None:
    """
    Takes an epoch observing window and finds the perihelion closest to the observation time window
    """

    # find JPL's solution for perihelion given the observing window
    perihelion_base = closest_perihelion_to_window(
        horizons_id=horizons_id, time_start=time_start, time_stop=time_stop
    )
    if not perihelion_base:
        return None

    # now look at ephemera for +/- 1 day around perihelion guess and get a more exact solution
    dt = 1 * u.day  # type: ignore
    refine_window_start = Time(perihelion_base.t_perihelion - dt)
    refine_window_stop = Time(perihelion_base.t_perihelion + dt)
    refined_perihelion_result = refine_perihelion_with_ephemerides(
        horizons_id=horizons_id,
        time_start=refine_window_start,
        time_stop=refine_window_stop,
    )

    return refined_perihelion_result


# # TODO: clean up, document, and decouple from scp
# def find_perihelia(scp: Products) -> list[OrbitPerihelion] | None:
#     """
#     Query JPL Horizons (via astroquery) for perihelion(s) within [t_start, t_end],
#     returning a list of OrbitPerihelion or None on failure.
#
#     Behavior
#     --------
#     - Aggregation is MEDIAN for both Tp and q within groups of repeated Tp values
#       (grouped by `unique_tol_days` in JD).
#     - Returns [] if no perihelion lies inside the window (unless `return_nearest_if_empty`
#       is True, in which case the single nearest perihelion to the interval is returned).
#     - Returns None on any failure (e.g., network/astroquery error, unexpected schema).
#
#     Returns None on failure; else a (possibly empty) list of perihelia.
#     """
#
#     obs_log_df = scp.load_obs_log()
#     if obs_log_df is None:
#         return None
#
#     time_start = Time(np.min(obs_log_df.MID_TIME))
#     time_end = Time(np.max(obs_log_df.MID_TIME))
#     horizons_id = scp.cfg.jpl_horizons_id
#     return_nearest_if_empty = True
#     perihelion_time_column = "Tp_jd"
#
#     step = "1d"
#     unique_tol_days: float = 1
#
#     if time_end < time_start:
#         # TODO: remove or log this
#         print("End of window is before the start!")
#         return None
#
#     obj = Horizons(
#         id=horizons_id,
#         location=None,
#         epochs={"start": time_start.utc.isot, "stop": time_end.utc.isot, "step": step},  # type: ignore
#     )
#     tbl = obj.elements()  # type: ignore
#     if len(tbl) == 0:
#         return []
#
#     # # --- Extract Tp (JD TDB) ---
#     # if "Tp_jd" in tbl.colnames:
#     #     tp_jd_all = np.asarray(tbl["Tp_jd"], dtype=float)
#     # elif "Tp" in tbl.colnames:
#     #     tp_jd_all = np.asarray(tbl["Tp"], dtype=float)
#     # elif "Tp_str" in tbl.colnames:
#     #     tp_jd_all = Time(np.asarray(tbl["Tp_str"], dtype=str), scale="utc").tdb.jd  # type: ignore
#     # else:
#     #     return None
#
#     if perihelion_time_column in tbl.colnames:
#         tp_jd_all = np.asarray(tbl[perihelion_time_column], dtype=float)
#     else:
#         return None
#
#     # --- Extract q (AU) ---
#     if "q" not in tbl.colnames:
#         return None  # schema failure
#     q_all = np.asarray(tbl["q"], dtype=float)
#
#     # --- Sort and group near-identical Tp values; aggregate with MEDIAN ---
#     order = np.argsort(tp_jd_all)
#     tp_sorted, q_sorted = tp_jd_all[order], q_all[order]
#
#     groups_tp, groups_q = [], []
#     if tp_sorted.size:
#         run_tp, run_q = [tp_sorted[0]], [q_sorted[0]]
#         for jd, qv in zip(tp_sorted[1:], q_sorted[1:]):
#             if abs(jd - run_tp[-1]) <= unique_tol_days:
#                 run_tp.append(jd)
#                 run_q.append(qv)
#             else:
#                 groups_tp.append(float(np.median(run_tp)))
#                 groups_q.append(float(np.median(run_q)))
#                 run_tp, run_q = [jd], [qv]
#         groups_tp.append(float(np.median(run_tp)))
#         groups_q.append(float(np.median(run_q)))
#
#     tp_unique_jd = np.asarray(groups_tp)
#     q_by_group = np.asarray(groups_q)
#     if tp_unique_jd.size == 0:
#         return []
#
#     t0_tdb, t1_tdb = time_start.tdb, time_end.tdb
#
#     # --- Select those whose Tp lies inside the window ---
#     inside = (tp_unique_jd >= t0_tdb.jd) & (tp_unique_jd <= t1_tdb.jd)  # type: ignore
#     tp_in, q_in = tp_unique_jd[inside], q_by_group[inside]
#
#     def _pack(jd_arr: np.ndarray, q_arr: np.ndarray) -> list[OrbitPerihelion]:
#         times_tdb = [Time(jd, format="jd", scale="utc") for jd in jd_arr]
#         times_out = [t.utc for t in times_tdb]
#         return [
#             OrbitPerihelion(t_perihelion=Time(t), rh_au=float(q))
#             for t, q in zip(times_out, q_arr)
#         ]
#
#     if tp_in.size:
#         return _pack(tp_in, q_in)
#
#     # --- If none inside, optionally return the closest perihelion to the interval ---
#     if return_nearest_if_empty:
#         a, b = t0_tdb.jd, t1_tdb.jd  # type: ignore
#         dist = np.where(
#             tp_unique_jd < a,
#             a - tp_unique_jd,
#             np.where(tp_unique_jd > b, tp_unique_jd - b, 0.0),
#         )
#         i = int(np.argmin(dist))
#         return _pack(tp_unique_jd[i : i + 1], q_by_group[i : i + 1])
#
#     return []


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
