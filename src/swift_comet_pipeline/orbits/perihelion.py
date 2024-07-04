from dataclasses import dataclass

import numpy as np
from astropy.time import Time
import astropy.units as u

from swift_comet_pipeline.pipeline.files.data_ingestion_files import DataIngestionFiles


@dataclass
class OrbitPerihelion:
    t_perihelion: Time
    r_h: u.Quantity


def find_perihelion(
    data_ingestion_files: DataIngestionFiles,
    t_start_search: Time | None = None,
    t_end_search: Time | None = None,
) -> list[OrbitPerihelion] | None:

    # either they're both None,
    # or both not None
    assert (t_start_search == t_end_search) or (
        t_start_search is not None and t_end_search is not None
    )

    if not data_ingestion_files.comet_orbital_data.exists():
        print("No comet orbital data found!")
        return None

    if data_ingestion_files.comet_orbital_data.data is None:
        data_ingestion_files.comet_orbital_data.read()
    raw_comet_df = data_ingestion_files.comet_orbital_data.data

    if raw_comet_df is None:
        print("Couldn't read comet orbital data!")
        return None

    raw_comet_df["DATE_OBS"] = raw_comet_df["datetime_jd"].apply(
        lambda x: Time(x, format="jd")
    )

    # filter the dataframe to the time limits specified
    if t_start_search is not None:
        t_start_mask = raw_comet_df.DATE_OBS > t_start_search
        t_end_mask = raw_comet_df.DATE_OBS < t_end_search
        t_mask = np.logical_and(t_start_mask, t_end_mask)
        comet_df = raw_comet_df[t_mask]
    else:
        comet_df = raw_comet_df

    # TODO: find multiple minima in comet_df.range and return a list of OrbitPerihelion based on this
    range_min_idx = np.argmin(comet_df.range)
    light_min_idx = np.argmin(comet_df.lighttime)

    # TODO: what if the actual perihelion is not in the time range of our dataframe?

    assert range_min_idx == light_min_idx

    # for now, return the minimum we found instead of searching for a bunch of local minima
    return [
        OrbitPerihelion(
            t_perihelion=Time(comet_df.iloc[range_min_idx].DATE_OBS),
            r_h=comet_df.iloc[range_min_idx].range * u.AU,  # type: ignore
        )
    ]
