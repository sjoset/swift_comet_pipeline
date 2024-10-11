from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class QvsApertureRadiusEntry:
    aperture_r_pix: float
    aperture_r_km: float
    dust_redness: float
    counts_uw1: float
    counts_uw1_err: float
    snr_uw1: float
    counts_uvv: float
    counts_uvv_err: float
    snr_uvv: float
    magnitude_uw1: float
    magnitude_uw1_err: float
    magnitude_uvv: float
    magnitude_uvv_err: float
    flux_OH: float
    flux_OH_err: float
    num_OH: float
    num_OH_err: float
    q_H2O: float
    q_H2O_err: float
    # TODO:
    # q_H2O_vmag_empirical: float


def q_vs_aperture_radius_entry_list_from_dataframe(
    df: pd.DataFrame,
) -> list[QvsApertureRadiusEntry]:
    return df.apply(lambda row: QvsApertureRadiusEntry(**row), axis=1).to_list()


def dataframe_from_q_vs_aperture_radius_entry_list(
    q_vs_r: list[QvsApertureRadiusEntry],
) -> pd.DataFrame:
    return pd.DataFrame(data=[asdict(qvsar) for qvsar in q_vs_r])
