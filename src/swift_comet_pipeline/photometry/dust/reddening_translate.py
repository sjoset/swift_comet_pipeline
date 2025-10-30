from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.swift.filters.filter_wavelengths import (
    effective_wavelength_of_filter_observing_solar_flux,
    pivot_wavelength_of_filter,
)


def recalculate_redness_under_filter_swap(
    known_redness: DustReddeningPercent,
    filter_to_drop: UvotFilter,
    filter_to_swap: UvotFilter,
    use_pivot: bool = False,
) -> DustReddeningPercent:

    if use_pivot:
        wave_func = pivot_wavelength_of_filter
    else:
        wave_func = effective_wavelength_of_filter_observing_solar_flux

    wave_to_drop_ang = float(wave_func(filter_to_drop).to_value(u.angstrom))  # type: ignore
    wave_to_swap_ang = float(wave_func(filter_to_swap).to_value(u.angstrom))  # type: ignore

    return recalculate_redness_under_swap(
        known_redness=known_redness,
        wavelength_to_drop_angstrom=wave_to_drop_ang,
        wavelength_to_swap_angstrom=wave_to_swap_ang,
    )


def recalculate_redness_under_swap(
    known_redness: DustReddeningPercent,
    wavelength_to_drop_angstrom: float,
    wavelength_to_swap_angstrom: float,
) -> DustReddeningPercent:

    wave_diff = wavelength_to_swap_angstrom - wavelength_to_drop_angstrom
    # print(
    #     f"Wavelength difference between dropped {wavelength_to_drop_angstrom} and swapped in {wavelength_to_swap_angstrom}: {wave_diff} angstroms"
    # )
    return known_redness / (1 + (known_redness / 200000) * wave_diff)


def recalculate_redness_under_new_filter_pair(
    known_redness: DustReddeningPercent,
    old_mid_wave_angstrom: float,
    new_mid_wave_angstrom: float,
) -> DustReddeningPercent:

    gamma = 200000

    return (gamma * known_redness) / (
        gamma + 2 * known_redness * (new_mid_wave_angstrom - old_mid_wave_angstrom)
    )


def demo_reddening_recalculation() -> None:
    """
    Use approximate data published on 3I/ATLAS reddening as measured at different wavelengths as a test
    It should roughly reproduce figure 4 of Xing et. al. 2025
    """

    import matplotlib.pyplot as plt

    rdata = {
        8000: (DustReddeningPercent(17.0), 3.0),
        7000: (DustReddeningPercent(18.0), 3.0),
        5500: (DustReddeningPercent(22.0), 3.0),
        4400: (DustReddeningPercent(28.0), 0.1),
    }

    uw1_uvv_mid_wave = (
        effective_wavelength_of_filter_observing_solar_flux(UvotFilter.uvv)
        + effective_wavelength_of_filter_observing_solar_flux(UvotFilter.uw1)
    ) / 2
    uw1_uvv_mid = float(uw1_uvv_mid_wave.to_value(u.angstrom))  # type: ignore

    translated_rednesses = [
        recalculate_redness_under_new_filter_pair(
            known_redness=z[0],
            old_mid_wave_angstrom=y,
            new_mid_wave_angstrom=uw1_uvv_mid,
        )
        for y, z in rdata.items()
    ]

    midpoint_lambdas = list(rdata.keys())
    measured_rednesses = [x[0] for x in rdata.values()]
    measured_redness_errs = [x[1] for x in rdata.values()]

    for sigma in [1, 2, 3]:
        dust_redness_up = [z[0] + sigma * z[1] for z in rdata.values()]
        dust_redness_down = [z[0] - sigma * z[1] for z in rdata.values()]
        sigma_up = [
            recalculate_redness_under_new_filter_pair(
                known_redness=z,
                old_mid_wave_angstrom=y,
                new_mid_wave_angstrom=uw1_uvv_mid,
            )
            - zorig
            for y, z, zorig in zip(
                midpoint_lambdas, dust_redness_up, translated_rednesses
            )
        ]
        sigma_down = [
            zorig
            - recalculate_redness_under_new_filter_pair(
                known_redness=z,
                old_mid_wave_angstrom=y,
                new_mid_wave_angstrom=uw1_uvv_mid,
            )
            for y, z, zorig in zip(
                midpoint_lambdas, dust_redness_down, translated_rednesses
            )
        ]
        print(f"{dust_redness_up=} {dust_redness_down=}")
        print(f"{sigma=}, {sigma_down=}, {sigma_up=}")

        sig_errs = np.vstack([sigma_down, sigma_up])

        plt.errorbar(
            translated_rednesses,
            list(rdata.keys()),
            xerr=sig_errs,
            color="#dbb89c",
            linestyle="none",
            alpha=0.33,
            uplims=True,
            lolims=True,
            fmt="D",
            mfc="white",
            mec="#dbb89c",
            mew=1.5,
            ecolor="#dbb89c",
            elinewidth=1.2,
            capsize=3,
        )
        print("----\n\n")

    plt.errorbar(
        measured_rednesses,
        midpoint_lambdas,
        xerr=measured_redness_errs,
        color="#688894",
        linestyle="none",
        alpha=0.9,
        uplims=True,
        lolims=True,
        fmt="D",
        mfc="white",
        mec="#688894",
        mew=1.5,
        ecolor="#688894",
        elinewidth=1.2,
        capsize=3,
    )

    plt.xlim(0, 100)
    plt.ylim(4000, 8500)
    plt.show()
