from functools import partial

from tqdm import tqdm

from swift_comet_pipeline.image_manipulation.get_stacked_uvot_image_center import (
    get_uvot_image_center,
)
from swift_comet_pipeline.modeling.magnitude import magnitude_from_count_rate
from swift_comet_pipeline.photometry.aperture.aperture_count_rate import (
    aperture_analysis,
)
from swift_comet_pipeline.photometry.aperture.concentric_annuli import (
    make_concentric_annular_apertures,
)
from swift_comet_pipeline.pipeline.product_system.product_facade import Products
from swift_comet_pipeline.pipeline.product_system.product_key import EpochSubpipelineKey
from swift_comet_pipeline.pipeline.product_system.product_kind import ProductKind
from swift_comet_pipeline.pipeline.product_system.product_reference import (
    ProductReference,
)
from swift_comet_pipeline.scp_types.compound.annular_aperture_profile import (
    AnnularApertureProfileEntry,
    dataframe_from_annular_aperture_profile,
)
from swift_comet_pipeline.scp_types.compound.count_rate import CountRate
from swift_comet_pipeline.scp_types.primitive import *


def do_aperture_photometry_analysis(scp: Products, ref: ProductReference) -> None:
    """
    Build concentric annuli around the comet and take photometry for each, building a radial profile from the sum or median values in each aperture
    See type AnnularAperturePhotometryAnalysis for complete list of what is calculated here
    """

    # TODO: magic numbers: take from config or calculate, but needs to be fixed across all epochs
    # parameters of analysis
    max_aperture_radius = 8e5 * u.km  # type: ignore
    num_concentric_apertures = 400

    # load epoch info and the image to be analyzed
    pkey = ref.key
    assert isinstance(pkey, EpochSubpipelineKey)

    eid = scp.load_epoch_index_entry(epoch_id=pkey.epoch_id)
    assert eid is not None
    img_ref = ProductReference(kind=ProductKind.bg_subtracted_stacked_image, key=pkey)
    img_fits = scp.load_fits_image(ref=img_ref)
    assert img_fits is not None
    img = img_fits.data
    assert isinstance(img, SwiftUvotImage)
    bg_result = scp.load_background_result(key=pkey)
    assert bg_result is not None

    # derived quantities & information we need
    r_max_pix = max_aperture_radius.to_value(u.km) / eid.km_per_pix  # type: ignore
    comet_center = get_uvot_image_center(img=img)

    # make annular apertures with a circle aperture at r=0
    annular_apertures = make_concentric_annular_apertures(
        ap_center=comet_center,
        min_radius=0.0,
        max_radius=r_max_pix,  # type: ignore
        num_concentric_apertures=num_concentric_apertures,
    )
    aperture_r_pix = [float(annular_apertures[0].r)] + [  # type: ignore
        float(x.r_out) for x in annular_apertures[1:]  # type: ignore
    ]
    aperture_dr_pix = np.array([annular_apertures[0].r]) + np.diff(aperture_r_pix)  # type: ignore

    aperture_r_km = [x * eid.km_per_pix for x in aperture_r_pix]
    aperture_dr_km = [x * eid.km_per_pix for x in aperture_dr_pix]

    assert eid.exposure_times.get(pkey.filter_type, None) is not None

    annular_analyses = [
        aperture_analysis(
            img=img,
            ap=ap,
            background=bg_result,
            exposure_time_s=eid.exposure_times[pkey.filter_type],
        )
        for ap in tqdm(annular_apertures)
    ]

    ap_profile_data = {
        "aperture_r_pix": aperture_r_pix,
        "aperture_r_km": aperture_r_km,
        "aperture_dr_pix": aperture_dr_pix,
        "aperture_dr_km": aperture_dr_km,
    }
    ap_profile_data_names = list(ap_profile_data.keys())
    ap_profile_data_lists = [ap_profile_data[name] for name in ap_profile_data_names]

    annular_aperture_profile = [
        AnnularApertureProfileEntry(
            **aperture_count_rate_analysis_kwargs(aa),
            **dict(zip(ap_profile_data_names, data_lists)),
        )
        for aa, *data_lists in zip(annular_analyses, *ap_profile_data_lists)
    ]

    annular_aperture_analysis_df = dataframe_from_annular_aperture_profile(
        annular_aperture_profile=annular_aperture_profile
    )

    # calculate a few more things - curves of growth etc.
    half_pi = np.pi / 2.0
    df = annular_aperture_analysis_df

    df["cumulative_aperture_area"] = df.ap_num_pixels.cumsum()

    # median growth curve
    df["area_scaled_median"] = df.median_count_rate * df.ap_num_pixels
    df["area_scaled_median_variance"] = (
        df.count_rate_shot_noise_variance * half_pi + df.bg_variance
    )
    df["area_scaled_median_err"] = np.sqrt(df.area_scaled_median_variance)

    df["cumulative_area_scaled_median"] = df.area_scaled_median.cumsum()
    df["cumulative_area_scaled_median_variance"] = (
        df.area_scaled_median_variance.cumsum()
    )
    df["cumulative_area_scaled_median_err"] = np.sqrt(
        df.cumulative_area_scaled_median_variance
    )

    # sum growth curve
    df["cumulative_sum"] = df.sum_count_rate.cumsum()
    df["cumulative_sum_variance"] = df.sum_count_rate_variance.cumsum()
    df["cumulative_sum_err"] = np.sqrt(df.cumulative_sum_variance)

    # magnitude from cumulative median counts
    magnitude_func = partial(magnitude_from_count_rate, filter_type=pkey.filter_type)
    df["cumulative_median_cr"] = [
        CountRate(x, e)
        for x, e in zip(
            df.cumulative_area_scaled_median, df.cumulative_area_scaled_median_variance
        )
    ]
    df["cumulative_median_mag"] = df.cumulative_median_cr.apply(magnitude_func)
    df["cumulative_median_magnitude"] = df.cumulative_median_mag.apply(
        lambda x: x.value
    )
    df["cumulative_median_magnitude_variance"] = df.cumulative_median_mag.apply(
        lambda x: x.sigma**2
    )
    df["cumulative_median_magnitude_err"] = np.sqrt(
        df.cumulative_median_magnitude_variance
    )

    # magnitude from cumulative sum counts
    df["cumulative_sum_cr"] = [
        CountRate(x, e) for x, e in zip(df.cumulative_sum, df.cumulative_sum_variance)
    ]
    df["cumulative_sum_mag"] = df.cumulative_sum_cr.apply(magnitude_func)
    df["cumulative_sum_magnitude"] = df.cumulative_sum_mag.apply(lambda x: x.value)
    df["cumulative_sum_magnitude_variance"] = df.cumulative_sum_mag.apply(
        lambda x: x.sigma**2
    )
    df["cumulative_sum_magnitude_err"] = np.sqrt(df.cumulative_sum_magnitude_variance)

    df = df.drop(
        [
            "cumulative_median_cr",
            "cumulative_median_mag",
            "cumulative_sum_cr",
            "cumulative_sum_mag",
        ],
        axis=1,
    )

    # TODO: make a dataclass for this so we can use cattrs to structure it after loading
    analysis_metadata = {
        "max_aperture_radius_km": str(max_aperture_radius.to_value(u.km)),  # type: ignore
        "num_concentric_apertures": str(num_concentric_apertures),
    }

    aapa = annular_aperture_photometry_analysis_from_dataframe(df=df)
    scp.save_annular_aperture_analysis(aapa=aapa, metadata=analysis_metadata, key=pkey)
