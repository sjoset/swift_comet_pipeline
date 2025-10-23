from swift_comet_pipeline.builders.afrho_calculator import (
    do_afrho_from_aperture_photometry_analysis,
    do_afrho_from_radial_profile,
)
from swift_comet_pipeline.builders.epoch_identifier import do_epoch_identification
from swift_comet_pipeline.builders.epoch_indexer import do_epoch_index
from swift_comet_pipeline.builders.image_vetoer import do_image_veto
from swift_comet_pipeline.builders.orbit_downloader import (
    do_comet_orbit_download,
    do_earth_orbit_download,
)
from swift_comet_pipeline.builders.radial_profile_extractor import (
    do_radial_profile_from_cone,
)
from swift_comet_pipeline.builders.radial_profile_water_production import (
    do_radial_profile_water_production,
)
from swift_comet_pipeline.builders.raw_observation_log_builder import (
    do_observation_log_raw,
)
from swift_comet_pipeline.builders.stacker import do_stack
from swift_comet_pipeline.builders.aperture_photometry_analyzer import (
    do_aperture_photometry_analysis,
)
from swift_comet_pipeline.builders.aperture_water_production_calculator import (
    do_aperture_water_production,
)
from swift_comet_pipeline.builders.background_determiner import (
    do_background_determination,
)
from swift_comet_pipeline.builders.background_subtractor import (
    do_background_subtraction,
)


from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    ProductKind,
    ProductReference,
    Products,
)


# TODO: all builders should look for existing products and attempt to use those same parameters to re-build
# TODO: all builders should ask to replace existing products
def do_build(scp: Products, ref: ProductReference) -> None:

    # TODO: use log to record which ProductReference we are attempting to build

    if ref == ProductKind.observation_log_raw:
        do_observation_log_raw(scp=scp)

    if ref == ProductKind.observation_log_with_epochs:
        do_epoch_identification(scp=scp)

    if ref.kind == ProductKind.observation_log_with_vetoes:
        do_image_veto(scp=scp)

    if ref.kind == ProductKind.orbit_data_earth:
        do_earth_orbit_download(scp=scp)

    if ref.kind == ProductKind.orbit_data_comet:
        do_comet_orbit_download(scp=scp)

    if ref.kind == ProductKind.epoch_index:
        do_epoch_index(scp=scp)

    if (
        ref.kind == ProductKind.stacked_image_with_background
        or ref.kind == ProductKind.stacked_image_exposure_map
    ):
        do_stack(scp=scp, ref=ref)

    if ref.kind == ProductKind.background_determination:
        do_background_determination(scp=scp, ref=ref)

    if ref.kind == ProductKind.bg_subtracted_stacked_image:
        do_background_subtraction(scp=scp, ref=ref)

    if ref.kind == ProductKind.annular_aperture_photometry_analysis:
        do_aperture_photometry_analysis(scp=scp, ref=ref)

    if ref.kind == ProductKind.radial_profile_from_cone:
        do_radial_profile_from_cone(scp=scp, ref=ref)

    if ref.kind == ProductKind.aperture_water_production:
        do_aperture_water_production(scp=scp, ref=ref)

    if ref.kind == ProductKind.radial_profile_water_production:
        do_radial_profile_water_production(scp=scp, ref=ref)

    if ref.kind == ProductKind.afrho_from_aperture_photometry_analysis:
        do_afrho_from_aperture_photometry_analysis(scp=scp, ref=ref)

    if ref.kind == ProductKind.afrho_from_radial_profile:
        do_afrho_from_radial_profile(scp=scp, ref=ref)
