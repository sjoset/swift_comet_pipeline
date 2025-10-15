from swift_comet_pipeline.data_ingestion.orbit_data.orbit_data_download import (
    orbit_data_download,
)
from swift_comet_pipeline.pipeline.product_system.registry_and_store import Products


def do_earth_orbit_download(scp: Products) -> None:

    print("Downloading earth orbit data..")
    earth_df = orbit_data_download(scp=scp, horizons_id=None)

    scp.save_earth_orbit_data(df=earth_df)


def do_comet_orbit_download(scp: Products) -> None:

    print("Downloading comet orbit data..")
    comet_df = orbit_data_download(scp=scp, horizons_id=scp.cfg.jpl_horizons_id)

    scp.save_comet_orbit_data(df=comet_df)
