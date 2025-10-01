from astropy.time import Time
from astroquery.jplhorizons import Horizons


def get_comet_position_at_time(horizons_id: str, mt: Time):
    horizons_response = Horizons(
        id=horizons_id, location="@swift", epochs=mt.jd, id_type="designation"
    )
    eph = horizons_response.ephemerides(closest_apparition=True)  # type: ignore

    return eph["RA"][0], eph["DEC"][0]
