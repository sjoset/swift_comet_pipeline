from dataclasses import asdict

from swift_comet_pipeline.aperture.plateau import (
    ProductionPlateau,
    dict_to_production_plateau,
)
from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent


def dust_plateau_list_dict_serialize(
    p: dict[DustReddeningPercent, list[ProductionPlateau]]
) -> dict[DustReddeningPercent, list[dict]]:
    """
    Converts the ProductionPlateau lists into lists of dictionaries so that we can store them easily,
    in text or in the metadata of a pandas DataFrame in df.attrs
    """
    serialized_dict = {}
    for dust_redness, plateau_list_at_redness in p.items():
        serialized_dict[dust_redness] = [asdict(x) for x in plateau_list_at_redness]
    return serialized_dict


def dust_plateau_list_dict_unserialize(
    serialized_dict: dict[DustReddeningPercent, list[dict]]
) -> dict[DustReddeningPercent, list[ProductionPlateau]]:
    """
    Takes a stored dictionary serialized with dust_plateau_list_dict_serialize and reverses the process to produce
    a list of ProductionPlateau data structures, as a function of DustReddeningPercent
    """
    unserialized_dict = {}
    for dust_redness, plateau_dict_list_at_redness in serialized_dict.items():
        unserialized_dict[dust_redness] = [
            dict_to_production_plateau(raw_yaml=x) for x in plateau_dict_list_at_redness
        ]

    return unserialized_dict
