from dataclasses import dataclass, asdict


@dataclass
class Plateau:
    begin_index: int
    end_index: int


# TODO: rename begin_r and end_r to begin_r_pix, end_r_pix
@dataclass
class ProductionPlateau:
    begin_r: float
    end_r: float
    begin_q: float
    end_q: float


def production_plateau_to_dict(plateau: ProductionPlateau) -> dict:
    """
    Converts ProductionPlateau to dictionary of {'begin_r': value, 'end_r': value, ...}
    """
    return asdict(plateau)


def dict_to_production_plateau(raw_yaml: dict) -> ProductionPlateau:
    """
    Converts dictionary of {'begin_r': value, 'end_r': value, ...} and produces a ProductionPlateau
    """
    # TODO: we only use this on dicts produced by production_plateau_to_dict, so error checking shouldn't be
    # necessary - but maybe we should check anyway
    return ProductionPlateau(**raw_yaml)
