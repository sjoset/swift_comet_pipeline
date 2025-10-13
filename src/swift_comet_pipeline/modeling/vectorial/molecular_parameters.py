import astropy.units as u
from pyvectorial_au.model_input.vectorial_model_config import (
    ParentMolecule,
    FragmentMolecule,
)


def make_water_molecule_parent() -> ParentMolecule:

    # we take 93% of H2O destruction to be photodissociation
    total_to_dissociation_ratio = 0.93

    return ParentMolecule(
        tau_d_s=86000,
        tau_T_s=86000 * total_to_dissociation_ratio,
        v_outflow_kms=0.85,
        sigma_cm_sq=3.0e-16,
    )


@u.quantity_input
def make_slow_water_molecule_parent(
    v_outflow: u.Quantity[u.km / u.s],  # type: ignore
) -> ParentMolecule:

    # we take 93% of H2O destruction to be photodissociation
    total_to_dissociation_ratio = 0.93

    return ParentMolecule(
        tau_d_s=86000,
        tau_T_s=86000 * total_to_dissociation_ratio,
        v_outflow_kms=v_outflow.to_value(u.km / u.s),  # type: ignore
        sigma_cm_sq=3.0e-16,
    )


def make_hydroxyl_fragment() -> FragmentMolecule:
    return FragmentMolecule(tau_T_s=129000, v_photo_kms=1.05)
