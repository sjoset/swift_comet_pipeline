from typing import Any, Type, TypeVar
import cattrs
import astropy.units as u
from astropy.time import TimeDelta
import numpy as np
import pandas as pd


T = TypeVar("T")
conv = cattrs.Converter()


# numpy
# ---------
@conv.register_unstructure_hook
def unstructure_ndarray(a: np.ndarray) -> dict:
    d = {"shape": a.shape, "dtype": str(a.dtype)}
    if np.iscomplexobj(a):
        d["complex"] = True
        d["real"] = a.real.ravel().tolist()  # type: ignore
        d["imag"] = a.imag.ravel().tolist()  # type: ignore
    else:
        d["complex"] = False
        d["data"] = a.ravel().tolist()
    return d


@conv.register_structure_hook
def structure_ndarray(d: dict, _) -> np.ndarray:
    dtype = np.dtype(d["dtype"])
    shape = tuple(d["shape"])
    if d.get("complex", False):
        real = np.asarray(d["real"], dtype=dtype)
        imag = np.asarray(d["imag"], dtype=dtype)
        arr = real + 1j * imag
    else:
        arr = np.asarray(d["data"], dtype=dtype)
    return arr.reshape(shape)


# pandas
# ---------
@conv.register_unstructure_hook
def unstructure_pandas_timestamp(ts: pd.Timestamp) -> dict:
    if ts.tz is None:
        ns = int(ts.value)
        plaintext_date = ts.strftime("%Y-%m-%d %H:%M:%S")
        return {"ns": ns, "tz": None, "date": plaintext_date}
    else:
        # value is UTC-based ns since epoch
        tzname = getattr(ts.tz, "zone", None) or str(ts.tz)
        ns = int(ts.tz_convert("UTC").value)
        plaintext_date = ts.strftime("%Y-%m-%d %H:%M:%S")
        return {"ns": ns, "tz": tzname, "date": plaintext_date}


@conv.register_structure_hook
def structure_pandas_timestamp(d: dict, _) -> pd.Timestamp:
    ns = int(d["ns"])
    tz = d.get("tz")
    if tz is None:
        return pd.Timestamp(ns, tz=None)  # type: ignore
    # reconstruct in UTC, then convert to the recorded zone
    ts = pd.Timestamp(ns, tz="UTC")
    try:
        return ts.tz_convert(tz)  # type: ignore
    except Exception:
        # keep UTC if the original zone is missing
        return ts  # type: ignore


# use the timedelta in nanoseconds
@conv.register_unstructure_hook
def unstructure_pandas_timedelta(td: pd.Timedelta) -> dict:
    ns = int(td.value)
    return {"ns": ns, "days": str(td)}


@conv.register_structure_hook
def structure_pandas_timedelta(d: dict, _) -> pd.Timedelta:
    ns = d["ns"]
    td = pd.Timedelta(ns, unit="ns")
    assert isinstance(td, pd.Timedelta)
    return td


# astropy
# ---------
@conv.register_unstructure_hook
def unstructure_astropy_timedelta(td: TimeDelta) -> dict:
    return {"value": float(td.to_value(u.s)), "unit": "s", "days": float(td.to_value(u.day))}  # type: ignore


@conv.register_structure_hook
def structure_astropy_timedelta(d: dict, _) -> TimeDelta:
    return TimeDelta(float(d["value"]), format="sec")


# scp primitives
# ---------

# cattrs defaults to asdict() for dataclasses

# @conv.register_unstructure_hook
# def unstructure_pixelcoord(pc: PixelCoord) -> dict:
#     return asdict(pc)
#
#
# @conv.register_structure_hook
# def structure_pixelcoord(d: dict, _) -> PixelCoord:
#     return PixelCoord(**d)


def to_unstructured(obj: Any) -> Any:
    return conv.unstructure(obj)


def from_unstructured(obj: Any, c: Type[T]) -> T:
    return conv.structure(obj, c)
