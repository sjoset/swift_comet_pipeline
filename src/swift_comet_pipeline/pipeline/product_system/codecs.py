from typing import Protocol, Any
from pathlib import Path
import json

from astropy.table import Table
from astropy.io import fits

from swift_comet_pipeline.data_ingestion.observation_log.observation_log_io import (
    read_observation_log,
    write_observation_log,
)
from swift_comet_pipeline.scp_types.primitive import *


# -----------------------------------------------------------------------------
# Codecs (format adapters with structural typing)
# -----------------------------------------------------------------------------
class Codec(Protocol):
    suffix: str

    def dump(self, obj: Any, path: Path) -> None: ...
    def load(self, path: Path) -> Any: ...


class ObservationLogCodec:
    suffix = ".parquet"

    def dump(self, obj: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_observation_log(obs_log=obj, obs_log_path=path)

    def load(self, path: Path) -> Any:
        return read_observation_log(obs_log_path=path)


class JSONCodec:
    suffix = ".json"

    def dump(self, obj: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    def load(self, path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class PandasDataframeToCSVCodec:
    suffix = ".csv"

    def dump(self, obj: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.to_csv(path, index=False)

    def load(self, path: Path) -> Any:
        return pd.read_csv(path)


class PandasDataframeToECSVCodec:
    suffix = ".ecsv"

    def dump(self, obj: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        t = Table.from_pandas(obj)
        t.meta = obj.attrs
        t.write(path, format="ascii.ecsv", overwrite=True)

    def load(self, path: Path) -> Any:
        t = Table.read(path, format="ascii.ecsv")
        df = t.to_pandas()
        df.attrs.update(t.meta)
        return df


class InternalFITSImageCodec:
    """
    We use only extension 1 to store image/header info for FITS we create ourselves
    """

    suffix = ".fits"

    def dump(self, obj: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.writeto(path, overwrite=True)

    def load(self, path: Path) -> Any:
        hdul = fits.open(path, lazy_load_hdus=False, memmap=True)
        img = fits.ImageHDU(data=hdul[1].data, header=hdul[1].header)  # type: ignore
        hdul.close()
        return img


class LightcurveToCSVCodec:
    suffix = ".csv"

    def dump(self, obj: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.to_csv(path, index=False)

    def load(self, path: Path) -> Any:
        df = pd.read_csv(path)
        return df
