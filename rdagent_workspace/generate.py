from pathlib import Path

import qlib
from qlib.constant import REG_CN
from qlib.data import D


def _get_provider_uri() -> str:
    for candidate in (
        Path("/root/.qlib/qlib_data/cn_data"),
        Path("/Users/apple/github/qlib/DATA/qlib_data/cn_data"),
    ):
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("No Qlib cn_data provider directory found")


qlib.init(provider_uri=_get_provider_uri(), region=REG_CN)

instruments = D.instruments(market="csi300")
fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]

data = D.features(
    instruments,
    fields,
    start_time="2014-01-01",
    end_time="2020-12-31",
    freq="day",
).swaplevel().sort_index()
data.to_hdf("./daily_pv_all.h5", key="data")

debug_data = D.features(
    instruments,
    fields,
    start_time="2018-01-01",
    end_time="2019-12-31",
    freq="day",
).swaplevel().sort_index()
debug_instruments = debug_data.reset_index()["instrument"].drop_duplicates().iloc[:100]
debug_data = debug_data.swaplevel().loc[debug_instruments].swaplevel().sort_index()
debug_data.to_hdf("./daily_pv_debug.h5", key="data")
