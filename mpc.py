#!/usr/bin/env python3

import sys
import pathlib
import pandas as pd


def main():
    mpc_comet_data_path = pathlib.Path("data/cometels.json.gz")

    comet_df = pd.read_json(mpc_comet_data_path, compression="gzip")
    print(comet_df.columns)
    for _, row in comet_df.iterrows():
        # print(row["Designation_and_name"])
        if "C/" in row["Designation_and_name"]:
            print(
                row["Designation_and_name"],
                row["Year_of_perihelion"],
                row["Month_of_perihelion"],
                row["Day_of_perihelion"],
            )


if __name__ == "__main__":
    sys.exit(main())
