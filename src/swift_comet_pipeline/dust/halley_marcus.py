from dataclasses import asdict
from functools import partial
from io import StringIO
from typing import Callable

import pandas as pd
from scipy.interpolate import splev, splrep

from swift_comet_pipeline.types import HalleyMarcusCurve, HalleyMarcusCurveEntry


__all__ = [
    "get_halley_marcus_curve",
    "halley_marcus_curve_to_dataframe",
    "dataframe_to_halley_marcus_curve",
    "halley_marcus_curve_interpolation",
]

# Halley-Marcus curve for phase correction factors to Afrho
halley_marcus_curve_column_names = [
    "phase_deg",
    "phase_correction_zero",
    "phase_correction_ninety",
]
# https://asteroid.lowell.edu/static/comet/dustphaseHM_table.txt
halley_marcus_curve_raw_string = """
  0.0          1.0000             1.8604
  1.0          0.9596             1.7853
  2.0          0.9217             1.7146
  3.0          0.8859             1.6481
  4.0          0.8522             1.5855
  5.0          0.8205             1.5264
  6.0          0.7906             1.4708
  7.0          0.7624             1.4184
  8.0          0.7358             1.3689
  9.0          0.7107             1.3222
 10.0          0.6871             1.2782
 11.0          0.6647             1.2367
 12.0          0.6436             1.1974
 13.0          0.6237             1.1604
 14.0          0.6049             1.1254
 15.0          0.5872             1.0924
 16.0          0.5704             1.0612
 17.0          0.5546             1.0317
 18.0          0.5396             1.0039
 19.0          0.5255             0.9776
 20.0          0.5122             0.9528
 21.0          0.4996             0.9294
 22.0          0.4877             0.9073
 23.0          0.4765             0.8864
 24.0          0.4659             0.8667
 25.0          0.4559             0.8482
 26.0          0.4465             0.8307
 27.0          0.4377             0.8142
 28.0          0.4293             0.7988
 29.0          0.4215             0.7842
 30.0          0.4142             0.7706
 31.0          0.4073             0.7578
 32.0          0.4009             0.7458
 33.0          0.3949             0.7346
 34.0          0.3893             0.7242
 35.0          0.3840             0.7145
 36.0          0.3792             0.7055
 37.0          0.3747             0.6972
 38.0          0.3706             0.6895
 39.0          0.3668             0.6825
 40.0          0.3634             0.6761
 41.0          0.3603             0.6703
 42.0          0.3575             0.6651
 43.0          0.3540             0.6586
 44.0          0.3509             0.6528
 45.0          0.3482             0.6477
 46.0          0.3458             0.6433
 47.0          0.3438             0.6395
 48.0          0.3421             0.6364
 49.0          0.3407             0.6339
 50.0          0.3397             0.6319
 51.0          0.3389             0.6305
 52.0          0.3385             0.6297
 53.0          0.3383             0.6295
 54.0          0.3385             0.6297
 55.0          0.3389             0.6305
 56.0          0.3396             0.6318
 57.0          0.3405             0.6335
 58.0          0.3418             0.6358
 59.0          0.3432             0.6386
 60.0          0.3450             0.6418
 61.0          0.3470             0.6455
 62.0          0.3493             0.6498
 63.0          0.3518             0.6545
 64.0          0.3546             0.6596
 65.0          0.3576             0.6653
 66.0          0.3609             0.6714
 67.0          0.3645             0.6781
 68.0          0.3683             0.6852
 69.0          0.3724             0.6929
 70.0          0.3768             0.7010
 71.0          0.3815             0.7097
 72.0          0.3865             0.7190
 73.0          0.3917             0.7288
 74.0          0.3973             0.7391
 75.0          0.4032             0.7500
 76.0          0.4094             0.7616
 77.0          0.4159             0.7737
 78.0          0.4228             0.7865
 79.0          0.4300             0.8000
 80.0          0.4376             0.8141
 81.0          0.4456             0.8290
 82.0          0.4540             0.8445
 83.0          0.4627             0.8609
 84.0          0.4720             0.8781
 85.0          0.4816             0.8961
 86.0          0.4918             0.9149
 87.0          0.5024             0.9347
 88.0          0.5136             0.9555
 89.0          0.5253             0.9772
 90.0          0.5375             1.0000
 91.0          0.5504             1.0239
 92.0          0.5638             1.0490
 93.0          0.5780             1.0753
 94.0          0.5928             1.1029
 95.0          0.6084             1.1318
 96.0          0.6247             1.1622
 97.0          0.6419             1.1941
 98.0          0.6599             1.2276
 99.0          0.6788             1.2628
100.0          0.6987             1.2998
101.0          0.7196             1.3388
102.0          0.7416             1.3797
103.0          0.7648             1.4228
104.0          0.7892             1.4682
105.0          0.8149             1.5161
106.0          0.8420             1.5665
107.0          0.8706             1.6197
108.0          0.9008             1.6759
109.0          0.9327             1.7353
110.0          0.9664             1.7980
111.0          1.0021             1.8643
112.0          1.0399             1.9346
113.0          1.0799             2.0090
114.0          1.1223             2.0879
115.0          1.1673             2.1717
116.0          1.2151             2.2606
117.0          1.2659             2.3551
118.0          1.3200             2.4557
119.0          1.3776             2.5628
120.0          1.4389             2.6770
121.0          1.5045             2.7989
122.0          1.5744             2.9291
123.0          1.6493             3.0683
124.0          1.7294             3.2174
125.0          1.8153             3.3772
126.0          1.9075             3.5488
127.0          2.0066             3.7331
128.0          2.1132             3.9315
129.0          2.2281             4.1452
130.0          2.3521             4.3759
131.0          2.4861             4.6252
132.0          2.6312             4.8951
133.0          2.7884             5.1876
134.0          2.9592             5.5053
135.0          3.1450             5.8509
136.0          3.3474             6.2276
137.0          3.5685             6.6389
138.0          3.8104             7.0888
139.0          4.0755             7.5821
140.0          4.3669             8.1241
141.0          4.6877             8.7209
142.0          5.0418             9.3797
143.0          5.4336            10.1086
144.0          5.8682            10.9172
145.0          6.3518            11.8168
146.0          6.8912            12.8204
147.0          7.4948            13.9433
148.0          8.1724            15.2039
149.0          8.9355            16.6237
150.0          9.7981            18.2284
151.0         10.7767            20.0490
152.0         11.8914            22.1227
153.0         13.1662            24.4945
154.0         14.6309            27.2193
155.0         16.3215            30.3644
156.0         18.2826            34.0130
157.0         20.5698            38.2681
158.0         23.2524            43.2588
159.0         26.4178            49.1477
160.0         30.1769            56.1411
161.0         34.6719            64.5036
162.0         40.0862            74.5764
163.0         46.6588            86.8040
164.0         54.7036           101.7706
165.0         64.6368           120.2504
166.0         77.0148           143.2783
167.0         92.5871           172.2490
168.0        112.3702           209.0535
169.0        137.7479           256.2662
170.0        170.6031           317.3900
171.0        213.4773           397.1532
172.0        269.7281           501.8021
173.0        343.5886           639.2122
174.0        439.8869           818.3657
175.0        562.9161          1047.2490
176.0        713.6276          1327.6328
177.0        884.4779          1645.4827
178.0       1053.2719          1959.5071
179.0       1182.2266          2199.4144
180.0       1231.1735          2290.4753
"""


def halley_marcus_curve_to_dataframe(hmc: HalleyMarcusCurve) -> pd.DataFrame:
    data_dict = [asdict(hmc_entry) for hmc_entry in hmc if hmc_entry is not None]
    df = pd.DataFrame(data=data_dict)
    return df


def dataframe_to_halley_marcus_curve(df: pd.DataFrame) -> HalleyMarcusCurve:
    return df.apply(lambda row: HalleyMarcusCurveEntry(**row), axis=1).to_list()  # type: ignore


def get_halley_marcus_curve() -> HalleyMarcusCurve:
    hmc_io = StringIO(halley_marcus_curve_raw_string)
    hmc_df = pd.read_csv(
        hmc_io, sep="\\s+", header=None, names=halley_marcus_curve_column_names
    )

    return dataframe_to_halley_marcus_curve(df=hmc_df)


def halley_marcus_curve_interpolation(
    hmc: HalleyMarcusCurve = get_halley_marcus_curve(),
    normalization_phase_deg: float = 0.0,
) -> Callable | None:

    # TODO: we can produce the curve for an arbitrary normalization phase by finding the value of the curve's interpolation at normalization_phase
    # and dividing by that value, instead of doing the two cases below

    # still need the same function signature in case normalization_phase is outside [0, 180] degrees

    hmc_df = halley_marcus_curve_to_dataframe(hmc=hmc)

    if normalization_phase_deg == 0.0:
        spline_func = splrep(hmc_df.phase_deg, hmc_df.phase_correction_zero)
    elif normalization_phase_deg == 90.0:
        spline_func = splrep(hmc_df.phase_deg, hmc_df.phase_correction_ninety)
    else:
        return None

    f = partial(splev, tck=spline_func)

    return f
