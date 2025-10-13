# # TODO: test this
# def read_solar_spectrum_sorce(
#     solar_spectrum_path: pathlib.Path, solar_spectrum_time: Time
# ) -> SolarSpectrum:
#     # load the solar spectrum
#     solar_spectrum_df = pd.read_csv(solar_spectrum_path)
#
#     # select the spectrum from the current date
#     solar_spectrum_df["time (Julian Date)"].map(lambda x: Time(x, format="jd"))
#     solar_mask = solar_spectrum_df["time (Julian Date)"] == np.round(
#         solar_spectrum_time.jd
#     )
#     solar_spectrum = solar_spectrum_df[solar_mask]
#
#     # TODO: put the dataframe columns into an np.array() and test
#     solar_lambdas = solar_spectrum["wavelength (nm)"]
#     solar_irradiances = solar_spectrum["irradiance (W/m^2/nm)"]
#
#     return SolarSpectrum(
#         lambdas=np.array(solar_lambdas), irradiances=np.array(solar_irradiances)
#     )


# TODO: this is probably broken: make it use the base directory of the code instead of relative path in spectrum_path

# def get_sorce_spectrum(t: Time) -> SolarSpectrum:
#     """
#     Looks in the directory data/solar_spectra/[year] for the file sorce_ssi_l3.csv, which should contain columns
#     named 'time (Julian Date)', 'wavelength (nm)' and 'irradiance (W/m^2/nm)'
#     """
#     year = t.to_datetime().date().year  # type: ignore
#     spectrum_path = pathlib.Path(
#         "data/solar_spectra/sorce/" + str(year) + "/sorce_ssi_l3.csv"
#     )
#     ss = read_solar_spectrum_sorce(spectrum_path, t)
#
#     # fix the units to be the same as those given from the file read by read_solar_spectrum
#     ss.irradiances = ss.irradiances * 100
#
#     return ss
