# def show_lightcurves(df) -> None:
#
#     dust_rednesses = [DustReddeningPercent(x) for x in set(df.assumed_redness_percent)]
#
#     dust_cmap = LinearSegmentedColormap.from_list(
#         name="custom", colors=["#8e8e8e", "#bb0000"], N=(len(dust_rednesses) + 1)
#     )
#     dust_line_colors = dust_cmap(
#         np.array(dust_rednesses).astype(np.float32) / DustReddeningPercent(100.0)
#     )
#
#     fig, axs = plt.subplots(2, 4)
#
#     print(axs)
#     print(list(axs))
#     print(np.ravel(axs))
#     for dust_redness, line_color, ax in zip(
#         dust_rednesses, dust_line_colors, np.ravel(axs)
#     ):
#         ax.plot(
#             df.time_from_perihelion,
#             df.far_fit_q,
#             label=f"Q(H20) at {dust_redness=}",
#             color=line_color,
#             alpha=0.65,
#         )
#
#     # ax.set_xscale("log")
#     # ax.set_yscale("log")
#     ax.set_xlabel("days from perihelion")
#     ax.set_ylabel("Q(H2O)")
#     ax.legend()
#     # fig.suptitle(
#     #     f"Rh: {helio_r.to_value(u.AU):1.4f} AU, Delta: {delta.to_value(u.AU):1.4f} AU,\nTime from perihelion: {time_from_perihelion.to_value(u.day)} days\nfitting data from {fit_begin_r.to(u.km):1.3e} to {fit_end_r.to(u.km):1.3e}"  # type: ignore
#     # )
#     plt.show()
#     plt.close()
