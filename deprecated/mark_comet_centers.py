def mark_comet_centers(swift_project_config: SwiftProjectConfig) -> None:
    del swift_project_config
    return
    # """
    # Finds images in uvv and uw1 filters, and outputs pngs images of each
    # observation annotated with the center of the comet marked.
    # Output images are placed in image_save_dir/[filter]/
    # """
    #
    # pipeline_files = PipelineFiles(
    #     base_product_save_path=swift_project_config.project_path
    # )
    #
    # swift_data = SwiftData(swift_project_config.swift_data_path)
    #
    # # select the epoch we want to process
    # epoch_id = epoch_menu(pipeline_files=pipeline_files)
    # if epoch_id is None:
    #     print("no epoch selected!")
    #     wait_for_key()
    #     return
    #
    # # load the epoch database
    # epoch = pipeline_files.read_pipeline_product(
    #     PipelineProductType.epoch, epoch_id=epoch_id
    # )
    # if epoch is None:
    #     print("Error loading epoch!")
    #     wait_for_key()
    #     return
    #
    # image_save_dir: pathlib.Path = swift_project_config.project_path / "centers"
    # plt.rcParams["figure.figsize"] = (15, 15)
    #
    # # directories to store the uw1 and uvv images: image_save_dir/[filter]/
    # dir_by_filter = {
    #     SwiftFilter.uw1: image_save_dir
    #     / pathlib.Path(filter_to_file_string(SwiftFilter.uw1)),
    #     SwiftFilter.uvv: image_save_dir
    #     / pathlib.Path(filter_to_file_string(SwiftFilter.uvv)),
    # }
    # # create directories we will need if they don't exist
    # for fdir in dir_by_filter.values():
    #     fdir.mkdir(parents=True, exist_ok=True)
    #
    # progress_bar = tqdm(epoch.iterrows(), total=epoch.shape[0])
    # for _, row in progress_bar:
    #     obsid = row["OBS_ID"]
    #     extension = row["EXTENSION"]
    #     px = round(float(row["PX"]))
    #     py = round(float(row["PY"]))
    #     filter_type = row.FILTER
    #     filter_str = filter_to_file_string(filter_type)  # type: ignore
    #
    #     # ask where the raw swift data FITS file is and read it
    #     image_path = swift_data.get_uvot_image_directory(obsid=obsid) / row["FITS_FILENAME"]  # type: ignore
    #     image_data = fits.getdata(image_path, ext=row["EXTENSION"])
    #
    #     output_image_name = pathlib.Path(f"{obsid}_{extension}_{filter_str}.png")
    #     output_image_path = dir_by_filter[filter_type] / output_image_name  # type: ignore
    #
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(1, 1, 1)
    #
    #     zscale = ZScaleInterval()
    #     vmin, vmax = zscale.get_limits(image_data)
    #
    #     im1 = ax1.imshow(image_data, vmin=vmin, vmax=vmax)  # type: ignore
    #     fig.colorbar(im1)
    #     # mark comet center
    #     plt.axvline(px, color="w", alpha=0.3)
    #     plt.axhline(py, color="w", alpha=0.3)
    #
    #     ax1.set_title("C/2013US10")
    #     ax1.set_xlabel(f"{row['MID_TIME']}")
    #     ax1.set_ylabel(f"{row['FITS_FILENAME']}")
    #
    #     plt.savefig(output_image_path)
    #     plt.close()
    #     # num_processed += 1
    #
    #     progress_bar.set_description(f"Processed {obsid} extension {extension}")
    #
    # print("")
