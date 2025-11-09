"""
Helper functions for cellpose segmentation and mask saving
Allie Spangaro, Toronto Metropolitan University
"""

import numpy as np
from cellpose import models, core, io, plot, utils
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
import cv2 as cv
import tifffile as tf


def file_sort_key(filename):
    """
    Generate a key to sort a list of image files in the 'MAX_chN-rXXcYYfZZ.tif' filename nomenclature by their plate location and channel.
    > e.g: MAX_ch1-r02c02f01.tif, MAX_ch2-r02c02f01.tif, MAX_ch3-r02c02f01.tif, MAX_ch3-r02c02f01.tif, MAX_ch1-r02c02f02.tif, MAX_ch2-r02c02f02.tif, MAX_ch3-r02c02f02.tif, MAX_ch3-r02c02f02.tif
    Parameters:
          filename (str): the filename from the image path
    Returns:
          (location,channel) (tuple of str): the list of files sorted by location and channel. Images will be ordered by location first, and then by channel to match the order seen in CellProfiler
    """
    parts = filename.split("-")
    channel = parts[0][
        -1:
    ]  # get the last character of the first part for the channel number
    location = parts[1]  # get the rXXcYYfZZ.tif portion
    return (location, channel)


def plate_location(filename):
    """
    Parse a string from an image filename in the MAX_chN-rXXcYYfZZ.tif filename nomenclature
    Parameters:
          filename (str): the filename from the image path
    Returns:
          location (str): the rowcolfield plate location of the image
    """
    parts = filename.split("-")  # split from the MAX_chN
    pre_location = parts[1]
    location = pre_location.split(".")[
        0
    ]  # get the first part of the second part (avoid the .tif)
    return location


def find_row_col_field_string(img_set_name):
    import re

    rowcol_pattern = r"r(\d{1,2})c(\d{1,2})f(\d{1,2})"  # Matches "RX" where X is the replicate number (placeholder for now)
    match = re.search(rowcol_pattern, img_set_name)
    if match:
        row_metadata = int(match.group(1))
        col_metadata = int(match.group(2))
        field_metadata = int(match.group(3))
        return f"r{str(row_metadata).rjust(2, '0')}c{str(col_metadata).rjust(2, '0')}f{str(field_metadata).rjust(2, '0')}"
    else:
        return ""


def sort_files(dir, image_ext):
    """
    Sort the list of directories sorted by their plate location and channel
    Parameters:
          dir (Path object or str): the directory containing the images
          image_ext (str, optional): the image extension, tif by default
    Returns:
          files (list of Path objects): the list of files sorted by location and channel
    """
    if not dir.exists():
        raise FileNotFoundError("directory does not exist")
    files = sorted(
        [
            f
            for f in dir.glob("*" + image_ext)
            if "_masks" not in f.name and "_flows" not in f.name and "SUM" not in f.name
        ],
        key=lambda x: file_sort_key(x.name),
    )
    # sort by number in filename
    if len(files) == 0:
        raise FileNotFoundError(
            "no image files found, did you specify the correct folder and extension?"
        )
    else:
        return files


def load_mask_list(dir, image_ext=".tif"):
    """
    Sort the list of masks
    Parameters:
          dir (Path object or str): the directory containing the images
          image_ext (str, optional): the image extension, tif by default
    Returns:
          files (list of Path objects): the list of files sorted by location and channel
    """
    if not dir.exists():
        raise FileNotFoundError("directory does not exist")
    try:
        files = [
            f
            for f in dir.glob("*" + image_ext)
            if "_masks" in f.name and "_flows" not in f.name and "SUM" not in f.name
        ]
    except IndexError as e:
        print(f"{e}, empty list, returning []")
        return []
    if len(files) == 0:
        return []
    else:
        return files


def print_files(files):
    """
    Print a list of filenames from a list of Path objects
    Parameters:
            grouped_files_by_channel (2D list of Path objects): list of files grouped into image sets by their channels
    """
    for f in files:
        print(f.name)


def load_sorted_directory_list(dir, image_ext=".tif"):
    """
    Load the list of directories sorted by their location and channel to allow grouping into a 2D list and force the objects to be Path objects
    Parameters:
          dir (Path object or str): the directory containing the images
          image_ext (str, optional): the image extension, tif by default
    Returns:
          file_list (list of Path objects): the list of files sorted by location and channel
    """
    dir = Path(dir)
    file_list = sort_files(dir, image_ext)
    return file_list


def get_nchannels(ordered_files):
    """
    Get the number of channels based on the last element list element in a list of image files ordered by channels
    Parameters:
          ordered_files (1D list of Path objects): list of files grouped into image sets by their channels
    Returns:
          nchannels (int): the number of channels based on the length of the first element
    """
    lastitem = ordered_files[-1].name
    parts = lastitem.split("-")
    ch_char = parts[0][-1:]
    nchannels = int(ch_char)
    return nchannels


def group_files_by_channel(files, nchannels=None):
    """
    Load the list of directories
    Parameters:
          dir (Path object or str): the directory containing the images
          nchannels (int,optionsl): the number of image channels to use for grouping
    Returns:
          grouped_files_by_channel (2D list of Path objects): list of files grouped into image sets by their channels
    """
    grouped_files_by_channel = []
    if nchannels == None:
        nchannels = get_nchannels(files)

    for i in range(0, len(files), nchannels):
        grouped_files_by_channel.append(files[i : i + nchannels])
    return grouped_files_by_channel


def print_grouped_files(grouped_files_by_channel):
    """
    Print a list of files grouped by channel to confirm that the images are loaded in the correct order
    Parameters:
            grouped_files_by_channel (2D list of Path objects): list of files grouped into image sets by their channels
    """
    for i in range(len(grouped_files_by_channel)):
        print(f"\n Group {i + 1} of {len(grouped_files_by_channel)}")
        for j in range(len(grouped_files_by_channel[i])):
            item = grouped_files_by_channel[i][j]
            print(" " + item.name)


def load_image_set_hap1(single_grouped_files_by_channel, nchannels=None):
    """
    Load an image set given file paths for a single image set; load images from a single element of the list made by the `group_files_by_channel` function
    Parameters:
          single_grouped_files_by_channel (list of Path objects): list of file paths grouped by channel and ordered by platemap location
          nchannels (int or None): the number of channels. Assumes based on length of first element otherwise
    Returns:
          image_set (list of 2D arrays): a list of 2D arrays representing the loaded single-channel grayscale images
    """
    if nchannels == None:
        nchannels = len(single_grouped_files_by_channel)
    # load the images from the channels - skip ch3 at position 2 as DAPI is always the last channel

    image_set = []
    for i in range(nchannels):
        image_channel = io.imread(single_grouped_files_by_channel[i])
        image_set.append(image_channel)
    return image_set


def img_preprocessing_hap1(
    img_set, selected_channels=[1, 2, 3, 4], nucleus_channel=3, mode="retain"
):
    """
    Preprocess a grayscale image given an list of single-channel grayscale images with historam equalization, and median filter smoothing and stack the image together
    Parameters:
           img_set (list of 2D arrays): a list of 2D arrays representing grayscale images
           selected channels (list of int): a list with the desiered channels to use/process (1-indexed)
           mode (str, "retain" or "remove): whether to keep the other channels downstream or discard them
    Returns:
          multi_channel_image (3D array): a 3D array containing the preprocessed grayscale images
    """
    from skimage import exposure, filters, morphology

    # ch1,ch2,ch3 = io.imread(files[0]), io.imread(files[1]), io.imread(files[3])
    # channels = [ch1, ch2, ch3]
    img_stack = []
    if mode == "retain":
        for i, channel in enumerate(img_set):
            if i + 1 in selected_channels:
                # footprint = morphology.disk(5)
                channel = img_01_normalization(channel)
                # don't apply histogram equalization to nuc; will bring out background too much
                if i + 1 != nucleus_channel:
                    channel = exposure.equalize_adapthist(
                        channel, kernel_size=100, clip_limit=0.01
                    )
                # channel = filters.gaussian(channel, sigma=2)
                channel = filters.median(channel, morphology.disk(2))
                img_stack.append(channel)
            else:
                channel = img_01_normalization(channel)
                img_stack.append(channel)
    elif mode == "remove":
        for chnum in selected_channels:
            # footprint = morphology.disk(5)
            channel = img_set[chnum - 1]
            # footprint = morphology.disk(5)
            channel = img_01_normalization(channel)
            channel = exposure.equalize_adapthist(
                channel, kernel_size=100, clip_limit=0.02
            )
            # channel = filters.gaussian(channel, sigma=2)
            channel = filters.median(channel, morphology.disk(2))
            img_stack.append(channel)
    multi_channel_image = np.stack(img_stack, axis=-1)
    return multi_channel_image


def get_multichannel_img_normalized(img_set, selected_channels=[1, 2, 3, 4]):
    """
    Create a multichannel grayscale image given an list of single-channel grayscale images and stack the image together
    Parameters:
           img_set (list of 2D arrays): a list of 2D arrays representing grayscale images
    Returns:
          multi_channel_image (3D array): an array containing the grayscale images with channel in the third dimension
    """
    # ch1,ch2,ch3 = io.imread(files[0]), io.imread(files[1]), io.imread(files[3])
    # channels = [ch1, ch2, ch3]
    img_stack = []
    for chnum in selected_channels:
        # footprint = morphology.disk(5)
        channel = img_set[chnum - 1]
        channel = img_01_normalization(channel)
        img_stack.append(channel)

    multi_channel_image = np.stack(img_stack, axis=-1)
    return multi_channel_image


def get_image_set_name(grouped_files_by_channel, index=1):
    """
    Get the name of the image set from the specified index in a list of filenames grouped by channel
    Corresponds to image set index in cellprofiler based on the order in the folder
    Parameters:
          grouped_files_by_channel (2D list of Path objects): an ordered 2D list of file paths grouped by channel and ordered by platemap location
          index (int, optional): the index to look up the rowcolfield of that image set
    Returns:
          set_name (str): String name of the image set from the rowcolfield filename nomenclature
    """
    set_name = plate_location(grouped_files_by_channel[index - 1].name)
    return set_name


def img_zscore_normalization(img):
    """
    ### Preprocess a grayscale image by normalizing pixels to a standard-scaled distribution (mean of 0, std of 1)
    Recommented for 2D only
    Parameters:
          img (2D or 3D array): a grayscale image as a 2D array
    Returns:
          norm_img (2D or 3D array): the grayscale image array normalized by z score
    """
    # Normalize each channel to z score
    norm_img = (img - np.mean(img)) / np.std(img)
    return norm_img


def img_01_normalization(img):
    """
    Normalize the intensity of each channel in a 16-bit grayscale image to the range [0, 1] to aid in preprocessing or segmentation by some algorithms
    Parameters:
          img (2D or 3D array): a grayscale image as a 2D array
    Returns:
          norm_img (2D or 3D array): the grayscale image array normalized by to [0,1]
    """
    norm_img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return norm_img


def img_rescaled(img, factor=0.5, anti_aliasing=False, channel_axis=-1):
    """
    Rescale a multichannel grayscale image by a given factor using skimage.transform.rescale
    Parameters:
          img (3D array): 3D array containing the grayscale image stack
          factor (float, optional): Factor to rescale the image by, 50% by default
          anti_aliasing (bool, optional): Flag whether to downsample the image or not, will downsample by default to reduce noise
    Returns:
          rescaled_image (3D array): a 3D array containing the rescaled grayscale image
    """
    from skimage import transform

    rescaled_img = transform.rescale(
        img, factor, anti_aliasing=anti_aliasing, channel_axis=channel_axis
    )
    return rescaled_img


def load_model(model_name=None, gpu=True):
    """
    Loads a cellpose model, uses cp_sam by default but can specify any model
    Parameters:
          model_name (None or str): the name of the model specified, by default assumes cpsam
          gpu (bool, optional): specify whether or not to use gpu, true by default

    Returns:
          model (Cellpose.model object): the model loaded
    """
    io.logger_setup()
    if gpu == True:
        if core.use_gpu() == False:
            raise ImportError("No GPU access, change your runtime")

    model = models.CellposeModel(gpu=gpu)
    return model


def plot_result(image, background):
    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Original image")
    ax[0].axis("off")

    ax[1].imshow(background, cmap="gray")
    ax[1].set_title("Background")
    ax[1].axis("off")

    ax[2].imshow(image - background, cmap="gray")
    ax[2].set_title("Result")
    ax[2].axis("off")

    fig.tight_layout()


def segment_nuclei_v3(
    orig_img,
    model,
    nucleus_channel=3,
    show_plot=True,
    flow_threshold=0.4,
    cellprob_threshold=0,
    tile_norm_blocksize=0,
    min_size=50,
    max_size_frac=0.4,
    diameter=None,
    niter=None,
):
    """
    Run cellpose-SAM on the nucleus channel from a multichannel cell image and return the predicted masks
    Designed for a 2160x2160 image rescaled to 1/4 of original size
    Parameters:
           img (2D or 3D array): grayscale image to be segmented by cellpose
           model (Cellpose.model): the cellpose model used for segmentation
           show_plot (bool, optional): flag whether to show a plot of the predicted mask flow
           flow_threshold (float, optional): the flow threshold for cellpose, 0.5 by default (from original 0.4 default). Down for more stringent, up for more lenient
           cellprob_threshold (float, optional): the cell probability threshold for cellpose, 1 by default (from original 0 default). Up to be more stringent, down for a more lenient threshold
           tile_norm_blocksize (int, optional): the tile normalization blocksize for cellpose, 100 by default. Generally between 100-200; 0 to turn off
           diameter (int or None, optional): the diameter for cellpose, None by default
           min_size (int, optional): the minimum size of masks to keep, 400 pixels by default
           max_size_frac (float, optional): the maximum size of masks to keep as a fraction of the image size, 0.4 by default
           niter (int or None, optional): the number of iterations for cellpose, None by default
    Returns:
          masks (list of 2D or 3D arrays): the predicted nuclei masks from the cellpose model
    """
    from skimage import morphology, filters

    img = orig_img[:, :, nucleus_channel - 1]  # get the DAPI channel (and 0-index it)

    # remove speckle-shaped autofluor
    # bg2 = morphology.white_tophat(img, morphology.disk(3))
    # img = img - bg2
    # img = morphology.closing(img, morphology.disk(2.5))

    img = img_01_normalization(img)
    # tf.imshow(img, cmap="plasma")
    # do a rolling ball background subtraction
    from skimage import data, restoration, util

    background = restoration.rolling_ball(
        img, kernel=restoration.ellipsoid_kernel((25, 25), 0.1)
    )

    img = img - background
    img = img_01_normalization(img)
    # plot_result(img, background)
    # plt.show()
    img = filters.gaussian(img, sigma=1)

    masks, flows, styles = model.eval(
        img,
        batch_size=64,
        diameter=diameter,
        niter=niter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize},
        max_size_fraction=max_size_frac,
    )
    # dilate before removing the ones touching edges to catch the stragglers
    masks = utils.dilate_masks(masks, n_iter=1)
    masks_removed_edges = utils.remove_edge_masks(masks)
    masks_removed_edges = utils.fill_holes_and_remove_small_masks(
        masks_removed_edges, min_size=min_size
    )
    if show_plot:
        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, img, masks_removed_edges, flows[0])
        plt.tight_layout()
        plt.show()
    return masks_removed_edges


def segment_cell_hap1(
    img,
    model,
    selected_channels=[1, 2, 3, 4],
    nucleus_channel=3,
    show_plot=True,
    flow_threshold=0.4,
    cellprob_threshold=0,
    tile_norm_blocksize=100,
    diameter=None,
    min_size=200,
    max_size_frac=0.4,  # keep masks up to 70% of image size
    niter=None,
):
    """
    Run cellpose-SAM on a grayscale multichannel cell image and return the predicted masks
    Designed for a 2160x2160 image rescaled to 1/4 of original size
    Parameters:
           img (2D or 3D array): grayscale image to be segmented by cellpose
           model (Cellpose.model): the cellpose model used for segmentation
           show_plot (bool, optional): flag whether to show a plot of the predicted mask flow
           flow_threshold (float, optional): the flow threshold for cellpose, 0.4 by default (from original 0.4 default). Down for more stringent, up for more lenient
           cellprob_threshold (float, optional): the cell probability threshold for cellpose, from original 0 default.
           tile_norm_blocksize (int, optional): the tile normalization blocksize for cellpose, 100 by default. Generally between 100-200; 0 to turn off
           diameter (int or None, optional): the diameter for cellpose, 60 as experimentally determined on MRC, but None by default
           min_size (int, optional): the minimum size of masks to keep, 200 pixels by default
           max_size_frac (float, optional): the maximum size of masks to keep as a fraction of the image size, 0.4 by default
           niter (int or None, optional): the number of iterations for cellpose, None by default
    Returns:
          masks (list of 2D or 3D arrays): the predicted masks from the cellpose model
    """
    from skimage import filters, morphology, exposure

    channels_to_add_list = []
    try:
        for chnum in selected_channels:
            zeroindex_chnum = chnum - 1
            if chnum == nucleus_channel:
                continue
            else:
                ch = img[:, :, zeroindex_chnum]
                channels_to_add_list.append(ch)
        # stack everything in the list (excludes nuclei channel)
        segment_image_pre = np.stack(channels_to_add_list, axis=-1)
    except IndexError as e:
        print(
            f"Error: you selected {len(selected_channels)} channels, when your image only has {np.shape(img)[-1]} channels"
        )
        raise e
    if len(selected_channels) > 2:
        segment_image = np.sum(segment_image_pre, axis=-1)
    else:
        segment_image = channels_to_add_list[0]

    # now we have nuclei seperated from the cyto segment image
    segment_image = img_01_normalization(segment_image)
    nuc_image = img[:, :, nucleus_channel - 1]
    # tf.imshow(img_combo, cmap="viridis")
    # smooth image and improve outline (sigma is the gaussian kernel)
    segment_image = filters.gaussian(segment_image, sigma=1)
    # Can also use unsharp mask, but it tends to chop the outlines too short img_combo = filters.unsharp_mask(img_combo, radius=0.5, amount=2)
    # stack the images
    img_selected_channels = np.stack([segment_image, nuc_image], axis=-1)

    masks, flows, styles = model.eval(
        img_selected_channels,
        batch_size=64,
        niter=niter,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize},
        max_size_fraction=max_size_frac,
        min_size=min_size,
    )
    masks = utils.fill_holes_and_remove_small_masks(masks, min_size=min_size)
    masks = utils.dilate_masks(masks, n_iter=2)
    # plot if true
    if show_plot:
        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, img_selected_channels, masks, flows[0])
        plt.tight_layout()
        plt.show()
    return masks


def save_masks(set_name, masks, outdir, image_ext=".tif", mask_type="cell"):
    """
    Save masks from a previously run cellpose model to a folder given an output directory
    Parameters:
           set_name (str): the image set name to be saved in the output file
           masks (list of 2D or 3D arrays): the masks output from Cellpose
           outdir (Path or str): the output directory to save masks
           image_ext (str, optional): image extension, .tif by default
           mask_type (str, optional): specify the mask type to save, cell by default
    """
    # save the masks to a file
    masks_ext = ".png" if image_ext == ".png" else ".tif"
    io.imsave(outdir / (set_name + "_" + mask_type + "_masks" + masks_ext), masks)


def save_mask_folder(
    ordered_files,
    outdir,
    image_ext=".tif",
    nchannels=None,
    resize_factor=0.25,
    v2=True,
):
    """
    Run cellpose and save cell and nuclear masks to a folder given an ordered list of files ordered by channel and an output directory
    Parameters:
          ordered_files (list): an ordered list of image file paths to be processed and saved; assumed to be ordered by location,channel by the `load_sorted_directory_list`
          outdir (Path or str): the output directory to save masks
          image_ext (str, optional): image extension, .tif by default
          nchannels (int, optional): specify the number of channels, default from the number of elements in the 1st element of the 2D list given
          resize_factor (float): the factor to rescale the image by for cellpose, 512x512 by default
    """
    model = load_model()
    if nchannels == None:  # handle default case when nchannels isn't specified
        nchannels = get_nchannels(ordered_files)

    grouped_files_by_channel = group_files_by_channel(ordered_files, nchannels)

    for i in trange(len(grouped_files_by_channel)):
        file_group = grouped_files_by_channel[i]
        img_set = load_image_set_hap1(file_group, nchannels)
        img_set_name = get_image_set_name(file_group)
        # print("Set name: ", img_set_name)
        if v2:
            stacked_img = img_preprocessing_hap1(img_set)
        else:
            # old function
            stacked_img = img_preprocessing_hap1(img_set)

        # rescale to 512 by 512 for processing speed
        rescaled_img = img_rescaled(stacked_img, factor=resize_factor)

        if v2:
            nuc_masks = segment_nuclei_v3(rescaled_img, model, show_plot=False)
            save_masks(
                img_set_name, nuc_masks, outdir, image_ext=image_ext, mask_type="nuclei"
            )
            cell_masks = segment_cell_hap1(rescaled_img, model, show_plot=False)
            save_masks(
                img_set_name,
                cell_masks,
                outdir,
                image_ext=image_ext,
                mask_type="v2_cell",
            )
        else:
            nuc_masks = segment_nuclei_v3(rescaled_img, model, show_plot=False)
            save_masks(
                img_set_name, nuc_masks, outdir, image_ext=image_ext, mask_type="nuclei"
            )
            cell_masks = segment_cell_hap1(rescaled_img, model, show_plot=False)
            save_masks(
                img_set_name, cell_masks, outdir, image_ext=image_ext, mask_type="cell"
            )


def save_imageJ_masks(set_name, masks, outdir, image_ext=".tif", mask_type="cell"):
    """
    Save masks as ImageJ ROIs
    Parameters:
             set_name (str): the image set name to be saved in the output file
             masks (list of 2D or 3D arrays): the masks output from Cellpose
             outdir (Path or str): the output directory to save masks
             image_ext (str, optional): image extension, .tif by default
             mask_type (str, optional): specify the mask type to save, cell by default
    """
    masks_ext = ".png" if image_ext == ".png" else ".tif"
    masks0 = io.imsave(outdir / (set_name + "_" + mask_type + "_masks" + masks_ext))
    io.save_rois(masks0, masks)


def preload_and_save_masks(
    ordered_files, outdir, masks_ext=".tif", mask_type="cell", nchannels=None
):
    """
    Load all images into memory and then batch-run cellpose on GPU
    ONLY use if image files are small, will crash with large files
    Parameters:
           ordered_files (list): a 1D list of file paths ordered by location,channel
           outdir (Path or str): the output directory to save masks
           image_ext (str, optional): image extension, .tif by default
           nchannels (int, optional): specify the number of channels, default = 4
    """
    model = load_model()
    if nchannels == None:  # handle default case when nchannels isn't specified
        nchannels = get_nchannels(ordered_files)

    grouped_files_by_channel = group_files_by_channel(ordered_files, nchannels)
    # if you have small images, you may want to load all of them first and then run, so that they can be batched together on the GPU
    print("loading images")
    imgs = load_image_set_hap1(
        [grouped_files_by_channel[i] for i in trange(len(grouped_files_by_channel))],
        nchannels,
    )

    print("running cellpose-SAM")
    flow_threshold = 0.4
    cellprob_threshold = 0
    tile_norm_blocksize = 0

    masks, flows, styles = model.eval(
        imgs,
        batch_size=32,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize},
    )

    print("saving masks")
    for i in trange(len(grouped_files_by_channel)):
        f = grouped_files_by_channel[i]
        set_name = get_image_set_name(f)
        io.imsave(outdir / (set_name + mask_type + "_masks" + masks_ext), masks[i])
