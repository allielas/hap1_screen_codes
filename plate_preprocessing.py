import os
import re
import numpy as np
import pandas as pd
import sqlite3

# import matplotlib.pyplot as plt
from scipy import stats


# Debugging functions
def passage_group(passage_num):
    """
    Group passages into bins for plotting
    returns string of the group that the passage number belongs to
    """
    # use this function to group passages into groups for plotting
    try:
        passage = int(passage_num)
        if 6 <= passage <= 10:
            return "P6-10"
        elif 11 <= passage <= 13:
            return "P11-13"
        elif 14 <= passage <= 16:
            return "P14-16"
        elif 17 <= passage <= 19:
            return "P17-19"
        elif 20 <= passage <= 22:
            return "P20-22"
        elif 23 <= passage <= 25:
            return "P23-25"
        elif 26 <= passage <= 28:
            return "P26-28"
        elif passage >= 29:
            return "P29+"
        else:
            return "Unknown"
    except ValueError as e:
        # print(f"value is not a number, caught {e}; returning NaN")
        return None


def add_drug_to_group(init_df, group, drug):
    """
    Add the name of a drug treatment from the "Drug" column to the main "group" column

    Returns
        Series object: A series containing the column with the drug added to the group
    """
    if drug is not None:
        # Replace values in 'col1' with values from 'col2' only if 'col2' is not None or NaN
        df = init_df.copy()
        df[group] = np.where(df[drug].notna(), df[drug], df[group])
        newcol = df[group]

    return newcol


def take_drug_from_condition(init_df, group_variable_col, drug_metadata_col, drug_name):
    """Grab a string "drug name" from a column with the drug metadata and add it to the grouping variable column

    Args:
        init_df (DataFrame): The dataframe to add to
        group_variable_col (str): _description_
        drug_metadata_col (str): The column to take from
        drug_name (str): The drug used

    Returns:
        Series: The modified grouping variable column with the added drug
    """
    if drug_metadata_col is not None:
        # Replace values in 'col1' with values from 'col2' only if 'col2' is not None or NaN
        df = init_df.copy()

        df[group_variable_col] = np.where(
            df[drug_metadata_col].str.contains(drug_name, na=False),
            df[drug_metadata_col],
            df[group_variable_col],
        )
        newcol = df[group_variable_col]

    return newcol


def passage_groups_sort_key(group_name):
    """
    Key function for natural sorting of strings containing numbers.
    Extract numeric parts and convert to int .
    """
    digit_pattern = r"([0-9]+)"  # Matches "RX" where X is the replicate number (placeholder for now)
    match = re.search(digit_pattern, group_name)
    if match:
        first_digit = int(match.group(1))
        return first_digit
    else:
        text = group_name.lower()
        if text == "doxo":
            return 999
        else:
            return ValueError


def sort_df_by_replicate_number(df, x_value):
    """Sort a pandas dataframe of experimental data that has been grouped by an x_value by the integer representation of the replicate number using a sort key
    Ideally used before plotting so that your plots have the same pallete and are easily comparable
    Args:
        df (DataFrame): your df to be sorted
        x_value (str): the column containing your grouping variable to be sorted by replicate number

    Returns:
        DataFrame: The dataframe sorted by replicate number
    """
    df_sorted = df.sort_values(
        by=[x_value], key=lambda x: x.map(passage_groups_sort_key)
    ).reset_index(drop=True)
    return df_sorted


def enforce_objects_one_to_one(
    df,
    parent_obj="Cell",
    child_obj="Nuclei",
    parent_colname="Cell_AreaShape_Area",
    child_colname="Cell_Mean_Nuclei_AreaShape_Area",
):
    """apply filters based on the number of nuclei to remove to enforce a 1:1 cell-nucleus relationship:
    - Cells that have less or more than one nucleus
    - Poorly segmented cells where the cell area is smaller than the nuclei area
    - Image is not flagged as empty by CellProfiler
    Can also use other objects instead of nuclei

    Args:
        df (DataFrame): The parent dataframe
        child_obj (str, optional): _description_. Defaults to "Nuclei".
        parent_colname (str, optional): _description_. Defaults to "Cell_AreaShape_Area".
        child_colname (str, optional): _description_. Defaults to "Cell_Mean_Nuclei_AreaShape_Area".

    Returns:
        DataFrame: the filtered dataframe with the removed objects
    """
    if child_obj == "Nuclei" and parent_obj == "Cell":
        try:
            not_empty_df = df[df["Metadata_EmptyImage_Cells"] == 0]
            normal_cells = not_empty_df[
                not_empty_df["Cell_Classify_one_nuc"] == 1
            ]  # one nucleus only
        except KeyError as e:
            print(f"KeyError {e}; skipping emptyimage")
            normal_cells = df[df["Cell_Classify_one_nuc"] == 1]
        # remove if my cell area is bigger than nuclear area
        size_filtered_cells = normal_cells[
            normal_cells[parent_colname] > normal_cells[child_colname]
        ]
        final_df = size_filtered_cells.reset_index(drop=True)
        return final_df
    else:
        # just do the size excludion
        not_empty_df = df[df["Metadata_EmptyImage_Cells"] == 0]
        size_filtered_cells = not_empty_df[
            not_empty_df[parent_colname] > not_empty_df[child_colname]
        ]
        return size_filtered_cells


def filter_nuclei_outisde_bbox(overlapped_df, nuc_prefix="Nuclei", cell_prefix=""):
    """Exclude all rows with cells that have nuclei extruding the cell boundaries

    Args:
        overlapped_df (Dataframe that contains objects to enforce overlap): _description_
        nuc_prefix (str, optional): _description_. Defaults to "Nuclei".
        cell_prefix (str, optional): _description_. Defaults to "".

    Returns:
        _type_: _description_
    """
    df = overlapped_df.copy()
    # Get the coordinate of the cell and nuclei objects
    # Note- make sure the cells and nuclei are 1:1 or this will fail
    nuc_min_x, nuc_min_y, nuc_max_x, nuc_max_y = (
        df[f"{nuc_prefix}_AreaShape_BoundingBoxMinimum_X"],
        df[f"{nuc_prefix}_AreaShape_BoundingBoxMinimum_Y"],
        df[f"{nuc_prefix}_AreaShape_BoundingBoxMaximum_X"],
        df[f"{nuc_prefix}_AreaShape_BoundingBoxMaximum_Y"],
    )
    cell_min_x, cell_min_y, cell_max_x, cell_max_y = (
        df[f"{cell_prefix}AreaShape_BoundingBoxMinimum_X"],
        df[f"{cell_prefix}AreaShape_BoundingBoxMinimum_Y"],
        df[f"{cell_prefix}AreaShape_BoundingBoxMaximum_X"],
        df[f"{cell_prefix}AreaShape_BoundingBoxMaximum_Y"],
    )
    # now get two boolean series that match the conditions
    df_is_contained_x = (nuc_min_x >= cell_min_x) & (nuc_max_x <= cell_max_x)
    df_is_contained_y = (nuc_min_y >= cell_min_y) & (nuc_max_y <= cell_max_y)
    # and then we get the subset of the dataframe where both are true
    df_is_contained_xy = df[df_is_contained_x & df_is_contained_y]

    final_df = df_is_contained_xy.reset_index(drop=True)
    return final_df


def filter_out_images_with_n_cells(
    df, n, image_count_col="Image_Count_Cell", prefix=""
):
    """Filter out rows from images with less than n cells

    Args:
        df (DataFrame): the cell df
        n (int): Threshold min number of cells to excluded
        image_count_col (str, optional): The name of the column counting cells per image. Defaults to "Image_Count_Cell".
        prefix (str, optional): perfix for column. Defaults to "".

    Returns:
        DataFrame: the filtered df
    """
    filter_df = df.copy()
    filter_df = filter_df[image_count_col] > n
    final_df = filter_df.reset_index(drop=True)
    return final_df


def filter_out_empty_compartment_from_cells(df, organelle, prefix=""):
    """Remove rows from a dataframe where a parent "cell" object doesn't have any child objects of {organelle}

    Args:
        df (DataFrame): _description_
        organelle (str): the organelle in plural. Typically "Mitochondria or Lysosomes (or Nuclei)
        prefix (str, optional): _description_. Defaults to "".

    Returns:
        Dataframe: _description_
    """
    organelle = organelle.title()
    this_df = df.copy()
    atleast_one_df = this_df[
        this_df[f"{prefix}Children_{organelle}_Count"] > 1
    ].reset_index(drop=True)
    return atleast_one_df


def define_cell_features(df):
    """_summary_

    Args:
        df (DataFrame): _description_

    Returns:
        list: A list of columns that are numerical features
    """
    # Get the columns of the dataframe
    columns_list = df.columns.tolist()
    columns_list = [
        col
        for col in columns_list
        if "Metadata" not in col
        and "FileName" not in col
        and "PathName" not in col
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    # old_columns_list = columns_list = [col for col in columns_list if 'Metadata' not in col and 'FileName' not in col and 'PathName' not in col]
    # print("Original columns:", len(old_columns_list), "Filtered columns:", len(columns_list))
    return columns_list


def mean_intensity_per_compartment_per_cell(df, compartment, name, tag, math=None):
    # Calculate the mean intensity of each compartment per cell
    # mean_intesity_per_compartment = integrated / (children*mean_area)
    colname = f"Mean_Intensity_Per_{compartment} Per_Cell"
    integrated = "Intensity_IntegratedIntensity_" + tag
    # children = 'Children_' + compartment + '_Count'
    # mean_area = 'Mean_'+ compartment + '_AreaShape_Area'
    total_organelle_area = name + "_AreaShape_Area"
    total_organelle_area = math if math is not None else total_organelle_area

    df[colname] = df.apply(lambda x: x[integrated] / x[total_organelle_area], axis=1)

    return df[colname]


def calculate_corrected_features(full_df):
    """_summary_

    Args:
        full_df (DataFrame): _description_

    Returns:
        df (DataFrame): the df with all the feature calcs
    """
    df = full_df.copy()
    df["Math_Total_Mitochondria_AreaShape_Area_PerCell"] = (
        df["Children_Mitochondria_Count"] * df["Mean_Mitochondria_AreaShape_Area"]
    )
    df["Math_Total_Lysosomes_AreaShape_Area_PerCell"] = (
        df["Children_Lysosomes_Count"] * df["Mean_Lysosomes_AreaShape_Area"]
    )

    # Total intensity per cell based on ingegrated instensity if I don't already have the merged area
    df["Mean_Intensity_Per_Lysosomes_PerCell_Area"] = (
        mean_intensity_per_compartment_per_cell(
            df,
            "Lysosomes",
            "MergedLysoPerCell",
            "LAMP1",
            math="Math_Total_Lysosomes_AreaShape_Area_PerCell",
        )
    )
    df["Mean_Intensity_Per_Mitochondria_PerCell_Area"] = (
        mean_intensity_per_compartment_per_cell(
            df,
            "Mitochondria",
            "MergedMitoPerCell",
            "MitoTracker",
            math="Math_Total_Mitochondria_AreaShape_Area_PerCell",
        )
    )
    # same thing but for medians
    df["Median_Intensity_Per_Lysosomes_PerCell_Area"] = (
        df["Children_Lysosomes_Count"]
        * df["Mean_Lysosomes_Intensity_MeanIntensity_LAMP1"]
    )
    df["Median_Intensity_Per_Mitochondria_PerCell_Area"] = (
        df["Children_Mitochondria_Count"]
        * df["Mean_Mitochondria_Intensity_MeanIntensity_MitoTracker"]
    )

    # Corrected mitochondria and lysosomes counts per cell area (density)
    df["Density_Children_Mitochondria_Count_PerCell_Area"] = (
        df["Children_Mitochondria_Count"] / df["AreaShape_Area"]
    )
    df["Density_Children_Lysosomes_Count_PerCell_Area"] = (
        df["Children_Lysosomes_Count"] / df["AreaShape_Area"]
    )

    # organelle area fractions per cell ratio
    df["OccupiedAreaFraction_Mitochondria_PerCell_Area"] = (
        df["Math_Total_Mitochondria_AreaShape_Area_PerCell"] / df["AreaShape_Area"]
    )
    df["OccupiedAreaFraction_Lysosomes_PerCell_Area"] = (
        df["Math_Total_Lysosomes_AreaShape_Area_PerCell"] / df["AreaShape_Area"]
    )

    # Compartment diameter ratios
    df["Mean_Lysosomes_DiameterRatio_PerCell"] = (
        df["Mean_Lysosomes_AreaShape_MaxFeretDiameter"]
        / df["Mean_Lysosomes_AreaShape_MinFeretDiameter"]
    )
    df["Mean_Mitochondria_DiameterRatio_PerCell"] = (
        df["Mean_Mitochondria_AreaShape_MaxFeretDiameter"]
        / df["Mean_Mitochondria_AreaShape_MinFeretDiameter"]
    )
    df["Median_Mitochondria_DiameterRatio_PerCell"] = (
        df["Median_Mitochondria_AreaShape_MaxFeretDiameter"]
        / df["Median_Mitochondria_AreaShape_MinFeretDiameter"]
    )
    df["Median_Lysosomes_DiameterRatio_PerCell"] = (
        df["Median_Lysosomes_AreaShape_MaxFeretDiameter"]
        / df["Median_Lysosomes_AreaShape_MinFeretDiameter"]
    )

    # mean and median area per organelle per cell area
    df["Mean_Mitochondria_Area_PerCell_Area"] = (
        df["Mean_Mitochondria_AreaShape_Area"] / df["AreaShape_Area"]
    )
    df["Mean_Lysosomes_Area_PerCell_Area"] = (
        df["Mean_Lysosomes_AreaShape_Area"] / df["AreaShape_Area"]
    )
    df["Median_Mitochondria_Area_PerCell_Area"] = (
        df["Median_Mitochondria_AreaShape_Area"] / df["AreaShape_Area"]
    )
    df["Median_Lysosomes_Area_PerCell_Area"] = (
        df["Median_Lysosomes_AreaShape_Area"] / df["AreaShape_Area"]
    )

    # mitolyso related
    df["Children_Lysosomes_Mitochondria_Ratio"] = (
        df["Children_Lysosomes_Count"] / df["Children_Mitochondria_Count"]
    )
    df["Density_Lysosomes_Mitochondria_Ratio"] = (
        df["OccupiedAreaFraction_Lysosomes_PerCell_Area"]
        / df["OccupiedAreaFraction_Mitochondria_PerCell_Area"]
    )
    df["Area_Lysosomes_Mitochondria_Ratio"] = (
        df["Math_Total_Lysosomes_AreaShape_Area_PerCell"]
        / df["Math_Total_Mitochondria_AreaShape_Area_PerCell"]
    )

    # Ratio of centroid distance to minimum distance for mitochondria and lysosomes
    df["Mean_Mitochondria_Distance_Centroid_Minimum_Ratio"] = (
        df["Mean_Mitochondria_Distance_Centroid_Nuclei"]
        / df["Mean_Mitochondria_Distance_Minimum_Nuclei"]
    )
    df["Mean_Lysosomes_Distance_Centroid_Minimum_Ratio"] = (
        df["Mean_Lysosomes_Distance_Centroid_Nuclei"]
        / df["Mean_Lysosomes_Distance_Minimum_Nuclei"]
    )

    # transform the mitoends - number of ends times the mean to get total per cell
    df["MitoEnds_Math_Total_NumberBranchEnds_MitoSkeleton"] = (
        df["Children_MitoEnds_Count"]
        * df["Mean_MitoEnds_ObjectSkeleton_NumberBranchEnds_MitoSkeleton"]
    )
    df["MitoEnds_Math_Total_NumberNonTrunkBranches_MitoSkeleton"] = (
        df["Children_MitoEnds_Count"]
        * df["Mean_MitoEnds_ObjectSkeleton_NumberNonTrunkBranches_MtSkltn"]
    )
    df["MitoEnds_Math_Total_NumberTrunks_MitoSkeleton"] = (
        df["Children_MitoEnds_Count"]
        * df["Mean_MitoEnds_ObjectSkeleton_NumberTrunks_MitoSkeleton"]
    )
    df["MitoEnds_Math_TotalObjectSkeltnLngth_MitoSkeleton_PerCell"] = (
        df["Children_MitoEnds_Count"]
        * df["Mean_MitoEnds_ObjectSkeleton_TotalObjectSkeltnLngth_MtSkltn"]
    )
    df["MitoEnds_Total_ObjectSkeltnLngth_MitoSkeleton_PerCell_Area"] = (
        df["MitoEnds_Math_TotalObjectSkeltnLngth_MitoSkeleton_PerCell"]
        / df["AreaShape_Area"]
    )
    return df


def make_feature_dict(columns_list):
    """
    Create a dictionary of features from the columns list.
    Args:
        columns_list (list): List of column names from the DataFrame.
    Returns:
        dict: A dictionary with keys as feature types and values as lists of corresponding column names."""
    # Add the different types of features to a dictionary
    feature_dict = {
        "intensity": [],
        "texture": [],
        "areashape": [],
        "granularity": [],
        "radialdistribution": [],
        "totals": [],
        "count": [],
        "distance": [],
        "per_cell_area": [],
        "coloc": [],
        "other": [],
    }
    for col in columns_list:
        if "Texture" in col:
            feature_dict["texture"].append(col)
        elif "Intensity" in col:
            feature_dict["intensity"].append(col)
        elif "Math_" in col or "Corr_" in col:
            feature_dict["totals"].append(col)
        elif "Count" in col:
            feature_dict["count"].append(col)
        elif "AreaShape" in col:
            feature_dict["areashape"].append(col)
        elif "Distance" in col:
            feature_dict["distance"].append(col)
        elif "PerCell" in col:
            feature_dict["per_cell_area"].append(col)
        elif "Granularity" in col:
            feature_dict["granularity"].append(col)
        elif "RadialDistribution" in col:
            feature_dict["radialdistribution"].append(col)
        elif "Lysosomes_Mitochondria_Ratio" in col:
            feature_dict["coloc"].append(col)
        else:
            feature_dict["other"].append(col)

    return feature_dict


def multinucleate_cells(df):
    multinuc_df = df[df["Cell_Classify_multinucleate"] == 1]
    return multinuc_df


def filter_saturated_cells(df):
    """Filter a cellprofiler output dataframe to only have cells classified as "normal" by having pixels vales below 65535

    Args:
        df (DataFrame): Cellprofiler output

    Returns:
        DataFrame: the filtered dataframe
    """
    not_saturated_df = df["Cell_Classify_Normal"] == 1
    return not_saturated_df


def plate_df_setup_fromcsv(
    curr_plates,
    curr_plate_datafolders,
    parent_dir,
    csv_names=[
        "Cell.csv",
        "Nuclei.csv",
        "MergedMitoPerCell.csv",
        "MergedLysoPerCell.csv",
    ],
):
    """
    Combine the cellprofiler feature data from different plates into a single DataFrame
    Returns a DataFrame with the combined data
    """
    # Initialize a list to store the combined DataFrames
    plate_dfs = {}

    for i, plate in enumerate(curr_plates):
        # Construct the full path to the folder
        folder_path = os.path.join(parent_dir, plate)

        # Construct the full path to the metadata file and CSV file
        map_file = os.path.join(folder_path, "metadata/map.csv")
        csv_folder_path = os.path.join(folder_path, curr_plate_datafolders[i])

        # Make a list of the csv file paths for each compartment
        compartment_paths = []

        for file in csv_names:
            cp_file = os.path.join(csv_folder_path, file)
            if os.path.exists(cp_file) and file in csv_names:
                compartment_paths.append(cp_file)

        # Join the file dataframes
        if "Cell.csv" in compartment_paths[0]:
            pre_cell_df = pd.read_csv(compartment_paths[0])
        else:
            return FileNotFoundError("Cell.csv not found in the folder")

        for j, compartment in enumerate(compartment_paths):
            if j == 0 and "Cell.csv" in compartment:
                continue

            compartment_df = pd.read_csv(compartment)
            excluded_columns = ["ImageNumber", "ObjectNumber"]

            prefix = csv_names[j].replace(".csv", "") + "_"

            keys_df = compartment_df[excluded_columns]
            excluded_keys_df = compartment_df.drop(columns=excluded_columns)

            prefixed_compartment_df = excluded_keys_df.add_prefix(prefix)
            combined_prefixed_compartment_df = pd.concat(
                [keys_df, prefixed_compartment_df], axis=1
            )

            pre_cell_df = pre_cell_df.merge(
                combined_prefixed_compartment_df,
                on=["ImageNumber", "ObjectNumber"],
                how="left",
            )

        # Join the metadata with the data
        if os.path.exists(cp_file) and os.path.exists(map_file):
            # Read the metadata file and merge with dataframes (map.csv)
            platemap_df = pd.read_csv(map_file)
            cell_df = pre_cell_df.merge(
                platemap_df,
                on=[
                    "Metadata_Well",
                    "Metadata_WellRow",
                    "Metadata_WellColumn",
                    "Metadata_Field",
                ],
                how="left",
            )

            # Add a column to the cell_df to group passages and identify the plate replicate
            cell_df["Passage Group"] = cell_df["PassageNumber"].apply(passage_group)
            cell_df["Metadata_Plate"] = plate
            cell_df["Replicate_Number"] = i + 1
            # Append the merged DataFrame to the list
            plate_dfs[plate] = cell_df

    # Combine all the different replicate DataFrames into a single DataFrame
    combined_replicates_df = pd.concat(plate_dfs.values(), ignore_index=True)

    # Filter DataFrames to only include cells that were stained with LAMP1-488 and MitoRed
    combined_replicates_df_mitolyso = combined_replicates_df[
        combined_replicates_df["Staining"].str.startswith("LAMP1-488 + MitoRed")
    ]
    return combined_replicates_df_mitolyso


def calculate_aggregated_object_features(
    parent_df,
    object_df,
    feature,
    parent_key,
    child_key="Cell_Number_Object_Number",
    aggregation="Median",
):
    """
    Calculate the aggregated function (typically median) of specified features grouped by a parent key.

    Parameters:
    df (DataFrame): The DataFrame containing the features.
    object_df (DataFrame): The DataFrame containing the object features.
    feature (str): The feature column to calculate the median for.
    child_key (str): The column name in the PARENT table that identifies the child obj.
    parent_key (str): The column name in the CHILD table that identifies the parent key.
    aggregation (str): the type of aggregation, can be "Mean","Median","Mode", or "Std"

    Returns:
    modified_df (DataFrame): The DataFrame with the new median feature column added.

    """
    agg_title = aggregation.title()  # make it title case for the column syntax
    if agg_title == "Median":
        agg_values = object_df.groupby([parent_key, "ImageNumber"])[feature].median()
    elif agg_title == "Mean" or agg_title == "Avg" or agg_title == "Average":
        agg_title = "Averaged"
        agg_values = object_df.groupby([parent_key, "ImageNumber"])[feature].mean()
    elif agg_title == "Sum":
        agg_title = "Total"
        agg_values = object_df.groupby([parent_key, "ImageNumber"])[feature].sum()
    elif agg_title == "Max":
        agg_values = object_df.groupby([parent_key, "ImageNumber"])[feature].max()
    elif agg_title == "Min":
        agg_values = object_df.groupby([parent_key, "ImageNumber"])[feature].min()
    elif agg_title == "Std":
        agg_values = object_df.groupby([parent_key, "ImageNumber"])[feature].std()
    else:
        ValueError('aggregation (str) not in "Mean","Median","Mode", or "Std"')
        return pd.DataFrame()

    # create the new column name for the aggregated feature
    col_name = f"Cell_{agg_title}_{feature}"

    # reformat the agg_values dataframe for merging
    agg_values = agg_values.reset_index()
    agg_values.columns = [child_key, "ImageNumber", col_name]
    agg_values["CellNumber_ImageNumber_Index"] = (
        agg_values[child_key].astype(str) + "_" + agg_values["ImageNumber"].astype(str)
    )

    # Merge the aggregated values back into the parent dataframe
    modified_df = parent_df.copy()
    # add a column to parent_df to merge on
    modified_df["CellNumber_ImageNumber_Index"] = (
        modified_df[child_key].astype(str)
        + "_"
        + modified_df["ImageNumber"].astype(str)
    )

    modified_merged_df = modified_df.merge(
        agg_values[["CellNumber_ImageNumber_Index", col_name]],
        how="left",
        left_on="CellNumber_ImageNumber_Index",
        right_on="CellNumber_ImageNumber_Index",
    )

    return modified_merged_df


# group by parent key to find median


def add_median_object_features_to_parent(
    parent_df, object_df, object_name, child_key="Cell_Number_Object_Number"
):
    """
    Add median features from object_df to parent_df based on the specified feature and keys.

    Parameters:
    parent_df (DataFrame): The DataFrame containing the parent features (e.g cell).
    object_df (DataFrame): The DataFrame containing the object features (e.g. mitochondria).
    object_name (str): The object of interest to add to the parent table.
    child_key (str): The column name in the PARENT table that identifies the child obj.

    Returns:
    modified_df (DataFrame): The DataFrame with the new median feature column added.

    """
    # print(parent_df)
    modified_df = parent_df.copy()
    feature_types = ["AreaShape", "Distance", "Intensity", "Location"]
    features_to_exclude = [
        "Zernike",
        "Maximum_X",
        "Maximum_Y",
        "Minimum_X",
        "Mimimum_Y",
        "Centroid_X",
        "Centroid_Y",
    ]
    other_channels = ["DAPI", "LAMP1", "MitoTracker", "Phalloidin"]
    # use the other channels list as a check to discard irrelavent features

    if "Lysosomes" in object_name:
        parent_key = "Lysosomes_Parent_Cell"
        other_channels.remove("LAMP1")
    elif "Mitochondria" in object_name:
        parent_key = "Mitochondria_Parent_Cell"
        other_channels.remove("MitoTracker")
    elif "Nuclei" in object_name:
        parent_key = "Nuclei_Parent_Cell"
        other_channels.remove("DAPI")
    elif "MitoEnds" in object_name:
        parent_key = "MitoEnds_Parent_Cell"
        other_channels.remove("MitoTracker")
    else:
        raise ValueError(
            f"Unknown object name: {object_name}. Expected 'Lysosomes', 'Mitochondria', 'MitoEnds', or 'Nuclei'."
        )

    for feature in object_df.columns:
        # print("checking feature:", feature)
        # Exclude if matches any in other_channels, unless it also matches feature_types
        if object_name not in feature or (
            any(ch in feature for ch in other_channels)
            or any(exclusion in feature for exclusion in features_to_exclude)
        ):
            # print("oops, skipping:", feature)
            continue  # Skip the parent key column or non-feature columns
        if any(ft in feature for ft in feature_types):
            modified_df = calculate_aggregated_object_features(
                modified_df,
                object_df,
                feature,
                parent_key,
                child_key,
                aggregation="Median",
            )

    return modified_df


def load_organelle_medians(db_path, df, organelles=["Lysosomes", "Mitochondria"]):
    """
    Load organelle median features from the database.

    Parameters:
    db_path (str): Path to the database file containing the extra features
    df (DataFrame): the Pandas dataframe to add the medians to.
    organelle (str): Name of the organelle (e.g., 'Lysosomes', 'Mitochondria', 'Nuclei').

    Returns:
    DataFrame: DataFrame containing the median features for the specified organelle.
    """
    import gc
    import sqlite3

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    cell_df = df.copy()

    conn = sqlite3.connect(db_path)
    # open each one individually to not explode the ram
    for organelle in organelles:
        query = f"SELECT * FROM Per_{organelle}"
        org_df = pd.read_sql_query(query, conn)
        cell_df = add_median_object_features_to_parent(
            parent_df=cell_df, object_df=org_df, object_name=organelle
        )
        del org_df  # Free memory
        gc.collect()

    # conn.close()
    return cell_df


def exclude_borders(
    df, min_x=0.0, min_y=0.0, max_x=2160.0, max_y=2160.0, prefix="Cell_"
):
    """
    Exclude rows in the DataFrame that have bounding box coordinates outside the specified limits.

    Parameters:
    df (DataFrame): The DataFrame containing the bounding box coordinates.
    min_x (float): Minimum x-coordinate for exclusion.
    min_y (float): Minimum y-coordinate for exclusion.
    max_x (float): Maximum x-coordinate for exclusion.
    max_y (float): Maximum y-coordinate for exclusion.

    Returns:
    DataFrame: Filtered DataFrame with rows excluded based on bounding box coordinates.
    """
    filtered_df = df[
        (df[f"{prefix}AreaShape_BoundingBoxMaximum_X"] < max_x)
        & (df[f"{prefix}AreaShape_BoundingBoxMaximum_Y"] < max_y)
        & (df[f"{prefix}AreaShape_BoundingBoxMinimum_X"] > min_x)
        & (df[f"{prefix}AreaShape_BoundingBoxMinimum_Y"] > min_y)
    ]
    return filtered_df


def well_namer(row, col):
    """
    Convert row and column numbers to a well name in the format A01, B02, etc.

    Args:
        row (int): The row number (1-8)
        col (int): The column number (1-12)

    Returns:
        str: Well name in the format A01, B02, etc.
    """
    well_name = str(chr(ord("@") + row)) + str(col).rjust(
        2, "0"
    )  # make the number have a left align, adding a zero
    return well_name


def add_well_metadata(image_df):
    """
    Add well metadata to the image DataFrame.

    Args:
        image_df (DataFrame): DataFrame containing image metadata

    Returns:
        DataFrame: Updated DataFrame with well metadata
    """
    image_df.columns = image_df.columns.str.replace(
        r"^Image_Metadata_", "Metadata_", regex=True
    )
    image_df[["Metadata_WellRow", "Metadata_WellColumn", "Metadata_Field"]] = image_df[
        "Image_URL_DAPI"
    ].str.extract(r"r(\d{2})c(\d{2})f(\d{2}).tif")
    # Convert extracted columns to int
    image_df["Metadata_WellRow"] = image_df["Metadata_WellRow"].astype(int)
    image_df["Metadata_WellColumn"] = image_df["Metadata_WellColumn"].astype(int)
    image_df["Metadata_Field"] = image_df["Metadata_Field"].astype(int)
    # apply well namer function
    image_df["Metadata_Well"] = image_df.apply(
        lambda x: well_namer(x["Metadata_WellRow"], x["Metadata_WellColumn"]), axis=1
    )

    return image_df


def update_database_with_well_metadata(db_path):
    """
    Update the database with well metadata.

    Args:
        db_path (str): Path to the database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Read Per_Image table
    image_df = pd.read_sql_query("SELECT * FROM Per_Image", conn)

    # Add well metadata
    updated_image_df = add_well_metadata(image_df)

    # Write updated DataFrame back to the database
    try:
        updated_image_df.to_sql("Per_Image", conn, if_exists="replace", index=False)
        print("Database updated successfully with well metadata.")
    except Exception as e:
        print(f"Error updating database: {e}")
    # cursor.execute("SELECT Metadata_Well FROM Per_Image LIMIT 5;")
    # cursor.fetchall()
    conn.close()


def standardize_group(df, columns):
    """_summary_

    Args:
        df (_type_): _description_
        columns (_type_): _description_

    Returns:
        _type_: _description_
    """
    from sklearn.preprocessing import StandardScaler

    # Import the scaler and transform all time values to that of a standard distribution - only use for ML, not very desceiptive
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df[columns])
    return scaled_df


def group_by_condition(df, feature_list, groupby_column="AgeGroup"):
    """Group by a condition

    Args:
        df (_type_): _description_
        feature_list (_type_): _description_
        groupby_column (str, optional): _description_. Defaults to "AgeGroup".

    Returns:
        _type_: _description_
    """
    # Group columns by age group and apply groupby function to the DF
    df_groupby = df.groupby(groupby_column).apply(
        lambda x: standardize_group(x, feature_list)
    )
    return df_groupby


def average_groups_by_plate(df, x_value, y_value, replicates):
    """
    Group the DataFrame by the specified columns and calculate the mean of the y_value column.
    Returns the averaged dataframe for plotting

    Args:
        df (DataFrame): your dataframe
        x_value (string): the grouping variable (x value)
        y_value (string): the quantitavie feature to measure (y value)
        replicates (string): the variable representing experimental replicates for grouping

    Returns:
        DataFrame: your data grouped by replicate
    """
    df = df.dropna(subset=[x_value, y_value, replicates])
    df = df[df[y_value] != 0]

    df.reset_index(drop=True, inplace=True)

    group_averages = df.groupby(
        [x_value, replicates], as_index=False, observed=True
    ).agg({y_value: "mean"})

    # Reset the index to get a clean DataFrame
    average_df = group_averages.reset_index()

    return average_df


def normalize_features(df, feature_list):
    """
    Normalize the features in the DataFrame to the control (age group 0) for each plate.
    Args:
        df (DataFrame): The DataFrame containing the features to be normalized.
        feature_list (list): A list of feature column names to normalize.
    Returns:
        DataFrame: A DataFrame with normalized features for each plate.
    """
    # Normalize the features to the control (age group 0) for each plate
    norm_df = df.copy()
    for feature in feature_list:
        # print('Normalizing feature: ', feature, '...', norm_df[feature].values[0])
        norm_df[feature] = normalize_to_control(df, feature)
        # print('Normalized feature: ', feature, '...', norm_df[feature].values[0])
    return norm_df


def normalize_to_control(df, feature, norm_column="AgeGroup"):
    """
    Normalize a feature to the control group (AgeGroup = 0) for each plate.
    Args:
        df (DataFrame): The DataFrame containing the feature to be normalized.
        feature (str): The name of the feature column to normalize.
        norm_column (str): The column used to identify the control group (default is 'AgeGroup').'`
    Returns:
        Series: A Series containing the normalized feature values.
    """
    # Take the t0 df - lowest passage data point
    t0_df = df[df[norm_column] == 0]
    treatment_df = df[[feature, norm_column]].copy()

    # calculate the mean
    mean_zero = t0_df[feature].mean()
    # Check for non-numeric values
    if not pd.api.types.is_numeric_dtype(treatment_df[feature]):
        print(f"[normalize_to_control] WARNING: {feature} is not numeric!")
    # now update the column to have all rows dividied by the mean of group 0
    treatment_df["norm_" + feature] = treatment_df[feature] / mean_zero
    # return the normalized feature columnn
    return treatment_df["norm_" + feature]


def apply_feature_normalization(df, feature_dict, curr_plates):
    """
    Apply feature normalization to the DataFrame for each plate in a list of plates.
    Args:
        df (DataFrame): The DataFrame containing the features to be normalized.
        feature_dict (dict): A dictionary containing lists of feature columns to normalize.
        curr_plates (list): A list of plate names to apply normalization to.
    Returns:
        DataFrame: A DataFrame with normalized features for each plate.
    """
    # Normalize the features to the control (age group 0) for each plate
    norm_cell_df = df.copy()
    for plate in curr_plates:
        curr_plate_df = norm_cell_df[norm_cell_df["Metadata_Plate"] == plate].copy()
        for feature_type in feature_dict:
            # get the normalized features, locate the corresponding features on the plate, and replace them on that plate to the plate
            curr_plate_features_df = normalize_features(
                curr_plate_df, feature_dict[feature_type]
            )
            curr_plate_df.loc[:, feature_dict[feature_type]] = curr_plate_features_df[
                feature_dict[feature_type]
            ].astype(float)
        norm_cell_df.loc[norm_cell_df["Metadata_Plate"] == plate] = curr_plate_df
    return norm_cell_df


def mean_intesity_per_compartment_per_cell(df, compartment, tag):
    """_summary_

    Args:
        df (_type_): _description_
        compartment (_type_): _description_
        tag (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Calculate the mean intensity of each compartment per cell
    # mean_intesity_per_compartment = integrated / (children*mean_area)
    colname = "MeanIntensity_Per_" + compartment + "_Per_Cell"
    integrated = "Intensity_IntegratedIntensity_" + tag
    children = "Children_" + compartment + "_Count"
    mean_area = "Mean_" + compartment + "_AreaShape_Area"
    df[colname] = df.apply(
        lambda x: x[integrated] / (x[children] * x[mean_area]), axis=1
    )
    return df[colname]


def proportion_area_occupied_per_cell(df, compartment):
    """_summary_

    Args:
        df (_type_): _description_
        compartment (_type_): _description_

    Returns:
        _type_: _description_
    """
    # proportion of area occupied = children * mean organelle area / cell area
    colname = "Total_Area_Proportion_" + compartment + "_Per_Cell"

    # children = 'Children_' + compartment + '_Count'
    # mean_organelle_area = 'Mean_'+ compartment + '_AreaShape_Area'
    organelle_area = compartment + "_AreaShape_Area"
    cell_area = "AreaShape_Area"
    # df[colname] = df.apply(lambda x: (x[children] * x[mean_organelle_area]) / x[cell_area], axis=1)
    df[colname] = df.apply(lambda x: (x[organelle_area]) / x[cell_area], axis=1)
    return df[colname]


def proportion_area_occupied_per_cell_fromtotal(df, compartment):
    """_summary_

    Args:
        df (DataFrame): _description_
        compartment (string): _description_

    Returns:
        Series: The column to add
    """
    # proportion of area occupied = children * mean organelle area / cell area
    colname = "Total_Area_Proportion_" + compartment + "_Per_Cell"

    # children = 'Children_' + compartment + '_Count'
    # mean_organelle_area = 'Mean_'+ compartment + '_AreaShape_Area'
    organelle_area = compartment + "_AreaShape_Area"
    cell_area = "AreaShape_Area"
    # df[colname] = df.apply(lambda x: (x[children] * x[mean_organelle_area]) / x[cell_area], axis=1)
    df[colname] = df.apply(lambda x: (x[organelle_area]) / x[cell_area], axis=1)
    return df[colname]


def mean_intesity_per_compartment_per_cell_fromtotal(df, compartment, name, tag):
    """_summary_

    Args:
        df (DataFrame): _description_
        compartment (_type_): _description_
        name (_type_): _description_
        tag (_type_): _description_

    Returns:
        Series: the column to add
    """
    # Calculate the mean intensity of each compartment per cell
    # mean_intesity_per_compartment = integrated / (children*mean_area)
    colname = "MeanIntensity_Per_" + compartment + "_Per_Cell"
    integrated = "Intensity_IntegratedIntensity_" + tag
    # children = 'Children_' + compartment + '_Count'
    # mean_area = 'Mean_'+ compartment + '_AreaShape_Area'
    total_organelle_area = name + "_AreaShape_Area"
    df[colname] = df.apply(lambda x: x[integrated] / x[total_organelle_area], axis=1)
    return df[colname]


def cell_nuc_area_ratio(
    df,
    cell_area_col="Cell_AreaShape_Area",
    nuc_area_col="Nuclei_AreaShape_Area",
    ratio_col_name="Cell_Nuclei_Area_Ratio",
):
    """
    Calculate the ratio of cell area to nuclear area.

    Args:
        df (Series): A DataFrame containing 'Cell_AreaShape_Area' and 'Nuclei_AreaShape_Area' cols.

    Returns:
        Series: The column with the cell/nuc area ratio column to be added
    """
    df_overzero = df[df[nuc_area_col] > 0]
    final_df = df_overzero.dropna(subset=[nuc_area_col]).reset_index(drop=True)
    final_df[ratio_col_name] = final_df[cell_area_col] / final_df[nuc_area_col]
    return final_df[ratio_col_name]


def make_single_feature_df(data, group, feature, replicates):
    """Make a dataframe for a single feature from a larger dataframe in "tidy" format

    Args:
        data (DataFrame): your dataframe
        group (string): the grouping variable (x value)
        feature (string): the quantitavie feature to measure (y value)
        replicates (string): the variable representing experimental replicates for grouping

    Returns:
        _type_: _description_
    """
    pd.options.mode.copy_on_write = True

    subset = [group, feature, replicates]

    df = data.dropna(subset=subset).reset_index(drop=True)
    df = df[df[feature] != 0]

    df_subset = df[subset]
    df_subset[group] = df[group].astype("category")
    df_subset.reset_index(drop=True, inplace=True)

    return df_subset


def relate_objects(
    obj_df_1,
    obj_df_2,
    obj1_name="",
    obj2_name="",
    feature_cols=[],
    ratio_colname="Cell_Nuclei_Area_Ratio",
    metadata_cols=[
        "ImageNumber",
        "Replicate_Number",
        "Metadata_WellRow",
        "Metadata_WellColumn",
        "Metadata_Field",
        "SerialPassage_BatchNumber",
        "AgeGroup",
        "Drug",
        "slice",
        "Filename",
        "Parent_Folder",
        "Path",
        "Metadata_Well_ID",
        "Metadata_Well_x",
        "Block",
        "Metadata_Well_y",
        "TimepointName",
        "Staining",
        "Passage Group",
        "AllGroups",
    ],
):
    """Relates two tables of segmented images
    Based off of ImageNumber and where objects in obj_df_2 are contained within larger objects in obj_df_1.
    NOTE: Bounding box (min_row, min_col, max_row, max_col). Pixels belonging to the bounding box are in the half-open interval [min_row; max_row) and [min_col; max_col)
    Args:
        obj_df_1 (pd.DataFrame): DataFrame for the first (larger) objects e.g. cells.
                                 Expected columns: "ImageNumber", "label", "bbox-0", "bbox-1", "bbox-2", "bbox-3".
                                 "bbox-0", "bbox-1" typically represent min_x, max_x.
                                 "bbox-2", "bbox-3" typically represent max_y, max_y.
        obj_df_2 (pd.DataFrame): DataFrame for the second (smaller, contained) objects e.g. nuclei.
                                 Expected columns: "ImageNumber", "label", "bbox-0", "bbox-1", "bbox-2", "bbox-3".
                                 "bbox-0", "bbox-1" typically represent min_x, max_x.
                                 "bbox-2", "bbox-3" typically represent max_y, max_y.
        obj1_name (str, optional): Name of the first object (e.g., "Cell"). Defaults to "".
        obj2_name (str, optional): Name of the second object (e.g., "Nucleus"). Defaults to "".

    Returns:
        pd.DataFrame: DataFrame with the related objects, including a "Parent_Obj1Name_Number_Object_Number"
                      column in the obj_df_2 data, and a combined DataFrame with aggregated means
                      and a calculated ratio.
    """
    obj_df_1 = obj_df_1.sort_values(by="ImageNumber").copy()
    obj_df_2 = obj_df_2.sort_values(by="ImageNumber").copy()

    related_second_objects = []
    for i, obj1_row in obj_df_1.iterrows():
        obj1_min_x, obj1_min_y, obj1_max_x, obj1_max_y = (
            # Bounding box (min_row, min_col, max_row, max_col)
            obj1_row["bbox-0"],
            obj1_row["bbox-1"],
            obj1_row["bbox-2"],
            obj1_row["bbox-3"],
        )
        parent_obj_label = obj1_row["label"]
        img_number = obj1_row["ImageNumber"]

        # Filter obj_df_2 to only use the current ImageNumber
        current_img_obj2_df = obj_df_2[obj_df_2["ImageNumber"] == img_number]

        for j, obj2_row in current_img_obj2_df.iterrows():
            if obj_df_2["ImageNumber"][j] > i:
                break
            # relate if the second object is contained within the parent object  (min_row, min_col, max_row, max_col)
            obj2_min_x, obj2_min_y, obj2_max_x, obj2_max_y = (
                obj2_row["bbox-0"],
                obj2_row["bbox-1"],
                obj2_row["bbox-2"],
                obj2_row["bbox-3"],
            )

            # assign conditions to booleans
            # Pixels belonging to the bounding box: [min_row; max_row) and [min_col; max_col)
            is_contained_x = (obj2_min_x >= obj1_min_x) and (obj2_max_x <= obj1_max_x)
            is_contained_y = (obj2_min_y >= obj1_min_y) and (obj2_max_y <= obj1_max_y)

            # also make a flag to check if second object is toucjing border
            touching_border_nuc = bool(
                (obj2_min_x == 0)
                or (
                    obj2_max_x == (max_x * 0.25)
                )  # multiply by in the origial downlampling factor from the masks (0.25)
                or (obj2_min_y == 0)
                or (obj2_max_y == (max_y * 0.25))
            )

            if is_contained_x and is_contained_y:
                # print(f"second object bbox: {obj2_row["slice"]}, touching border? {touching_border_nuc}")
                # Create a copy to avoid SettingWithCopyWarning
                second_obj_with_parent = obj2_row.copy()
                second_obj_with_parent[f"Parent_{obj1_name}_Number_Object_Number"] = (
                    parent_obj_label
                )
                second_obj_with_parent[f"{obj2_name}_Touching_Border"] = (
                    touching_border_nuc
                )
                related_second_objects.append(second_obj_with_parent)

    # case when you don't have any relationships
    if not related_second_objects:
        return pd.DataFrame  # empty df

    second_objs_related_df = pd.DataFrame(related_second_objects)
    # Aggregate means of the second objects by their assigned parent and ImageNumber - group by the parent label and ImageNumber for aggregation
    boolean_cols = second_objs_related_df.select_dtypes(include=bool).columns

    if not feature_cols:
        feature_cols = second_objs_related_df.select_dtypes(
            include=np.number
        ).columns.tolist()  # list of all numeric cols only

    # make a dictionary to tell pandas what aggregations to do
    aggregations = {col: "mean" for col in feature_cols}

    for col in boolean_cols:
        aggregations[col] = "mean"  # Proportion of True values

    for col in metadata_cols:
        aggregations[col] = "first"  # Assuming metadata is consistent within a group

    # Get a count column to see # of nuclei
    second_obj_counts = second_objs_related_df.groupby(
        ["ImageNumber", f"Parent_{obj1_name}_Number_Object_Number"]
    ).count()["area"]  # arbitrary col

    # aggregate using the dictionary above
    second_obj_means = second_objs_related_df.groupby(
        ["ImageNumber", f"Parent_{obj1_name}_Number_Object_Number"]
    ).agg(aggregations)  # Ensure only numeric columns are averaged

    second_obj_means[f"Children_{obj2_name}_Count"] = second_obj_counts

    # second_obj_means = second_objs_related_df.groupby(
    #     ["ImageNumber", f"Parent_{obj1_name}_Number_Object_Number"]
    # ).mean(numeric_only=True)  # Ensure only numeric columns are averaged
    second_obj_means = second_obj_means.rename_axis(  # change index of df to "label"
        index={f"Parent_{obj1_name}_Number_Object_Number".format(obj1_name): "label"}
    )

    # second_obj_means = second_obj_means[second_obj_means["label"] +1] #one-indexed
    # Join on ImageNumber and the label of the first object (which is the parent label in second_obj_means)
    joined_obj_df = obj_df_1.join(
        second_obj_means,
        on=["ImageNumber", "label"],
        how="left",
        rsuffix=f"_mean_{obj2_name}",
    )
    # display(joined_obj_df)
    # display(joined_obj_df.head(10), joined_obj_df.shape)
    # Calculate the ratio, ensuring the columns exist and handling potential NaNs
    if (
        f"{obj1_name}_AreaShape_Area" in joined_obj_df.columns
        and f"{obj2_name}_AreaShape_Area" in joined_obj_df.columns
    ):
        joined_obj_df[ratio_colname] = (
            joined_obj_df[f"{obj1_name}_AreaShape_Area"]
            / joined_obj_df[f"{obj2_name}_AreaShape_Area"]
        )
    else:
        print(
            f"Warning: 'AreaShape_Area' columns not found for ratio calculation. Expected: {obj1_name}_AreaShape_Area and {obj2_name}_AreaShape_Area_mean"
        )
        joined_obj_df[ratio_colname] = float("nan")

    # Drop rows where the ratio could not be calculated (due to missing second object data)
    relate_objects_df = joined_obj_df.dropna(subset=[ratio_colname])
    # second_obj_means = second_objs_df.groupby("ImageNumber").agg("mean")
    # joined_df = obj_df_1.join(second_obj_means, on=["ImageNumber","label"], x=obj1_name,y=obj2_name, how="left")
    # joined_df["CellNucRatio"] = joined_df.apply(lambda x: x[f"{obj1_name}_AreaShape_Area"]/x[f"{obj2_name}_AreaShape_Area"])

    return relate_objects_df


def combine_one_to_one_dfs(
    df, db_conn, tables_to_add=["Nuclei", "Cytoplasm"], main_df_prefix="Cell"
):
    """Merge table outputs from cellprofiler where the tables are objects in a one-to-one relationship e.g. cells and nuclei
    Args:
        df (DataFrame): _description_
        db_conn (sqlite3 Connection object): _description_
        tables_to_add (list of str, optional): _description_. Defaults to ["Nuclei","Cytoplasm"].

    Returns:
        DataFrame: _description_
    """
    # add all these to a list
    main_df = df.copy()
    dfs_to_add = []
    for table in tables_to_add:
        new_df = pd.read_sql_query(f"SELECT * FROM Per_{table};", db_conn)
        dfs_to_add.append(new_df)

    # now merge to the cell df
    for i, object_df in enumerate(dfs_to_add):
        compartment = tables_to_add[i]
        # special case for nuclei; where we want cell to be the primary table but nuclei are the parent of a cell
        if compartment == "Nuclei":
            main_df = pd.merge(
                main_df,
                object_df,
                how="left",  # left on the object index, image-by-imageCell_Parent_Nuclei
                left_on=[f"{main_df_prefix}_Parent_{compartment}", "ImageNumber"],
                right_on=[f"{compartment}_Number_Object_Number", "ImageNumber"],
            )
        else:
            main_df = pd.merge(
                main_df,
                object_df,
                how="left",  # left on the object index, image-by-imageCell_Parent_Nuclei
                left_on=[f"{main_df_prefix}_Number_Object_Number", "ImageNumber"],
                right_on=[f"{compartment}_Parent_{main_df_prefix}", "ImageNumber"],
            )

    merged_df_final = main_df.reset_index(drop=True)
    return merged_df_final
