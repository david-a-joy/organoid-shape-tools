""" Aggregate shape analysis

Segment a folder of phase aggregate images:

.. code-block:: python

    analyze_aggregate_shape('path/to/image/folder')

For each ``.tif`` file in the folder, this will write out two files:

* ${filename}_plot.png - The segmentation plot
* ${filename}_stats.csv - The statistics for each segmentation

See the ``./analyze_aggregate_shape.py`` script for details

Functions:

* :py:func:`analyze_aggregate_shape`: Analyze an entire folder of aggregate images
* :py:func:`analyze_aggregate_image`: Analyze and segment a single image
* :py:func:`extract_image_stats`: Extract segmentation statistics from single images
* :py:func:`load_convexity_contours`: Load contours and do a convexity analysis
* :py:func:`histogram_equalize`: Histogram equalize images
* :py:func:`get_nearest_area`: Get the nearest area in the bin

Per-Experiment Metadata:

* :py:func:`categorize_knockdown`: Categorize the knockdown images by type
* :py:func:`categorize_chir_dosing`: Categorize the CHIR dosing images by type

Classes:

* :py:class:`SegmentationParams`: Parameters for the segmentation algorithm
* :py:class:`RegionData`: Region data object

API Documentation
-----------------

"""

# Standard lib
import shutil
import pathlib
import traceback
from collections import namedtuple
from collections.abc import Mapping
from typing import Optional, List, Dict

# 3rd party
import numpy as np

from scipy import ndimage as ndi

from skimage.morphology import (
    remove_small_holes, binary_erosion, binary_dilation, remove_small_objects,
    skeletonize, convex_hull_image
)
from skimage.segmentation import random_walker
from skimage.exposure import equalize_adapthist
from skimage.measure import regionprops
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.transform import resize

from sklearn.neighbors import BallTree

import matplotlib.pyplot as plt

import networkx as nx

import pandas as pd

# Our own imports
from .utils import (
    load_image, mask_from_contours, contours_from_mask, perimeter_of_polygon,
    scale_polygon, poly, Hypermap)
from .plotting import set_plot_style, colorwheel, add_scalebar

# Constants

CONVEX_RESCALE_SIZE = 512  # pixel - rescale size for the images during segmentation

CONVEX_MIN_SIZE = 1000  # pixel - minimum area allowed to be a "defect"

THRESHOLD = 0.05  # 0.0 to 1.0 - Threshold between dark aggregates and bright background

BORDER_PIXELS = 5  # px - Ignore any masks that are this close to the border

SPACE_SCALE = 1000/466  # um/pixel - How many microns does a single (original) pixel represent

RESIZE_X, RESIZE_Y = 512, 512  # Resize the image to this many pixels

IMAGE_NORM = 2  # How to normalize the image: 1 - np.abs(img), 2 - img**2

SEGMENTATION_BETA = 1200  # Diffusion constant for the segmentation

IMAGE_SIGMA = 0.5  # width of the gaussian smoothing kernel (< 0 to disable)
BACKGROUND_SIGMA = -1  # width of the gaussian kernel to smooth the background (< 0 to disable)

MIN_HOLE_SIZE = 500  # Minimum size for holes in the segmentation
MIN_OBJECT_SIZE = 700  # Minimum size for objects in the segmentation
MIN_MASK_SIZE = 400  # Minimum size of the mask
MAX_MASK_SIZE = -1  # Maximum size of the mask

CALC_SPINES = False  # If True, calculate spine statistics

SAVE_INDIVIDUAL_STATS = True  # Save the individual stats plots
PLOT_INDIVIDUAL_ROIS = True  # Plot the individual ROI traces

DARK_AGGREGATES = False  # If True, assume dark aggregates on a light background

SUFFIX = '.png'

# Classes

SegmentationParams = namedtuple('SegmentationParams', [
    'threshold', 'border_pixels', 'space_scale',
    'image_sigma', 'background_sigma', 'image_norm',
    'min_hole_size', 'min_object_size', 'min_mask_size',
    'max_mask_size', 'resize_x', 'resize_y', 'dark_aggregates',
    'segmentation_beta',
])


class RegionData(Mapping):
    """ Fake regionprop object to allow for calculated values """

    def __init__(self, **kwargs):
        self._data = kwargs

    @property
    def csv_header(self) -> str:
        return ','.join(k.capitalize() for k in self._data.keys())

    @property
    def csv_data(self) -> str:
        return ','.join(f'{v}' for v in self._data.values())

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

# Functions


def load_convexity_contours(rootdir: pathlib.Path,
                            outfile: pathlib.Path,
                            merge_df: pd.DataFrame,
                            overwrite: bool = False,
                            rescale_size: int = CONVEX_RESCALE_SIZE,
                            min_size: int = CONVEX_MIN_SIZE) -> pd.DataFrame:
    """ Calculate contour stats for all the contours in merge_df

    :param Path rootdir:
        The directory to process
    :param Path outfile:
        The file to save the final contour and stat results to
    :param DataFrame merge_df:
        The merged segmentations for all the contours
    :param bool overwrite:
        If True, overwrite the cache file
    :returns:
        The stats for the merged contours
    """
    if overwrite and outfile.is_file():
        outfile.unlink()
    if outfile.is_file():
        return pd.read_hdf(outfile, key='stats')

    # Index the root directory for quick lookup
    daydirs = {}
    for p in rootdir.iterdir():
        if p.name.startswith('.'):
            continue
        if not (p / 'Segmentations').is_dir():
            continue
        if overwrite:
            if (p / 'cache').is_dir():
                shutil.rmtree(p / 'cache')
        daydirs[p.name.lower()] = p

    print(f'Generating contour stats for {rootdir}')
    all_stat_df = []
    for group, group_df in merge_df.groupby('FileName'):
        stat_df = {
            'FileName': [],
            'Label': [],
            'Day': [],
            'Condition': [],
            'NumDay': [],
            'MajorAxis': [],
            'MinorAxis': [],
            'AspectRatio': [],
            'ROIArea': [],
            'ConvexArea': [],
            'ROIPerimeter': [],
            'ConvexPerimeter': [],
            'ROICircularity': [],
            'ConvexCircularity': [],
            'NumDefects': [],
            'DefectAreaUnfiltered': [],
            'DefectAreaFiltered': [],
            'PctDefectUnfiltered': [],
            'PctDefectFiltered': [],
        }

        # Figure out where the files live
        daykey = group.split('-')[:4]
        daykey = '-'.join(daykey).lower()
        daydir = daydirs[daykey.lower()]
        imagedir = daydir / 'Segmentations' / group

        # See if we want to use the cache files
        cachefile = daydir / 'cache' / f'{group}_hands.h5'
        if cachefile.is_file():
            stat_df = pd.read_hdf(cachefile, key='stats')
            print(stat_df.head())
            print(stat_df.describe())
            all_stat_df.append(stat_df)
            continue

        if not imagedir.is_dir():
            raise OSError(f'Failed to find segmentation data for {imagedir}')

        imagefile = daydir / f'{group}.tif'
        if not imagefile.is_file():
            raise OSError(f'Failed to find image data for {imagefile}')

        # Load the image
        img = load_image(imagefile, ctype='gray')
        rows, cols = img.shape[:2]
        aspect = float(cols) / float(rows)

        resize_x = rescale_size
        resize_y = int(round(resize_x * aspect))

        # Fix the stupid rescaling that happens during segmentation
        scale_x = img.shape[0] / resize_x
        scale_y = img.shape[1] / resize_y

        # Load the contours
        contour_file = imagedir / f'{group}_contours.xlsx'
        contour_df = pd.read_excel(contour_file)

        contour_df['XCoord'] *= scale_x
        contour_df['YCoord'] *= scale_y

        # Pull out keys for generic attributes
        group_day = group_df['Day'].values[0]
        group_num_day = group_df['NumDay'].values[0]
        group_condition = group_df['Condition'].values[0]
        group_perimeter = group_df['Perimeter'].values[0]
        group_major_axis = group_df['MajorAxis'].values[0]
        group_minor_axis = group_df['MinorAxis'].values[0]

        if group_major_axis < group_minor_axis:
            group_major_axis, group_minor_axis = group_minor_axis, group_major_axis

        # Merge with the filtered metadata
        merge_df = group_df[['Label']].merge(
            contour_df[['Label', 'XCoord', 'YCoord']],
            how='left', left_on='Label', right_on='Label', validate='one_to_many')

        # Pull out each contour and analyze
        for label_id in np.unique(merge_df[['Label']]):
            mask_df = merge_df[merge_df['Label'] == label_id]
            xcoords = mask_df['XCoord'].values
            ycoords = mask_df['YCoord'].values

            # Convert the segmentation to a convex hull
            mask = mask_from_contours([np.stack([xcoords, ycoords], axis=1)], (rows, cols))
            for i in range(5):
                mask = binary_dilation(mask)
            for i in range(5):
                mask = binary_erosion(mask)
            convex_mask = convex_hull_image(mask)
            convex_coords = contours_from_mask(convex_mask, 0.5)[0]
            convex_perimeter = perimeter_of_polygon(convex_coords)

            bbox_x0 = int(np.floor(np.min(convex_coords[:, 1]) - 10))
            bbox_x1 = int(np.ceil(np.max(convex_coords[:, 1]) + 10))
            bbox_y0 = int(np.floor(np.min(convex_coords[:, 0]) - 10))
            bbox_y1 = int(np.ceil(np.max(convex_coords[:, 0]) + 10))

            bbox_x0 = max([0, bbox_x0])
            bbox_x1 = min([rows, bbox_x1])
            bbox_y0 = max([0, bbox_y0])
            bbox_y1 = min([cols, bbox_y1])

            # Work out the size of the convex hull and defects
            mask_area = np.sum(mask)
            convex_area = np.sum(convex_mask)

            delta_mask = np.logical_and(convex_mask, ~mask)
            delta_mask = remove_small_objects(delta_mask, min_size)

            delta_labels = ndi.label(delta_mask)[0]
            num_defects = np.unique(delta_labels).shape[0] - 1

            # Calculate areas
            defect_area_unfiltered = convex_area - mask_area
            defect_area_filtered = np.sum(delta_labels > 0)

            print(group, label_id, num_defects)

            if label_id == 6:
                convex_coords = scale_polygon(convex_coords, 1.02)
                exterior_coords = contours_from_mask(mask, [0.5])[0]
                with set_plot_style('light') as style:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                    ax.imshow(img, cmap='gray')
                    ax.plot(convex_coords[:, 0], convex_coords[:, 1], '-b', linewidth=2)
                    ax.plot(exterior_coords[:, 0], exterior_coords[:, 1], '-r', linewidth=2)
                    print(bbox_x0, bbox_x1)
                    print(bbox_y1, bbox_y0)

                    # ax.set_xlim([bbox_x0, bbox_x1])
                    # ax.set_ylim([bbox_y1, bbox_y0])
                    style.show()
                assert False

            # Write out the data
            stat_df['FileName'].append(group)
            stat_df['Label'].append(label_id)
            stat_df['Day'].append(group_day)
            stat_df['Condition'].append(group_condition)
            stat_df['NumDay'].append(group_num_day)
            stat_df['MajorAxis'].append(group_major_axis)
            stat_df['MinorAxis'].append(group_minor_axis)
            stat_df['AspectRatio'].append(group_major_axis/group_minor_axis)

            stat_df['ROIArea'].append(mask_area)
            stat_df['ConvexArea'].append(convex_area)

            stat_df['ROIPerimeter'].append(group_perimeter)
            stat_df['ConvexPerimeter'].append(convex_perimeter)

            roi_circularity = 4 * np.pi * mask_area / group_perimeter**2
            convex_circularity = 4 * np.pi * convex_area / convex_perimeter**2

            stat_df['ROICircularity'].append(roi_circularity)
            stat_df['ConvexCircularity'].append(convex_circularity)

            stat_df['NumDefects'].append(num_defects)
            stat_df['DefectAreaUnfiltered'].append(defect_area_unfiltered)
            stat_df['DefectAreaFiltered'].append(defect_area_filtered)
            stat_df['PctDefectUnfiltered'].append(defect_area_unfiltered/convex_area)
            stat_df['PctDefectFiltered'].append(defect_area_filtered/convex_area)

        stat_df = pd.DataFrame(stat_df)
        print(stat_df.head())
        print(stat_df.describe())

        # Stash the intermediate results
        cachefile.parent.mkdir(parents=True, exist_ok=True)
        stat_df.to_hdf(cachefile, key='stats')
        all_stat_df.append(stat_df)

    all_stat_df = pd.concat(all_stat_df, ignore_index=True)
    all_stat_df.to_hdf(outfile, key='stats')
    return all_stat_df


def get_nearest_area(df: pd.DataFrame, area_map: Dict[int, float]) -> pd.DataFrame:
    """ Relabel the values in the dataframes based on nearest area

    :param DataFrame df:
        The data frame of stats to label
    :param dict[int, float] area_map:
        The map of ROI indices to areas
    :returns:
        A re-mapped data frame
    """

    if df.shape[0] != len(area_map):
        raise ValueError(f'Data and area seem misaligned: {df.shape[0]} vs {len(area_map)}')

    area_map = area_map.copy()

    new_labels = []
    for _, rec in df.iterrows():
        area = rec['Area']
        best_label = None
        best_diff = np.inf
        for label, a in area_map.items():
            diff = np.abs(a-area)
            if diff < best_diff:
                best_label = label
                best_diff = diff
        assert best_label is not None
        if best_diff > 100:
            new_labels.append(np.nan)
            continue
        del area_map[label]
        new_labels.append(label)
    assert len(new_labels) == df.shape[0]
    df['Label'] = new_labels
    df = df.dropna(how='any')
    return df.astype({'Label': 'int32'})


def categorize_knockdown(indir: pathlib.Path) -> Dict[str, List[str]]:
    """ Get the category for the knockdown

    :param Path indir:
        Input directory to categorize
    :returns:
        A dictionary mapping
    """
    possible_days = {
        'd10': ['d10', 'day10'],
        'd11': ['d11', 'day11'],
        'd12': ['d12', 'day12'],
        'd13': ['d13', 'day13'],
        'd14': ['d14', 'day14'],
        'd15': ['d15', 'day15'],
        'd1': ['d1', 'day1'],
        'd2': ['d2', 'day2'],
        'd3': ['d3', 'day3'],
        'd4': ['d4', 'day4'],
        'd5': ['d5', 'day5'],
        'd6': ['d6', 'day6'],
        'd7': ['d7', 'day7'],
        'd8': ['d8', 'day8'],
        'd9': ['d9', 'day9'],
    }
    possible_conds = {
        'tbxt-dox2': ['lbc-tbxt-dox2'],
        'tbxt-dox5': ['lbc-tbxt-dox5'],
        'wt': ['neeb057', 'lbc', 'lbc-tbxt-nodox'],
        'chordin': ['chordinkd', 'chrd'],
        'noggin': ['nogginkd', 'nog'],
    }
    inname = indir.name.lower().strip()
    out_day = None
    for key, possible_vals in possible_days.items():
        for val in possible_vals:
            if val in inname:
                assert out_day is None
                out_day = key
                break
        if out_day is not None:
            break
    out_cond = None
    for key, possible_vals in possible_conds.items():
        for val in possible_vals:
            if val in inname:
                assert out_cond is None
                out_cond = key
                break
        if out_cond is not None:
            break
    if out_day is None:
        raise ValueError(f'Missing day for {indir}')
    if out_cond is None:
        raise ValueError(f'Missing condition for {indir}')
    return {'Day': out_day, 'Condition': out_cond}


def categorize_chir_dosing(indir: pathlib.Path) -> Dict[str, List[str]]:
    """ Get the category for the CHIR dosing

    :param Path indir:
        The name of the input directory
    :returns:
        Category labels for this image based on its directory name
    """
    possible_days = {
        'd10': ['d10', 'day10'],
        'd11': ['d11', 'day11'],
        'd12': ['d12', 'day12'],
        'd13': ['d13', 'day13'],
        'd14': ['d14', 'day14'],
        'd15': ['d15', 'day15'],
        'd1': ['d1', 'day1'],
        'd2': ['d2', 'day2'],
        'd3': ['d3', 'day3'],
        'd4': ['d4', 'day4'],
        'd5': ['d5', 'day5'],
        'd6': ['d6', 'day6'],
        'd7': ['d7', 'day7'],
        'd8': ['d8', 'day8'],
        'd9': ['d9', 'day9'],
    }
    possible_lines = {
        'lbc': ['lbc'],
        'wtc': ['wtc'],
        'wtb': ['wtb'],
        'h1': ['h1'],
    }
    possible_conds = {
        '0um': ['0um'],
        '2um': ['2um', '2mm', '2m'],
        '4um': ['4um', '4mm', '4m'],
        '6um': ['6um', '6mm', '6m'],
    }
    inname = indir.name.lower().strip()
    out_day = None
    for key, possible_vals in possible_days.items():
        for val in possible_vals:
            if val in inname:
                assert out_day is None, f'{out_day} from {inname}'
                out_day = key
                break
        if out_day is not None:
            break

    out_line = None
    for key, possible_vals in possible_lines.items():
        for val in possible_vals:
            if val in inname:
                assert out_line is None, f'{out_line} from {inname}'
                out_line = key
                break
    if out_line is None:
        out_line = 'lbc'

    out_cond = None
    for key, possible_vals in possible_conds.items():
        for val in possible_vals:
            if val in inname:
                assert out_cond is None, f'{out_cond} from {inname}'
                out_cond = key
                break
    if out_day is None:
        raise ValueError(f'Missing day for {indir}')
    if out_cond is None:
        raise ValueError(f'Missing condition for {indir}')
    if out_line is None:
        raise ValueError(f'Missing cell line for {indir}')
    return {'Day': out_day, 'Condition': out_cond, 'CellLine': out_line}


def extract_image_stats(imagedir: pathlib.Path, has_spines: bool = True) -> Optional[pd.DataFrame]:
    """ Locate the files and extract all the stats for this image

    :param Path imagedir:
        The individual image directory to load
    :returns:
        A data frame with the stats for this image, or None if there was an error
    """
    if not imagedir.is_dir():
        return None

    print(f'Extracting stats for {imagedir}')
    contour_file = None
    spine_file = None
    stat_file = None
    for imagefile in imagedir.iterdir():
        if imagefile.name.endswith('_contours.xlsx'):
            assert contour_file is None
            contour_file = imagefile
        elif imagefile.name.endswith('_spines.xlsx'):
            assert spine_file is None
            spine_file = imagefile
        elif imagefile.name.endswith('_stats.csv'):
            assert stat_file is None
            stat_file = imagefile
    if contour_file is None or stat_file is None:
        print(f'Skipping empty image dir {imagedir}')
        return None
    if has_spines and spine_file is None:
        print(f'Skipping empty image dir {imagedir}')
        return None

    contour_df = pd.read_excel(contour_file)
    labels = set(np.unique(contour_df['Label']))

    if has_spines:
        spine_df = pd.read_excel(spine_file)
        assert labels == set(np.unique(spine_df['Label']))
    else:
        spine_df = None

    value_df = {
        'Label': [],
        'Area': [],
        'Perimeter': [],
        'Circularity': [],
        'SpineLength': [],
        'SpineWidth': [],
        'MajorAxis': [],
        'MinorAxis': [],
        'CurveDiameter': [],
        'AspectRatio': [],
    }

    for label in np.unique(contour_df['Label']):
        # Extract parameters from the contour
        roi_contour_df = contour_df[contour_df['Label'] == label]
        coords = np.stack([roi_contour_df['XCoord'], roi_contour_df['YCoord']], axis=1)

        area = poly.area_of_polygon(coords)
        perimeter = poly.perimeter_of_polygon(coords)
        circularity = (4 * np.pi * area)/(perimeter**2)
        major_axis = poly.major_axis_of_polygon(coords)
        minor_axis = poly.minor_axis_of_polygon(coords)

        # Extract parameters from the spine
        if has_spines:
            roi_spine_df = spine_df[np.logical_and(spine_df['Label'] == label,
                                    spine_df['IsLongest'])]
            coords = np.stack([roi_spine_df['XCoord'], roi_spine_df['YCoord']], axis=1)

            spine_length = poly.arc_length(coords)
            spine_width = np.max(roi_spine_df['DistToEdge'])

            curve_diameter = np.max([2*major_axis, spine_length])
            aspect_ratio = curve_diameter / spine_width
        else:
            spine_length = np.nan
            spine_width = np.nan
            curve_diameter = np.nan
            aspect_ratio = np.nan

        value_df['Label'].append(label)
        value_df['Area'].append(area)
        value_df['Perimeter'].append(perimeter)
        value_df['Circularity'].append(circularity)
        value_df['SpineLength'].append(spine_length)
        value_df['SpineWidth'].append(spine_width)
        value_df['MajorAxis'].append(major_axis)
        value_df['MinorAxis'].append(minor_axis)
        value_df['CurveDiameter'].append(curve_diameter)
        value_df['AspectRatio'].append(aspect_ratio)

    value_df = pd.DataFrame(value_df)
    value_df['SpaceScale'] = 4.02360515  # um/pixel
    value_df['FileName'] = imagedir.name
    return value_df


def histogram_equalize(img: np.ndarray) -> np.ndarray:
    """ Adaptive histogram equalization

    Remove local inhomogeneity in the image

    :param ndarray img:
        The 2D image, normalized between 0 and 1
    :returns:
        A histogram equalized image, normalized to between 0 and 1
    """
    # Bin the image as 8-bit
    img = img * 255
    img[img < 0] = 0
    img[img > 255] = 255
    img = img.astype(np.uint8)
    return equalize_adapthist(img)


def analyze_aggregate_image(imagefile: pathlib.Path,
                            outdir: pathlib.Path,
                            params: SegmentationParams,
                            save_individual_stats: bool = SAVE_INDIVIDUAL_STATS,
                            plot_individual_rois: bool = PLOT_INDIVIDUAL_ROIS,
                            simplify_graph: bool = False,
                            suffix: str = SUFFIX):
    """ Segment the aggregates from a single image

    :param Path imagefile:
        The image to segment
    :param Path outdir:
        The output directory to write stats and files to
    :param SegmentationParams params:
        The parameters for the segmentation algorithm
    :param bool save_individual_stats:
        If True, write individual stats to a file for each image
    :param bool plot_individual_rois:
        If True, plot the segmentation for each individual image
    :returns:
        The region data for this image
    """

    print(f'Segmenting {imagefile.name}')

    # Rescale and normalize the image
    img = load_image(imagefile, ctype='gray')/255.0
    rows, cols = img.shape[:2]
    aspect = float(cols) / float(rows)

    resize_x = min([params.resize_x, params.resize_y])
    resize_y = int(round(resize_x * aspect))

    space_scale = params.space_scale * rows / resize_x
    print(f'Got effective space scale: {space_scale} um/pixel')

    raw_img = img = resize(img, (resize_x, resize_y))
    img = histogram_equalize(img)

    # Smooth the image as needed
    if params.image_sigma > 0:
        img = gaussian(img, sigma=params.image_sigma)

    # Pull out the edge image on its own
    edge_img = np.abs(img - gaussian(img, sigma=10))
    edge_mask = edge_img > 0.2
    edge_mask = binary_dilation(edge_mask, selem=np.ones((3, 3)))
    edge_mask = remove_small_objects(edge_mask, 100)

    # Background subtract
    if params.background_sigma > 0:
        img = (img - gaussian(img, sigma=params.background_sigma))

    # Convert to all positive values, then sharpen
    img = np.abs(img)**params.image_norm

    # Canny filter to sharpen further
    img = img + canny(img)*0.5

    # Segment the image
    if params.background_sigma <= 0 and params.dark_aggregates:
        low_mask = img < params.threshold
        high_mask = img < params.threshold*2.0
    else:
        low_mask = img > params.threshold*2.0
        high_mask = img > params.threshold

    # Merge the edge and high segmentations
    high_mask = binary_dilation(high_mask, selem=np.ones((3, 3)))
    high_mask = remove_small_objects(high_mask, 100)
    high_mask = np.logical_or(edge_mask, high_mask)
    high_mask = remove_small_holes(high_mask, 100)
    high_mask = binary_erosion(high_mask, selem=np.ones((3, 3)))

    # Remove small dust and holes in the aggregates
    # 500px seems good for the evos, might change with other scopes
    low_mask = binary_dilation(low_mask)
    low_mask = remove_small_holes(low_mask, params.min_hole_size)
    low_mask = remove_small_objects(low_mask, params.min_object_size)
    low_mask = binary_erosion(low_mask)

    background_mask = binary_erosion(~high_mask, selem=np.ones((5, 5)))
    background_mask = remove_small_objects(background_mask, params.min_object_size)

    # Watershed trasform to break up touching aggregates
    distance = ndi.distance_transform_edt(low_mask)
    distance[distance < 5] = 0
    foreground_mask = distance > 0
    foreground_mask = binary_erosion(foreground_mask, selem=np.ones((5, 5)))
    foreground_mask = remove_small_objects(foreground_mask, params.min_object_size*0.5)

    labels = ndi.label(foreground_mask)[0]
    labels = labels + 1
    labels[~foreground_mask] = 0
    labels[background_mask] = 1

    labels = random_walker(img, labels, beta=params.segmentation_beta, multichannel=False)

    # Clean up the label mask by removing border labels and small objects
    stat_labels = np.zeros_like(labels)

    border = np.zeros(labels.shape, dtype=np.bool)
    if params.border_pixels > 0:
        border[:params.border_pixels, :] = 1
        border[-params.border_pixels:, :] = 1
        border[:, :params.border_pixels] = 1
        border[:, -params.border_pixels:] = 1

    label_ct = 0
    print(f'Got labels from {np.min(labels)} to {np.max(labels)}')
    for label_index in np.unique(labels):
        if label_index < 2:
            continue
        # Filter small segmentations
        # This seems good for evos images, might change for other scopes
        label_mask = labels == label_index
        if params.min_mask_size > 0 and np.sum(label_mask) < params.min_mask_size:
            continue
        if params.max_mask_size > 0 and np.sum(label_mask) > params.max_mask_size:
            continue
        if params.border_pixels > 0 and np.any(np.logical_and(label_mask, border)):
            continue

        # Final mask cleaning
        if params.min_mask_size > 0:
            label_mask = remove_small_holes(label_mask, params.min_mask_size//2)
            label_mask = remove_small_objects(label_mask, params.min_mask_size//2)
        label_ct += 1
        stat_labels[label_mask] = label_ct

    print(f'Got {np.unique(stat_labels).shape[0] - 1} regions after stat filtering')

    contours = {
        'Label': [],
        'XCoord': [],
        'YCoord': [],
    }
    spine_coords = []
    xx, yy = np.meshgrid(np.arange(resize_y), np.arange(resize_x))
    final_dist = ndi.distance_transform_edt(stat_labels > 0)
    for label in np.unique(stat_labels):
        if label < 1:
            continue
        final_mask = stat_labels == label
        if params.min_mask_size > 0 and np.sum(final_mask) < params.min_mask_size:
            continue
        if params.max_mask_size > 0 and np.sum(final_mask) > params.max_mask_size:
            continue

        contour = contours_from_mask(final_mask, 0.5)[0]
        contour = np.concatenate([contour, contour[0:1, :]], axis=0)

        contours['Label'].extend(label for _ in range(contour.shape[0]))
        contours['XCoord'].extend(contour[:, 0])
        contours['YCoord'].extend(contour[:, 1])

        if CALC_SPINES:

            axis_mask = skeletonize(stat_labels == label)
            axis_dist = final_dist[axis_mask]
            axis_coords = np.stack([xx[axis_mask], yy[axis_mask]], axis=1)

            tree = BallTree(axis_coords)
            links = tree.query_radius(axis_coords, 1.45, count_only=False, return_distance=False)

            node_table = set(range(axis_coords.shape[0]))
            link_table = {i: link for i, link in enumerate(links)}
            for i, link in enumerate(links):
                link = [j for j in link_table[i] if j != i]
                link_table[i] = link

            if simplify_graph:
                print(f'Got {len(node_table)} nodes and {len(link_table)} links before simplification')

                # Delete intermediate nodes
                for i in range(len(node_table)):
                    link = link_table[i]
                    if len(link) != 2:
                        continue

                    # A link with only a left and a right can be deleted
                    left, right = link
                    link_table[left] = [j for j in link_table[left] if j != i] + [right]
                    link_table[right] = [j for j in link_table[right] if j != i] + [left]
                    node_table.remove(i)
                    del link_table[i]
                print(f'Got {len(node_table)} nodes and {len(link_table)} links after simplification')

            g = nx.Graph()
            g.add_nodes_from(node_table)
            g.add_edges_from((i, j) for i, allj in link_table.items() for j in allj)

            ecc = nx.eccentricity(g)
            # center = nx.center(g, e=ecc)
            periphery = nx.periphery(g, e=ecc)

            # Root the tree at each different peripheral node:
            longest_path_indices = []
            for i in periphery:
                tree = nx.dfs_tree(g, i)
                longest_path_indices.append(nx.dag_longest_path(tree))

            # Now extract the real coordinates for those paths
            base_spine_coords = {
                'Label': [],
                'SpineID': [],
                'XCoord': [],
                'YCoord': [],
                'DistToEdge': [],
            }
            longest_idx = -1
            longest_dist = 0
            for idx, path_indices in enumerate(longest_path_indices):
                line = axis_coords[path_indices, :]
                if line.shape[0] < 2:
                    dist = 0
                else:
                    ds = np.sqrt(np.sum((line[:-1, :] - line[1:, :])**2, axis=1))
                    dist = np.sum(ds)
                if dist > longest_dist:
                    longest_dist = dist
                    longest_idx = idx
                base_spine_coords['Label'].extend(label for _ in range(line.shape[0]))
                base_spine_coords['SpineID'].extend(idx for _ in range(line.shape[0]))
                base_spine_coords['XCoord'].extend(line[:, 0])
                base_spine_coords['YCoord'].extend(line[:, 1])
                base_spine_coords['DistToEdge'].extend(axis_dist[path_indices])

            base_spine_coords = pd.DataFrame(base_spine_coords)
            base_spine_coords['IsLongest'] = base_spine_coords['SpineID'] == longest_idx

            spine_coords.append(base_spine_coords)

    img_outdir = outdir / imagefile.stem
    if img_outdir.is_dir():
        shutil.rmtree(img_outdir)
    img_outdir.mkdir(parents=True, exist_ok=True)

    contours = pd.DataFrame(contours)
    outfile = img_outdir / f'{imagefile.stem}_contours.xlsx'
    contours.to_excel(outfile)

    if len(spine_coords) > 0:
        spine_coords = pd.concat(spine_coords, ignore_index=True)
        outfile = img_outdir / f'{imagefile.stem}_spines.xlsx'
        spine_coords.to_excel(outfile)

    # Add all the stats to a final output table
    regiondata = []
    for regionprop in regionprops(stat_labels):
        if regionprop.label == 0:
            continue
        circularity = (4 * np.pi * regionprop.area)/(regionprop.perimeter**2)

        if CALC_SPINES:
            spines = spine_coords[spine_coords['Label'] == regionprop.label]

            line = np.stack([spines['XCoord'], spines['YCoord']], axis=1)
            spine_ds = np.sqrt(np.sum((line[:-1, :] - line[1:, :])**2, axis=1))
            spine_length = np.sum(spine_ds)
            max_dist_to_axis = np.max(spines['DistToEdge'])

            spine_displacement = np.sqrt(np.sum((line[-1, :] - line[0, :])**2))
        else:
            spine_length = max_dist_to_axis = spine_displacement = np.nan

        regiondata.append(RegionData(
            filename=imagefile.name,
            index=regionprop.label,
            major_axis=regionprop.major_axis_length,
            minor_axis=regionprop.minor_axis_length,
            area=regionprop.area,
            convex_area=regionprop.convex_area,
            perimeter=regionprop.perimeter,
            circularity=circularity,
            spine_length=spine_length,
            spine_displacement=spine_displacement,
            spine_dist_to_axis=max_dist_to_axis,
            space_scale=space_scale,
        ))

    # Calculate the stats for each final ROI and save to a spreadsheet
    if save_individual_stats and len(regiondata) > 0:
        outfile = img_outdir / f'{imagefile.stem}_stats.csv'
        with outfile.open('wt') as fp:
            fp.write(regiondata[0].csv_header + '\n')
            for region in regiondata:
                fp.write(region.csv_data + '\n')

    # Plot the ROI data and save to a file
    dist_mask = final_dist > 0
    dist_edge = np.logical_and(dist_mask, ~binary_erosion(dist_mask))
    dist_edge[0, :] = 1
    dist_edge[-1, :] = 1
    dist_edge[:, 0] = 1
    dist_edge[:, -1] = 1

    xx_edge = xx[dist_edge]
    yy_edge = yy[dist_edge]

    if plot_individual_rois:
        palette = colorwheel('Set1', n_colors=10)

        plotfile = img_outdir / f'{imagefile.stem}_roi_labels{suffix}'
        fig, ax = plt.subplots(1, 1, figsize=(8*aspect, 8))
        ax.imshow(raw_img, cmap='gray')
        for i, label in enumerate(np.unique(contours['Label'])):
            contour = contours[contours['Label'] == label]
            xcoords = contour['XCoord']
            ycoords = contour['YCoord']
            cx, cy = poly.center_of_polygon(np.stack([xcoords, ycoords], axis=1))

            ax.plot(xcoords, ycoords, '-', color=palette[i], linewidth=5)
            ax.text(cx, cy, f'ROI{label:02d}', color=palette[i], fontsize=24)

        if CALC_SPINES and len(spine_coords) > 0:
            for i, label in enumerate(np.unique(spine_coords['Label'])):
                spines = spine_coords[spine_coords['Label'] == label]
                spines = spines[spines['IsLongest']]
                xcoords = spines['XCoord'].values
                ycoords = spines['YCoord'].values

                coords = np.stack([xcoords, ycoords], axis=1)
                if coords.shape[0] == 0:
                    continue

                # Get the center coordinates of the spine
                cx, cy = poly.arc_coords_frac_along(coords, 0.5)

                rr_edge = (xx_edge - cx)**2 + (yy_edge - cy)**2
                theta_edge = np.arctan2(yy_edge - cy, xx_edge - cx)

                inds = np.argsort(rr_edge)
                theta_edge = theta_edge[inds]

                left_x = xx_edge[inds[0]]
                left_y = yy_edge[inds[0]]
                left_theta = theta_edge[0]

                # Phase unwrap, then find the shortest axis on the other side
                theta_delta = np.abs(theta_edge - left_theta)
                theta_unwrap = theta_delta > np.pi
                theta_delta[theta_unwrap] = 2*np.pi - theta_delta[theta_unwrap]
                theta_mask = theta_delta > np.pi/2
                inds = inds[theta_mask]

                # Now take the closest edge on the other side
                right_x = xx_edge[inds[0]]
                right_y = yy_edge[inds[0]]

                ax.plot(xcoords, ycoords, '-', color=palette[i], linewidth=5)
                ax.plot(cx, cy, 'o', color=palette[i], markersize=10)
                ax.plot([cx, left_x], [cy, left_y], '--', color=palette[i], linewidth=3)
                ax.plot([cx, right_x], [cy, right_y], '--', color=palette[i], linewidth=3)

        add_scalebar(ax, (resize_x, resize_y), space_scale=space_scale, bar_len=500)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'ROIs for {imagefile.stem}')
        fig.savefig(str(plotfile), transparent=True)
        plt.close()

        plotfile = img_outdir / f'{imagefile.stem}_plot{suffix}'
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16*aspect, 16))
        ax1.imshow(raw_img, cmap='gray')
        for i, label in enumerate(np.unique(contours['Label'])):
            contour = contours[contours['Label'] == label]
            xcoords = contour['XCoord']
            ycoords = contour['YCoord']
            ax1.plot(xcoords, ycoords, '-', color=palette[i], linewidth=5)
            cx, cy = poly.center_of_polygon(np.stack([xcoords, ycoords], axis=1))
            ax1.text(cx, cy, f'ROI{label:02d}', color=palette[i], fontsize=24)

        if CALC_SPINES and len(spine_coords) > 0:
            for i, label in enumerate(np.unique(spine_coords['Label'])):
                spines = spine_coords[spine_coords['Label'] == label]
                spines = spines[spines['IsLongest']]
                xcoords = spines['XCoord']
                ycoords = spines['YCoord']
                ax1.plot(xcoords, ycoords, '-', color=palette[i], linewidth=5)

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title('Raw Image')

        ax2.imshow(img)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title('After Background Subtraction')

        ax3.imshow(distance)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title('After Size Filter and Distance Transform')

        ax4.imshow(stat_labels, cmap='tab20b')
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_title('Final Watershed Segmentation')

        fig.savefig(str(plotfile), transparent=True)
        plt.close()
    return regiondata


def maybe_analyze_aggregate_image(item):
    """ Parallel processing to analyze images

    Analyze the image inside a worker process, allowing multiple images to be
    processed in parallel.
    """

    imagefile, outdir, params, save_individual_stats, plot_individual_rois = item
    try:
        return analyze_aggregate_image(imagefile, outdir,
                                       params=params,
                                       save_individual_stats=save_individual_stats,
                                       plot_individual_rois=plot_individual_rois)
    except Exception:
        traceback.print_exc()
        return None

# Main function


def analyze_aggregate_shape(rootdir: pathlib.Path,
                            outdir: Optional[pathlib.Path] = None,
                            processes: int = 1,
                            threshold: float = THRESHOLD,
                            border_pixels: int = BORDER_PIXELS,
                            image_sigma: float = IMAGE_SIGMA,
                            background_sigma: float = BACKGROUND_SIGMA,
                            min_hole_size: int = MIN_HOLE_SIZE,
                            min_object_size: int = MIN_OBJECT_SIZE,
                            min_mask_size: int = MIN_MASK_SIZE,
                            max_mask_size: int = MAX_MASK_SIZE,
                            resize_x: int = RESIZE_X,
                            resize_y: int = RESIZE_Y,
                            image_norm: float = IMAGE_NORM,
                            dark_aggregates: bool = DARK_AGGREGATES,
                            save_individual_stats: bool = SAVE_INDIVIDUAL_STATS,
                            plot_individual_rois: bool = PLOT_INDIVIDUAL_ROIS,
                            space_scale: float = SPACE_SCALE,
                            blacklist: Optional[List[str]] = None,
                            suffix: str = SUFFIX,
                            segmentation_beta: float = SEGMENTATION_BETA):
    """ Analyze a folder of aggregates

    :param Path rootdir:
        Path to the image directory
    :param Path outdir:
        Path to write the plots and data (default rootdir / 'stats')
    :param int processes:
        Number of parallel processes to use when analyzing a directory
    :param float threshold:
        Threshold to segment the aggregates (dark) from the background (bright)
    :param int border_pixels:
        Ignore masks that are this many pixels from the border
    :param float image_sigma:
        How much to smooth the image by before segmenting
    :param float background_sigma:
        How much to smooth the background by before subtracting
    :param int min_hole_size:
        Minimum size of holes in the mask
    :param int min_object_size:
        Minimum size of objects in the mask
    :param int min_mask_size:
        Minimum size of segmentation masks to keep
    :param bool save_individual_stats:
        If True, write out individual stats to a file for each image
    :param bool plot_individual_rois:
        If True, write out a plot of the ROI segmentation for each image
    :param list[str] blacklist:
        If not None, the list of file names to ignore for the images
    :param int resize_x:
        Resize the image to this many pixels in x
    :param int resize_y:
        Resize the image to this many pixels in y
    :param float image_norm:
        Norm for the image (either 1 - np.abs(x) or 2 - x\\*\\*2)
    :param bool dark_aggregates:
        If True, aggregates are darker against a light background
    :param bool save_individual_stats:
        If True, save the stats for each aggregate
    :param bool plot_individual_rois:
        If True, plot the individual ROIs for each aggregate
    """

    if not rootdir.is_dir():
        raise OSError(f'Cannot find image dir: {rootdir}')

    datapath = rootdir / 'Segmentations'
    datapath.mkdir(parents=True, exist_ok=True)
    # finalpath = rootdir / 'SegmentationsFinal'

    if blacklist is None:
        blacklist = []

    # Make the output folder to store the data
    if outdir is None:
        outdir = rootdir / 'stats'
    if outdir.is_dir():
        shutil.rmtree(str(outdir))
    outdir.mkdir(exist_ok=True, parents=True)

    params = SegmentationParams(
        threshold=threshold,
        border_pixels=border_pixels,
        image_sigma=image_sigma,
        background_sigma=background_sigma,
        min_hole_size=min_hole_size,
        min_object_size=min_object_size,
        min_mask_size=min_mask_size,
        max_mask_size=max_mask_size,
        resize_x=resize_x,
        resize_y=resize_y,
        image_norm=image_norm,
        dark_aggregates=dark_aggregates,
        space_scale=space_scale,
        segmentation_beta=segmentation_beta,
    )

    # Collect all the images into one list
    items = [(imagefile, outdir, params, save_individual_stats, plot_individual_rois)
             for imagefile in sorted(rootdir.iterdir())
             if imagefile.suffix in ('.tif', ) and imagefile.name not in blacklist]
    already_processed = [item[0] for item in items if (datapath / item[0].stem).is_dir()]
    items = [item for item in items if item[0] not in already_processed]
    for imagefile in already_processed:
        print(f'Skipping {imagefile}')

    # Process all the images in parallel and write their stats to one file
    with (outdir / (rootdir.name + '.csv')).open('wt') as fp:
        fmt = '{filename},{index},{major_axis},{minor_axis},{area},{convex_area},{perimeter},{circularity}\n'
        fp.write('Filename,ROI Index,Major Axis,Minor Axis,Area,Convex Area,Perimeter,Circularity\n')
        with Hypermap(processes=processes) as pool:
            for regiondata in pool.map(maybe_analyze_aggregate_image, items):
                if regiondata is None:
                    continue
                for region in regiondata:
                    fp.write(fmt.format(**region))
