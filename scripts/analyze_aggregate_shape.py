#!/usr/bin/env python3

""" Threshold to segment aggregates from phase

Basic usage:

    $ python3 ./analyze_aggregate_shape.py /path/to/image/folder

For each ``.tif`` file in the folder, this will write out two files:

* ${filename}_plot.png - The segmentation plot
* ${filename}_stats.csv - The statistics for each segmentation

Segmentation that works well for the neural worms:

.. code-block:: bash

    $ ./analyze_aggregate_shape.py \\
        ~/Desktop/Segmentations/CHIR-Dosing/'untitled folder' \\
        --processes 1 \\
        --dark-aggregates \\
        --image-norm 4 \\
        --threshold 0.05 \\
        --border-pixels ' -1' \\
        --min-mask-size 2500

The threshold value will need to be tested between ``0.001`` and ``0.1`` to get
good segmentations in all conditions.

A segmentation that works well for the neural aggregates (0-2 uM CHIR):

.. code-block:: bash

    $ ./analyze_aggregate_shape.py \\
        ~/data/Segmentations/'LBC 0uM'/ \\
        --processes 1 \\
        --dark-aggregates \\
        --image-norm 1.5 \\
        --threshold 0.05 \\
        --border-pixels '-1' \\
        --min-mask-size 2000

To composite all the results into timeline plots, see ``merge_analyze_aggregate_shape.py``

To plot the merged results see ``plot_analyze_aggregate_shape.py``

For a shape analysis of the contours, see ``plot_analyze_aggregate_hands.py``

"""

# Standard lib
import sys
import pathlib
import argparse

THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
if (BASEDIR / 'organoid_shape_tools').is_dir():
    sys.path.insert(0, str(BASEDIR))

# Our own imports
from organoid_shape_tools import analyze_aggregate_shape
from organoid_shape_tools.utils import Hypermap

# Constants

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

# Command line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, default=Hypermap.cpu_count(),
                        help='Number of parallel processes to use')

    # Segmentation parameters
    parser.add_argument('--threshold', type=float, default=THRESHOLD,
                        help='Threshold to segment the dark aggregates from the light background')
    parser.add_argument('--border-pixels', type=int, default=BORDER_PIXELS,
                        help='Ignore masks that are this many pixels from the border')
    parser.add_argument('--image-sigma', type=float, default=IMAGE_SIGMA,
                        help='How much to smooth the image by before segmenting')
    parser.add_argument('--background-sigma', type=float, default=BACKGROUND_SIGMA,
                        help='How much to smooth the background by before subtracting')

    if DARK_AGGREGATES:
        parser.add_argument('--light-aggregates', action='store_false', dest='dark_aggregates',
                            help='If True, aggregates are assumed to be lighter than background')
    else:
        parser.add_argument('--dark-aggregates', action='store_true',
                            help='If True, aggregates are assumed to be darker than background')

    parser.add_argument('--min-hole-size', type=int, default=MIN_HOLE_SIZE,
                        help='Minimum size for holes in masks')
    parser.add_argument('--min-object-size', type=int, default=MIN_OBJECT_SIZE,
                        help='Minimum size for objects before mask separation')
    parser.add_argument('--min-mask-size', type=int, default=MIN_MASK_SIZE,
                        help='Minimum size for final masks')
    parser.add_argument('--max-mask-size', type=int, default=MAX_MASK_SIZE,
                        help='Maximum size for final masks')
    parser.add_argument('--resize-x', type=int, default=RESIZE_X,
                        help='How many pixels to resize the image to in the x direction')
    parser.add_argument('--resize-y', type=int, default=RESIZE_Y,
                        help='How many pixels to resize the image to in the y direction')
    parser.add_argument('--image-norm', type=float, default=IMAGE_NORM,
                        help='How to normalize the image to be all positive (1, 2, 4, 6, 8, etc)')

    parser.add_argument('--space-scale', default=SPACE_SCALE, type=float,
                        help='Microns per pixel')

    parser.add_argument('-b', '--blacklist', action='append', default=[],
                        help='Ignore images with this name')

    parser.add_argument('--suffix', default=SUFFIX,
                        help='Suffix to save plots with')

    parser.add_argument('--segmentation-beta', type=float, default=SEGMENTATION_BETA,
                        help='Segmentation diffusion constant')

    if SAVE_INDIVIDUAL_STATS:
        parser.add_argument('--skip-save-individual-stats',
                            dest='save_individual_stats',
                            action='store_false',
                            help="Don't save each image's stats to a separate file")
    else:
        parser.add_argument('--save-individual-stats',
                            action='store_true',
                            help="Save each image's stats to a separate file")

    if PLOT_INDIVIDUAL_ROIS:
        parser.add_argument('--skip-individual-rois',
                            dest='plot_individual_rois',
                            action='store_false',
                            help="Don't plot the individual segmentation images (faster)")
    else:
        parser.add_argument('--plot-individual-rois',
                            action='store_true',
                            help="plot the individual segmentation images (slower)")
    parser.add_argument('rootdirs', nargs='+', type=pathlib.Path,
                        help='Path(s) to the base directory to analyze')
    return parser.parse_args(args=args)


def main(args=None):
    args = vars(parse_args(args=args))
    for rootdir in args.pop('rootdirs'):
        analyze_aggregate_shape(rootdir, **args)


if __name__ == '__main__':
    main()
