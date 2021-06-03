#!/usr/bin/env python3

""" Analyze the actual contours underlaying the aggregates

Use the segmentations from ``analyze_aggregate_shape.py`` after they have been
merged and post-processed with ``merge_analyze_aggregate_shape.py``

.. code-block:: bash

    $ ./plot_analyze_aggregate_hands.py ~/data/TBXT-Segmentations

For a timeline analysis, see ``plot_analyze_aggregate_shape.py``

"""

# Imports
import sys
import shutil
import pathlib
import argparse

THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
if (BASEDIR / 'organoid_shape_tools').is_dir():
    sys.path.insert(0, str(BASEDIR))

# 3rd party
import pandas as pd

import matplotlib.pyplot as plt

# Our own imports
from organoid_shape_tools import load_convexity_contours
from organoid_shape_tools.plotting import set_plot_style, add_lineplot

# Constants

PLOT_STYLE = 'light'

# Functions


def call_defects(num_defects: int):
    """ Classify the convexity defects in the worm

    :param int num_defects:
        Number of detected convexity defect regions in the mask
    :returns:
        The category of the mask
    """
    if num_defects < 1:
        return 'sphere'
    elif num_defects < 2:
        return 'worm'
    else:
        return 'squid'


def plot_timeline(df: pd.DataFrame,
                  outdir: pathlib.Path,
                  suffixes: str = ('.svg', '.png'),
                  plot_style: str = PLOT_STYLE):
    """ Plot the timeline of aggregate wiggles """

    if outdir.is_dir():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Cheap calculated values
    df['DefectClass'] = df['NumDefects'].apply(call_defects)
    df['IsSquid'] = df['DefectClass'] == 'squid'
    df['IsWorm'] = df['DefectClass'] == 'worm'
    df['IsSphere'] = df['DefectClass'] == 'sphere'
    df['DefectPerimeter'] = df['ROIPerimeter'] - df['ConvexPerimeter']
    df['PctDefectPerimeter'] = df['DefectPerimeter'] / df['ROIPerimeter']

    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        add_lineplot(ax, df, x='NumDay', y='IsSquid', hue='Condition',
                     yscale=100)
        ax.legend()
        ax.set_xlabel('Day')
        ax.set_ylabel('% Squids')
        style.show(outfile=outdir / 'pct_squids.png', suffixes=suffixes)

    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        add_lineplot(ax, df, x='NumDay', y='IsWorm', hue='Condition',
                     yscale=100)
        ax.legend()
        ax.set_xlabel('Day')
        ax.set_ylabel('% Worms')
        style.show(outfile=outdir / 'pct_worms.png', suffixes=suffixes)

    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        add_lineplot(ax, df, x='NumDay', y='IsSphere', hue='Condition',
                     yscale=100)
        ax.legend()
        ax.set_xlabel('Day')
        ax.set_ylabel('% Sphere')
        style.show(outfile=outdir / 'pct_sphere.png', suffixes=suffixes)

    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        add_lineplot(ax, df, x='NumDay', y='NumDefects', hue='Condition')
        ax.legend()
        ax.set_xlabel('Day')
        ax.set_ylabel('Average number of defects')
        style.show(outfile=outdir / 'total_defects.png', suffixes=suffixes)

    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        add_lineplot(ax, df, x='NumDay', y='PctDefectUnfiltered', hue='Condition',
                     yscale=100)
        ax.legend()
        ax.set_xlabel('Day')
        ax.set_ylabel('% Convex Defect')
        style.show(outfile=outdir / 'pct_raw_defects.png', suffixes=suffixes)

    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        add_lineplot(ax, df, x='NumDay', y='PctDefectFiltered', hue='Condition',
                     yscale=100)
        ax.legend()
        ax.set_xlabel('Day')
        ax.set_ylabel('% Convex Defect Filtered')
        style.show(outfile=outdir / 'pct_filtered_defects.png', suffixes=suffixes)

    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        add_lineplot(ax, df, x='NumDay', y='PctDefectPerimeter', hue='Condition',
                     yscale=100)
        ax.legend()
        ax.set_xlabel('Day')
        ax.set_ylabel('% Defect Perimeter Anomaly')
        style.show(outfile=outdir / 'pct_defect_perimeter.png', suffixes=suffixes)

# Command-line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-style', default=PLOT_STYLE)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('rootdir', type=pathlib.Path)
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    datafile = args.rootdir / 'merge_knockdown_all.xlsx'
    data_df = pd.read_excel(datafile)

    print(data_df.head())

    contourfile = args.rootdir / 'merge_knockdown_all_hands.h5'
    if args.overwrite and contourfile.is_file():
        contourfile.unlink()
    stat_df = load_convexity_contours(args.rootdir, contourfile, data_df,
                                      overwrite=args.overwrite)
    print(stat_df.head())

    plot_timeline(stat_df, args.rootdir / 'plots', plot_style=args.plot_style)


if __name__ == '__main__':
    main()
