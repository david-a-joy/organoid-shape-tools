#!/usr/bin/env python3

""" Analyze worm segmentations

Analyze the results:

    $ ./analyze_segment_worms.py /data/NeuroWorm-48hrs-2019-08-31

To run the segmentation pipeline, see ``segment_worms.py``

"""

# Standard lib
import argparse
import pathlib
import sys
import shutil
from typing import Optional, Dict

THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
if (BASEDIR / 'organoid_shape_tools').is_dir():
    sys.path.insert(0, str(BASEDIR))

# 3rd party
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

# Our own imports
from organoid_shape_tools.plotting import set_plot_style, colorwheel, add_lineplot

# Constants

PLOT_STYLE = 'poster'
PALETTE = 'wheel_bluegrey'

# Helper functions


def plot_lines(analyze_df: pd.DataFrame,
               outfile: pathlib.Path,
               key_tiles: Optional[Dict] = None,
               x: str = 'Timepoint',
               y: str = 'Area',
               hue: str = 'Class',
               xlabel: str = 'Time (hours)',
               ylabel: str = 'Area ($\\mu m^2$)',
               plot_style: str = PLOT_STYLE,
               palette: str = PALETTE):
    """ Plot lines on the overlay """
    if key_tiles is None:
        key_tiles = {}

    ratio = 1.21 / 1.75
    figsize_y = 8
    figsize_x = figsize_y * ratio

    palette = colorwheel(palette, n_colors=2)
    with set_plot_style('light') as style:
        fig, ax = plt.subplots(1, 1, figsize=(figsize_x, figsize_y))
        sns.lineplot(x=x, y=y, hue=hue,
                     data=analyze_df, ax=ax, palette=palette)
        for i, (tile_cls, tile_nums) in enumerate(key_tiles.items()):
            for tile_num in tile_nums:
                tile_df = analyze_df[analyze_df['Tile'] == tile_num]
                xvals = tile_df[x].values
                yvals = tile_df[y].values
                ax.plot(xvals, yvals, linestyle='--', linewidth=2, marker='', color=palette[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        style.show(outfile=outfile, transparent=True)

# Main function


def analyze_segment_worms(datadir: pathlib.Path,
                          outdir: Optional[pathlib.Path] = None,
                          overwrite: bool = False,
                          plot_style: str = PLOT_STYLE,
                          palette: str = PALETTE):
    """ Analyze the results of segmenting worms

    :param Path datadir:
        The directory to load
    """
    if outdir is None:
        outdir = datadir / 'WormAnalysis'
    if overwrite and outdir.is_dir():
        shutil.rmtree(str(outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    infile = datadir / 'roi_frames' / 'worm_summary_data.xlsx'
    if not infile.is_file():
        raise ValueError(f'Cannot find summary data {infile}')
    df = pd.read_excel(infile)

    # Do some temporal smoothing
    window = None
    # window = 11
    categories = [
        'Area',
        'Perimeter',
        'MinRadius',
        'MeanRadius',
        'MaxRadius',
        'RadiusRatio',
        'Circularity',
        'SemiMajorAxis',
        'SemiMinorAxis',
        'EllipseAxisRatio',
    ]
    if window is not None and window > 1:
        for tile in np.unique(df['Tile']):
            mask = df['Tile'] == tile
            tile_df = df.loc[mask, :]
            for cat in categories:
                val = tile_df[cat].rolling(
                    window=window,
                    min_periods=1,
                    center=True).median()
                df.loc[mask, cat] = val
    # Work out what kinds of worms we got
    # class_df = df[['Tile', 'RadiusRatio']].groupby('Tile', as_index=False).max()
    # class_df['2Class'] = 'NA'
    # class_df.loc[class_df['RadiusRatio'] < 2, '2Class'] = 'Non-Elongating'
    # class_df.loc[class_df['RadiusRatio'] >= 2, '2Class'] = 'Elongating'
    #
    # class_df['3Class'] = 'NA'
    # mask = np.logical_and(class_df['RadiusRatio'] >= 2,
    #                       class_df['RadiusRatio'] < 3)
    # class_df.loc[class_df['RadiusRatio'] < 2, '3Class'] = 'Non-Elongating'
    # class_df.loc[mask, '3Class'] = 'Partially Elongating'
    # class_df.loc[class_df['RadiusRatio'] >= 3, '3Class'] = 'Elongating'
    #
    # order_2class = ['Non-Elongating', 'Elongating']
    # order_3class = ['Non-Elongating', 'Partially Elongating', 'Elongating']

    class_df = df[['Tile', 'EllipseAxisRatio']].groupby('Tile', as_index=False).max()
    class_df['2Class'] = 'NA'
    class_df.loc[class_df['EllipseAxisRatio'] < 1.5, '2Class'] = 'Non-Elongating'
    class_df.loc[class_df['EllipseAxisRatio'] >= 1.5, '2Class'] = 'Elongating'

    class_df['3Class'] = 'NA'
    mask = np.logical_and(class_df['EllipseAxisRatio'] >= 1.5,
                          class_df['EllipseAxisRatio'] < 2.0)
    class_df.loc[class_df['EllipseAxisRatio'] < 1.5, '3Class'] = 'Non-Elongating'
    class_df.loc[mask, '3Class'] = 'Partially Elongating'
    class_df.loc[class_df['EllipseAxisRatio'] >= 2.0, '3Class'] = 'Elongating'

    order_2class = ['Non-Elongating', 'Elongating']
    order_3class = ['Non-Elongating', 'Partially Elongating', 'Elongating']

    print(class_df.head())

    palette = colorwheel(PALETTE, n_colors=2)

    outfile = outdir / 'counts_2class.svg'
    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=(1.5*2.5, 8))
        sns.countplot(data=class_df, x='2Class', ax=ax, palette=palette, edgecolor='k',
                      linewidth=2.5, order=order_2class)
        ax.set_xlabel('')
        ax.set_ylabel('Number of Organoids')
        style.rotate_xticklabels(ax, 45, verticalalignment='top')
        for suffix in ('.png', '.pdf', '.svg'):
            outfile = outfile.parent / f'{outfile.stem}{suffix}'
            style.show(outfile, transparent=True, close=False)
        plt.close()

    palette = colorwheel(PALETTE, n_colors=3)

    outfile = outdir / 'counts_3class.svg'
    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=(1.5*3, 8))
        sns.countplot(data=class_df, x='3Class', ax=ax, palette=palette, edgecolor='k',
                      linewidth=2.5, order=order_3class)
        ax.set_xlabel('')
        ax.set_ylabel('Number of Organoids')
        style.rotate_xticklabels(ax, 45, verticalalignment='top')
        for suffix in ('.png', '.pdf', '.svg'):
            outfile = outfile.parent / f'{outfile.stem}{suffix}'
            style.show(outfile, transparent=True, close=False)
        plt.close()

    num_worms = class_df.shape[0]

    pct_2class = {c: np.sum(class_df['2Class'] == c)/num_worms*100
                  for c in order_2class}
    pct_3class = {c: np.sum(class_df['3Class'] == c)/num_worms*100
                  for c in order_3class}

    palette = colorwheel(PALETTE, n_colors=2)
    last_part = 0
    outfile = outdir / 'percents_2class.svg'
    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=(1.5, 8))
        pbars = []
        for i, cls in enumerate(order_2class):
            pct = pct_2class.get(cls, 0)
            pbars.append(ax.bar(0, pct, bottom=last_part, color=palette[i], edgecolor='k',
                                linewidth=2.5))
            last_part += pct
        ax.legend(pbars, order_2class)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_ylabel('Percentage of Organoids')
        for suffix in ('.png', '.pdf', '.svg'):
            outfile = outfile.parent / f'{outfile.stem}{suffix}'
            style.show(outfile, transparent=True, close=False)
        plt.close()

    palette = colorwheel(PALETTE, n_colors=3)
    last_part = 0
    outfile = outdir / 'percents_3class.svg'
    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=(1.5, 8))
        pbars = []
        for i, cls in enumerate(order_3class):
            pct = pct_3class.get(cls, 0)
            pbars.append(ax.bar(0, pct, bottom=last_part, color=palette[i], edgecolor='k',
                                linewidth=2.5))
            last_part += pct
        ax.legend(pbars, order_3class)
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel('Percentage of Organoids')
        for suffix in ('.png', '.pdf', '.svg'):
            outfile = outfile.parent / f'{outfile.stem}{suffix}'
            style.show(outfile, transparent=True, close=False)
        plt.close()

    # Assign the labels
    df['2Class'] = 'NA'
    df['3Class'] = 'NA'
    for i, rec in class_df.iterrows():
        mask = df['Tile'] == rec['Tile']
        df.loc[mask, '2Class'] = rec['2Class']
        df.loc[mask, '3Class'] = rec['3Class']

    hue_orders = {
        None: None,
        '2Class': order_2class,
        '3Class': order_3class,
    }
    ylabels = {
        'Area': 'Area $(\\mu m^2)$',
        'Perimeter': 'Perimeter $(\\mu m)$',
        'MinRadius': 'Minimum Radius $(\\mu m)$',
        'MeanRadius': 'Average Radius $(\\mu m)$',
        'MaxRadius': 'Maximum Radius $(\\mu m)$',
        'RadiusRatio': 'Radius Ratio',
        'Circularity': 'Circularity',
        'SemiMajorAxis': 'Semi-Major Axis $(\\mu m)$',
        'SemiMinorAxis': 'Semi-Minor Axis $(\\mu m)$',
        'EllipseAxisRatio': 'Ellipse Axis Ratio',
    }

    df['TimepointDays'] = df['Timepoint'] / 60 / 24 + 3.0

    xlabel = 'Timepoint (days)'

    ratio = 1.21 / 1.75
    figsize_y = 8
    figsize_x = figsize_y * ratio

    for hue_cat, hue_order in hue_orders.items():
        out_subdir = outdir / (hue_cat.lower() if hue_cat else 'average')

        palette = colorwheel(PALETTE, n_colors=len(hue_order) if hue_order else 1)

        for cat in categories:
            print(f'Plotting {hue_cat} with {cat}')
            outfile = out_subdir / f'{cat.lower()}.svg'
            ylabel = ylabels[cat]
            with set_plot_style(plot_style) as style:
                fig, ax = plt.subplots(1, 1, figsize=(figsize_x, figsize_y))
                add_lineplot(x='TimepointDays', y=cat, hue=hue_cat,
                             hue_order=hue_order,
                             data=df, ax=ax, palette=palette)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                for suffix in ('.svg', '.png', '.pdf'):
                    outfile = outfile.parent / f'{outfile.stem}{suffix}'
                    style.show(outfile=outfile, transparent=True, close=False)
                plt.close()

# Command line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type=pathlib.Path,
                        help='Directory to write the files to')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite the cached data')
    parser.add_argument('datadir', type=pathlib.Path)
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    analyze_segment_worms(**vars(args))


if __name__ == '__main__':
    main()
