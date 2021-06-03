#!/usr/bin/env python3

""" Plot all the aggregate shape things

Use the segmentations from ``analyze_aggregate_shape.py`` after they have been
merged and post-processed with ``merge_analyze_aggregate_shape.py``

.. code-block:: bash

    $ ./plot_analyze_aggregate_shape.py ~/data/Segmentations-H1

For a shape analysis of the contours, see ``plot_analyze_aggregate_hands.py``

"""

# Imports
import sys
import pathlib
import argparse

THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
if (BASEDIR / 'organoid_shape_tools').is_dir():
    sys.path.insert(0, str(BASEDIR))

# 3rd party
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# Our own imports
from organoid_shape_tools.plotting import set_plot_style, add_lineplot, colorwheel

# Constants

PLOT_STYLE = 'light'

# Functions


def plot_timeline(df: pd.DataFrame,
                  outfile: pathlib.Path,
                  suffix: str = '.svg',
                  plot_style: str = PLOT_STYLE):
    """ Plot the timeline of aggregate shapes """

    ycolumn = 'OldAspectRatio'
    xcolumn = 'BinDay'

    outfile = outfile.parent / f'{outfile.stem}{suffix}'

    df['NumDay'] = df['Day'].map(lambda x: float(x[1:]))
    df['OldAspectRatio'] = df['MajorAxis'] / df['MinorAxis']

    worm_mask = np.logical_and(df[ycolumn] >= 1.0, df[ycolumn] < 20.0)

    bin_day = np.empty((df.shape[0], ))

    window = 0
    for num_day in np.arange(1, 11):
        mask = np.logical_and(df['NumDay'] >= num_day - window, df['NumDay'] <= num_day + window)
        bin_day[mask] = num_day

    df['BinDay'] = bin_day

    ylabel = {
        'AspectRatio': 'Aspect Ratio',
        'Circularity': 'Circularity',
        'OldAspectRatio': 'Aspect Ratio',
    }[ycolumn]
    ymin, ymax = {
        'AspectRatio': (0.0, 4.0),
        'OldAspectRatio': (1.0, 5.0),
        'Circularity': (0.0, 1.0),
    }[ycolumn]

    if outfile.name.startswith('knockdown'):
        hue_order = ['wt', 'chordin', 'noggin']
        palette = colorwheel('wheel_bluegrey', n_colors=3)
    elif outfile.name.startswith('chir_dose_wtb'):
        hue_order = ['2um', '4um', '6um']
        palette = colorwheel('wheel_bluegrey', n_colors=3)
        ymin, ymax = (1.0, 2.0)
    elif outfile.name.startswith('chir_dose_0vs4_lbc'):
        hue_order = ['0um', '4um']
        palette = colorwheel.from_colors(colors=[
            (155, 155, 155), (46+40, 52+40, 142+40)],
            color_type='8bit')
        ymin, ymax = (1.0, 5.0)
    elif outfile.name.startswith('chir_dose_2vs4vs6_h1'):
        hue_order = ['2um', '4um', '6um']
        palette = colorwheel('wheel_bluegrey', n_colors=3)
        ymin, ymax = (1.0, 5.0)
    elif outfile.name.startswith('chir_dose'):
        hue_order = ['2um', '4um', '6um']
        palette = colorwheel('wheel_bluegrey', n_colors=3)
        ymin, ymax = (1.0, 5.0)
    else:
        raise ValueError('Unknown order...')
    outfile.parent.mkdir(parents=True, exist_ok=True)

    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        add_lineplot(ax, df[worm_mask], x=xcolumn, y=ycolumn,
                     hue='Condition', hue_order=hue_order,
                     savefile=outfile, palette=palette,
                     drop_missing=True)
        plt.legend()
        ax.set_ylim([ymin, ymax])
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Day')
        for suffix in ('.svg', '.pdf', '.png'):
            style.show(outfile=(outfile.parent / f'{outfile.stem}{suffix}'), close=False)
        plt.close()


def plot_aggregate_classes(df: pd.DataFrame,
                           outfile: pathlib.Path,
                           suffix: str = '.svg',
                           plot_style: str = PLOT_STYLE):
    """ Aggregate class ratios for the data """

    df['NumDay'] = df['Day'].map(lambda x: float(x[1:]))
    df['OldAspectRatio'] = df['MajorAxis'] / df['MinorAxis']

    is_round = df['OldAspectRatio'] < 2.0
    is_partial = np.logical_and(df['OldAspectRatio'] >= 2.0, df['OldAspectRatio'] < 3.0)
    is_worm = df['OldAspectRatio'] >= 3.0

    agg_class = np.zeros(df.shape[0], dtype=str)
    agg_class[is_round] = 'Round'
    agg_class[is_partial] = 'Partial'
    agg_class[is_worm] = 'Elongating'

    df['AggregateType'] = agg_class
    palette = colorwheel.from_colors(colors=[
        (155, 155, 155), (46+40, 52+40, 142+40)], color_type='8bit')

    with set_plot_style(PLOT_STYLE) as style:

        day_df = df[df['NumDay'] == 7]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sns.countplot(hue='Condition', x='AggregateType', data=day_df, ax=ax, palette=palette)
        plt.legend()
        ax.set_xticklabels(['Round', 'Partial', 'Elongating'])
        for suffix in ('.svg', '.pdf', '.png'):
            style.show(outfile=(outfile.parent / f'{outfile.stem}{suffix}'), close=False)
        plt.close()

        for condition in np.unique(day_df['Condition']):
            cond_df = day_df[day_df['Condition'] == condition]
            cond_df = cond_df[['Condition', 'AggregateType']]
            count_df = cond_df.groupby(['AggregateType']).count()
            print(count_df)


# Command-line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-style', default=PLOT_STYLE)
    parser.add_argument('rootdir', type=pathlib.Path)
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    chir_dose_file = args.rootdir / 'merge_wtc_chir_dosing_all.xlsx'
    chir_dose_df = pd.read_excel(chir_dose_file)
    print(chir_dose_df.head())
    print(chir_dose_df.columns)

    plot_timeline(chir_dose_df, args.rootdir / 'plots' / 'chir_dose_2vs4vs6_wtc.svg',
                  plot_style=args.plot_style)


if __name__ == '__main__':
    main()
