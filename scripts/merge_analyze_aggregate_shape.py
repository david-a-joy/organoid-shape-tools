#!/usr/bin/env python3

""" Merge all the aggregate shape things

Take the individual plots from ``analyze_aggregate_shape.py`` and merge them into
a single large summary table.

.. code-block:: bash

    $ ./merge_analyze_aggregate_shape.py ~/data/Segmentations

This table can then be plotted using ``plot_analyze_aggregate_shape.py``

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
import pandas as pd

# Our own imports
from organoid_shape_tools import (
    categorize_chir_dosing, categorize_knockdown, extract_image_stats)

# Experiment functions


def merge_lbc_0vs4_data(rootdir: pathlib.Path):
    """ Merge the LBC 0 vs 4 uM CHIR comparison

    :param Path rootdir:
        The directory for the 0 vs 4 uM comparison
    """
    outfile = rootdir / 'merge_lbc_0vs4_all.xlsx'

    all_df = []

    for chirdir in sorted(rootdir.iterdir()):
        if not chirdir.is_dir():
            continue
        for imagedir in sorted(chirdir.iterdir()):
            value_df = extract_image_stats(imagedir)
            if value_df is None:
                continue
            for key, val in categorize_chir_dosing(imagedir).items():
                value_df[key] = val
            print(value_df.head())
            all_df.append(value_df)

    all_df = pd.concat(all_df, ignore_index=True)
    all_df.to_excel(outfile)


def merge_chir_dosing_data(rootdir: pathlib.Path):
    """ Merge all the CHIR dosing studies

    :param Path rootdir:
        The directory for the CHIR data
    """

    outfile = rootdir / 'merge_chir_dosing_all.xlsx'

    all_df = []

    for chirdir in sorted(rootdir.iterdir()):
        if not chirdir.is_dir():
            continue
        for imagedir in sorted(chirdir.iterdir()):
            value_df = extract_image_stats(imagedir)
            if value_df is None:
                continue
            for key, val in categorize_chir_dosing(imagedir).items():
                value_df[key] = val
            print(value_df.head())
            all_df.append(value_df)

    all_df = pd.concat(all_df, ignore_index=True)
    all_df.to_excel(outfile)


def merge_chir_dosing_data_new(rootdir: pathlib.Path):
    """ Merge the h1/h7/wtb/wtc CHIR dosing

    :param Path rootdir:
        The directory for the CHIR dosing data
    """
    if rootdir.name.endswith('-H1'):
        outfile = rootdir / 'merge_h1_chir_dosing_all.xlsx'
    elif rootdir.name.endswith('-H7'):
        outfile = rootdir / 'merge_h7_chir_dosing_all.xlsx'
    elif rootdir.name.endswith('-WTB'):
        outfile = rootdir / 'merge_wtb_chir_dosing_all.xlsx'
    elif rootdir.name.endswith('-WTC'):
        outfile = rootdir / 'merge_wtc_chir_dosing_all.xlsx'
    else:
        raise KeyError(f'Unknown CHIR dosing directory: {rootdir}')

    all_df = []

    for imagedir in sorted(rootdir.iterdir()):
        print(imagedir)
        if not imagedir.is_dir():
            continue
        value_df = extract_image_stats(imagedir, has_spines=False)
        if value_df is None:
            continue
        for key, val in categorize_chir_dosing(imagedir).items():
            value_df[key] = val
        print(value_df.head())
        all_df.append(value_df)

    all_df = pd.concat(all_df, ignore_index=True)
    all_df.to_excel(outfile)


def merge_tbxt_knockdown_data(rootdir: pathlib.Path):
    """ Merge the TBXT KD data

    :param Path rootdir:
        The directory for the TBXT KD data
    """
    outfile = rootdir / 'merge_knockdown_all.xlsx'

    all_df = []

    for daydir in sorted(rootdir.iterdir()):
        segdir = daydir / 'Segmentations'
        if not segdir.is_dir():
            continue
        for imagedir in sorted(segdir.iterdir()):
            value_df = extract_image_stats(imagedir, has_spines=False)
            if value_df is None:
                print(f'Skipping invalid directory: {imagedir}')
                continue
            for key, val in categorize_knockdown(imagedir).items():
                value_df[key] = val
            print(value_df.head())
            all_df.append(value_df)

    all_df = pd.concat(all_df, ignore_index=True)
    all_df['NumDay'] = all_df['Day'].map(lambda x: float(x[1:]))

    all_df = all_df.sort_values(by=['NumDay', 'Condition', 'FileName'])
    all_df.to_excel(outfile, index=False)


# Main function


def merge_analyze_aggregate_shape(rootdir: pathlib.Path):
    """ Merge all the analysis tables for everything

    :param Path rootdir:
        The directory to merge
    """
    rootname = rootdir.name.lower()
    if rootname == 'tbxt-segmentations':
        merge_tbxt_knockdown_data(rootdir)
    elif rootname == 'lbc_0vs4':
        merge_lbc_0vs4_data(rootdir)
    elif rootdir.name.lower() == 'chir-dosing':
        merge_chir_dosing_data(rootdir)
    elif rootdir.name.lower() == 'chir-dosing-new':
        merge_chir_dosing_data_new(rootdir)
    else:
        raise ValueError(f'Unknown experiment directory: {rootdir}')

# Command line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('rootdirs', nargs='+', type=pathlib.Path,
                        help='Path(s) to the base directory to analyze')
    return parser.parse_args(args=args)


def main(args=None):
    args = vars(parse_args(args=args))
    for rootdir in args.pop('rootdirs'):
        merge_analyze_aggregate_shape(rootdir, **args)


if __name__ == '__main__':
    main()
