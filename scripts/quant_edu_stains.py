#!/usr/bin/env python3

""" Quantify the pH3 and EDU stains

Quantify pH3 and EDU stained sections

.. code-block:: bash

    $ ./quant_edu_stains.py ~/20201021_pH3

"""
# Imports
import pathlib
import shutil
import sys
import argparse

THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
if (BASEDIR / 'organoid_shape_tools').is_dir():
    sys.path.insert(0, str(BASEDIR))

# 3rd party
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from skimage.feature import peak_local_max
from skimage.measure import label, EllipseModel
from skimage.morphology import binary_dilation, remove_small_holes

# our own imports
from organoid_shape_tools.utils import contours_from_mask
from organoid_shape_tools.plotting import set_plot_style, colorwheel, get_histogram
from organoid_shape_tools import load_single_channel, group_infiles, plot_lines_fit

# Constants

PLOT_STYLE = 'light'

# Main function


def quant_edu_stains(rootdir: pathlib.Path,
                     threshold_dapi_abs: float = 0.05,
                     threshold_ph3_abs: float = 0.2,
                     bins: int = 15,
                     plot_style: str = PLOT_STYLE,
                     experiment: str = 'edu',
                     edge_size: int = 5,
                     max_extent_fit: float = 0.9):
    """ Quantify the EDU staining

    :param float max_extent_fit:
        Maximum extent to fit the lines/quads to avoid edge effects
    """

    outdir = rootdir / f'quant_{experiment}'
    if outdir.is_dir():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plotdir = rootdir / f'plots_{experiment}'
    if plotdir.is_dir():
        shutil.rmtree(plotdir)
    plotdir.mkdir(parents=True, exist_ok=True)

    gen = np.random.default_rng()

    if experiment == 'ph3':
        ph3_channel = '_c1'
        ylabel = '% PH3+ Cells'
    elif experiment == 'edu':
        ph3_channel = '_c3'
        ylabel = '% EDU+ Cells'
    else:
        raise KeyError(f'Unknown experiment type "{experiment}"')

    all_bin = []
    all_kern = []

    for dapi_file, ph3_file, instem, day in group_infiles(rootdir, ph3_channel=ph3_channel):

        df_bin = {
            'Filename': [],
            'ROI': [],
            'XBin': [],
            'DAPIBin': [],
            'PH3Bin': [],
            'PctBin': [],
            'Day': [],
        }
        df_kern = {
            'Filename': [],
            'ROI': [],
            'XKern': [],
            'DAPIKern': [],
            'PH3Kern': [],
            'PctKern': [],
            'Day': [],
        }
        print(instem)

        ph3_img = load_single_channel(ph3_file)
        dapi_img = load_single_channel(dapi_file)

        rows, cols = dapi_img.shape
        assert dapi_img.shape == ph3_img.shape

        dapi_img += gen.random((rows, cols))*1e-5

        mask = dapi_img > threshold_dapi_abs
        mask = binary_dilation(mask, selem=np.ones((5, 5)))
        mask = remove_small_holes(mask, 50000)

        # Split sections into major regions
        labels = label(mask)

        final_labels = np.zeros_like(labels)
        final_contours = []
        label_inds = np.unique(labels)
        for ind in label_inds:
            if ind == 0:
                continue
            mask = labels == ind
            if np.sum(mask) < 5000:
                continue

            # Work out how much of the aggregate touches the edge
            num_edge = np.sum(mask[:edge_size, :]) + np.sum(mask[-edge_size:, :])
            num_edge += np.sum(mask[:, :edge_size]) + np.sum(mask[:, -edge_size:])

            pct_edge = num_edge / np.sum(mask)

            if pct_edge > 0.01:
                continue

            contour = contours_from_mask(mask)[0]
            final_contours.append(contour)
            final_labels[mask] = ind

        # Calculate peaks only inside the major regions
        print(f'Got {len(final_contours)} slices')
        label_inds = np.unique(final_labels)
        label_peaks = []
        ph3_values = []
        dapi_values = []

        for ind in label_inds:
            if ind == 0:
                continue
            mask = final_labels == ind
            dapi_mask_img = dapi_img.copy()
            dapi_mask_img[~mask] = 0.0

            dapi_peaks = peak_local_max(dapi_mask_img, min_distance=5,
                                        threshold_abs=threshold_dapi_abs)

            ph3_value = ph3_img[dapi_peaks[:, 0], dapi_peaks[:, 1]]
            dapi_value = dapi_img[dapi_peaks[:, 0], dapi_peaks[:, 1]]

            ph3_mask = ph3_value > threshold_ph3_abs
            print(f'ROI {ind}: got {np.sum(ph3_mask)} PH3 cells')

            ph3_values.append(ph3_mask)
            dapi_values.append(dapi_value)
            label_peaks.append(dapi_peaks)

        # Back out the organoid "spine" using an ellipse model
        median_peak_lines = []
        median_convex_lines = []
        xx, yy = np.meshgrid(np.arange(final_labels.shape[0]),
                             np.arange(final_labels.shape[1]), indexing='ij')
        assert xx.shape == final_labels.shape
        assert yy.shape == final_labels.shape

        for i, ind in enumerate(label_inds):
            if ind == 0:
                continue
            dapi_peaks = label_peaks[i-1]
            contour = final_contours[i-1]

            model = EllipseModel()
            model.estimate(contour)
            xc, yc, a, b, theta = model.params
            if b > a:
                a, b = b, a
                theta += np.pi/2

            line = np.linspace(-a, a, 100)
            x = line*np.sin(theta) + yc
            y = line*np.cos(theta) + xc

            skel_coords = np.stack([x, y], axis=1)
            median_convex_lines.append(skel_coords)

            peaks_x = dapi_peaks[:, 0] - yc
            peaks_y = dapi_peaks[:, 1] - xc

            peaks_total = peaks_x * np.sin(theta) + peaks_y * np.cos(theta)
            peaks_total /= a

            median_peak_lines.append(peaks_total)

        plotfile = plotdir / f'{instem}_hists.png'

        with set_plot_style(plot_style) as style:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            for i, (peak_line, ph3_mask) in enumerate(zip(median_peak_lines, ph3_values)):

                if np.sum(ph3_mask) < 2:
                    continue

                # Larger number of cells always on the right
                if np.sum(peak_line < 0) > np.sum(peak_line > 0):
                    peak_line = peak_line * -1

                x_bin, y_bin, x_kern, y_kern = get_histogram(data=peak_line, bins=bins, range=(-1.0, 1.0))
                xp_bin, yp_bin, xp_kern, yp_kern = get_histogram(data=peak_line[ph3_mask], bins=bins, range=(-1.0, 1.0))

                pct_bin = yp_bin / (y_bin + 1e-5)
                pct_kern = yp_kern / (y_kern + 1e-5)
                x_bin = (x_bin[1:] + x_bin[:-1])*0.5

                df_bin['ROI'].extend(i for _ in x_bin)
                df_bin['Filename'].extend(instem for _ in x_bin)
                df_bin['XBin'].extend(x_bin)
                df_bin['DAPIBin'].extend(y_bin)
                df_bin['PH3Bin'].extend(yp_bin)
                df_bin['PctBin'].extend(pct_bin)
                df_bin['Day'].extend(day for _ in x_bin)

                df_kern['ROI'].extend(i for _ in x_kern)
                df_kern['Filename'].extend(instem for _ in x_kern)
                df_kern['XKern'].extend(x_kern)
                df_kern['DAPIKern'].extend(y_kern)
                df_kern['PH3Kern'].extend(yp_kern)
                df_kern['PctKern'].extend(pct_kern)
                df_kern['Day'].extend(day for _ in x_kern)

                ax1.plot(x_bin, pct_bin)
                ax2.plot(x_kern, pct_kern)
            style.show(outfile=plotfile)

        plotfile = plotdir / f'{instem}_seg.png'

        palette = colorwheel('tab20', n_colors=len(label_peaks))

        with set_plot_style(plot_style) as style:

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            ax1.imshow(ph3_img, cmap='inferno')
            ax2.imshow(dapi_img, cmap='gray')
            ax3.imshow(final_labels, cmap='tab20')

            ax4.imshow(dapi_img, cmap='gray')
            for i, contour in enumerate(final_contours):
                median_convex_line = median_convex_lines[i]
                ax4.plot(contour[:, 0], contour[:, 1], '-', color=palette[i])
                ax4.plot(median_convex_line[:, 1], median_convex_line[:, 0], '-', color=palette[i])

            for i, dapi_peaks in enumerate(label_peaks):
                ax2.plot(dapi_peaks[:, 1], dapi_peaks[:, 0], '.', color=palette[i])

            style.set_image_axis_lims(ph3_img, ax1)
            style.set_image_axis_lims(dapi_img, ax2)
            style.set_image_axis_lims(dapi_img, ax3)
            style.set_image_axis_lims(dapi_img, ax4)

            style.show(outfile=plotfile)

        outfile = outdir / f'{instem}_bin.xlsx'

        df_bin = pd.DataFrame(df_bin)
        df_bin.to_excel(outfile, index=False)

        outfile = outdir / f'{instem}_kern.xlsx'

        df_kern = pd.DataFrame(df_kern)
        df_kern.to_excel(outfile, index=False)

        all_bin.append(df_bin)
        all_kern.append(df_kern)

    outfile = outdir / f'{experiment}_all_bin.xlsx'
    all_bin = pd.concat(all_bin, ignore_index=True)
    all_bin.to_excel(outfile, index=False)

    outfile = outdir / f'{experiment}_all_kern.xlsx'
    all_kern = pd.concat(all_kern, ignore_index=True)
    all_kern.to_excel(outfile, index=False)

    plotfile = plotdir / f'{experiment}_all_bin.png'

    plot_lines_fit(data=all_bin, x='XBin', y='PctBin', plotfile=plotfile,
                   ylabel=ylabel, max_extent_fit=max_extent_fit,
                   plot_style=plot_style)

    plotfile = plotdir / f'{experiment}_all_kern.png'

    plot_lines_fit(data=all_kern, x='XKern', y='PctKern', plotfile=plotfile,
                   ylabel=ylabel, max_extent_fit=max_extent_fit,
                   plot_style=plot_style)

    plotfile = plotdir / f'{experiment}_day_bin.png'

    plot_lines_fit(data=all_bin, x='XBin', y='PctBin', hue='Day',
                   plotfile=plotfile, plot_style=plot_style,
                   ylabel=ylabel, max_extent_fit=max_extent_fit)

    plotfile = plotdir / f'{experiment}_day_kern.png'

    plot_lines_fit(data=all_kern, x='XKern', y='PctKern', hue='Day',
                   plotfile=plotfile, plot_style=plot_style,
                   ylabel=ylabel, max_extent_fit=max_extent_fit)

# Command line


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', choices=('ph3', 'edu'), default='ph3')
    parser.add_argument('rootdir', type=pathlib.Path)
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    quant_edu_stains(**vars(args))


if __name__ == '__main__':
    main()
