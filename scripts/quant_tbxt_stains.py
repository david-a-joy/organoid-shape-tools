#!/usr/bin/env python3

""" Quantify the TBXT and SOX2 in images

Quantify TBXT and SOX2 stained sections

.. code-block:: bash

    $ ./quant_tbxt_stains.py ~/Emily_IF_07162020

"""
# Imports
import re
import pathlib
import shutil
import sys
import argparse
from typing import Optional, Dict

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

from scipy.stats import ttest_ind

# our own imports
from organoid_shape_tools.utils import (
    contours_from_mask, center_of_polygon, area_of_polygon)
from organoid_shape_tools.utils.czi_utils import CZIManager
from organoid_shape_tools.plotting import (
    set_plot_style, colorwheel, get_histogram, add_histogram, add_barplot)
from organoid_shape_tools.plotting.auto_replot_bars import replot_bars
from organoid_shape_tools import load_single_channel, group_infiles, plot_lines_fit

# Constants

PLOT_STYLE = 'figure'

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

# Main class


class TBXTQuant(object):
    """ Quantify TBXT slides """

    def __init__(self,
                 infile: pathlib.Path,
                 outdir: pathlib.Path,
                 plot_style: str = 'light',
                 stain_table: Optional[Dict[str, str]] = None):

        # Default stains
        if stain_table is None:
            stain_table = {
                'af350': 'dapi',
                'af555': 'tbxt',
                'af488': 'sox2',
            }
        self.stain_table = {k.lower(): v.lower() for k, v in stain_table.items()}

        if 'dapi' not in self.stain_table.values():
            raise ValueError(f'A DAPI channel is required for this analysis, got {self.stain_table}')

        self.infile = infile
        self.outdir = outdir

        self.plotfile = self.outdir / f'{infile.stem}_seg.png'
        self.outfile = self.outdir / f'{infile.stem}_stats.xlsx'

        self.plot_style = plot_style

        self._gen = np.random.default_rng()

        # Initialize areas and coordinates
        self._dapi_coords = None
        for stain_name in stain_table.values():
            setattr(self, f'_{stain_name}_img', None)
            setattr(self, f'_{stain_name}_values', None)

        self._labels = None
        self._contours = None
        self._contour_center_x = None
        self._contour_center_y = None
        self._contour_area = None
        self._contour_radius = None

        self.threshold_dapi_abs = 1000
        self.threshold_sox2_abs = 500
        self.threshold_tbxt_abs = 1000

        self.threshold_island_abs = 1000
        self.large_island_size = 200000
        self.small_island_size = 5000
        self.edge_size = 5

        self.threshold_dapi_per_image = {
            '20210217_LBC_4uM_D5_20x_488T_555SOX2_647b-cat_A.czi': 150,
            '20210217_LBC_4uM_D5_20x_488T_555SOX2_647b-cat_B.czi': 150,
            '20210217_LBC_4uM_D5_20x_488T_555SOX2_647b-cat_D.czi': 200,
            '20210217_LBC_4uM_D5_20x_488T_555SOX2_647b-cat_E.czi': 200,
            '20210220_NEEB193.2_LBC_4uM_D3_20x_488TBXT_555SOX2_Dapi_A.czi': 1000,
            '20210220_NEEB193.2_LBC_4uM_D3_20x_488TBXT_555SOX2_Dapi_B.czi': 1000,
            '20210220_NEEB193.2_LBC_4uM_D3_20x_488TBXT_555SOX2_Dapi_C.czi': 1000,
            '20210220_NEEB194_LBC_4uM_D3_10cm_20x_488TBXT_555SOX2_Dapi_A.czi': 600,
            '20210220_NEEB194_LBC_4uM_D3_10cm_20x_488TBXT_555SOX2_Dapi_B.czi': 600,
            '20210220_NEEB194_LBC_4uM_D3_10cm_20x_488TBXT_555SOX2_Dapi_C.czi': 600,
            '20210220_NEEB194_LBC_4uM_D3_10cm_20x_488TBXT_555SOX2_Dapi_D.czi': 600,
            '20210220_WTC_4uM_D3_20x_488TBXT_555SOX2_Dapi_A.czi': 1000,
            '20210220_WTC_4uM_D3_20x_488TBXT_555SOX2_Dapi_B.czi': 1000,
            '20210220_WTC_4uM_D3_20x_488TBXT_555SOX2_Dapi_C good.czi': 1000,
            '20210220_WTC_4uM_D3_20x_488TBXT_555SOX2_Dapi_D.czi': 1000,
            '20210217_LBC_0uM_D1_20x_488CDX2_555pSMAD_647e-cad_a.czi': 275,
            '20210217_LBC_0uM_D3_20x_488CDX2_555pSMAD_647e-cad_A bad.czi': 275,
            '20210217_LBC_0uM_D3_20x_488CDX2_555pSMAD_647e-cad_B.czi': 275,
            '20210217_LBC_0uM_D5_20x_488CDX2_555pSMAD_647e-cad_A.czi': 250,
            '20210217_LBC_4uM_D1_20x_488CDX2_555pSMAD_647e-cad_A.czi': 250,
            '20210217_LBC_4uM_D1_20x_488CDX2_555pSMAD_647e-cad_B.czi': 250,
            '20210217_LBC_4uM_D1_20x_488CDX2_555pSMAD_647e-cad_C.czi': 250,
            '20210217_LBC_4uM_D1_20x_488CDX2_555pSMAD_647e-cad_D.czi': 250,
            '20210217_LBC_4uM_D5_20x_488CDX2_555pSMAD_647e-cad_A.czi': 160,
            '20210217_LBC_4uM_D5_20x_488CDX2_555pSMAD_647e-cad_B.czi': 150,
            '20210217_LBC_4uM_D5_20x_488CDX2_555pSMAD_647e-cad_C.czi': 160,
            '20210220_NEEB193.2_LBC_4uM_D3_20x_488CDX2_594TBXT_Dapi_A.czi': 750,
            '20210220_NEEB193.2_LBC_4uM_D3_20x_488CDX2_594TBXT_Dapi_B.czi': 750,
            '20210220_NEEB193.2_LBC_4uM_D3_20x_488CDX2_594TBXT_Dapi_C.czi': 750,
            '20210220_WTC_4uM_D3_20x_488CDX2_594TBXT_Dapi_A.czi': 750,
            '20210220_WTC_4uM_D3_20x_488CDX2_594TBXT_Dapi_B.czi': 750,
            '20210220_WTC_4uM_D3_20x_488CDX2_594TBXT_Dapi_C.czi': 750,
        }
        self.threshold_dapi_per_image = {
            'TBXT-KD_5Dox_D3_slide2_555TBXT_DAPi_20200706_A.czi': 700,
            'TBXT-KD_5Dox_D3_slide2_555TBXT_DAPi_20200706_B.czi': 700,
            'TBXT-KD_5Dox_D3_slide2_555TBXT_DAPi_20200706_C.czi': 700,
            'TBXT-KD_NoDox_D3_slide_555TBXT_DAPi_20200706_C.czi': 1000,
            'TBXT-KD_NoDox_D3_slide_555TBXT_DAPi_20200706_D.czi': 1000,
            'TBXT-KD_NoDox_D3_slide_555TBXT_DAPi_20200706_E.czi': 1000,
            'TBXT-KD_NoDox_D7_slide_555TBXT_DAPi_20200706_B.czi': 400,
            'TBXT-KD_NoDox_D7_slide_555TBXT_DAPi_20200706_D.czi': 400,
            'TBXT-KD_NoDox_D7_slide_555TBXT_DAPi_20200706_E.czi': 400,
            '20210305_TBXT-KD_NoDOX_350Dapi_488SOX2_555TBXT_20x_E.czi': 1200,
            '20210305_TBXT-KD_NoDOX_350Dapi_488SOX2_555TBXT_20x_A.czi': 1200,
            '20210305_TBXT-KD_NoDOX_350Dapi_488SOX2_555TBXT_20x_J.czi': 1200,
        }

        self.min_peak_distance = 9

    def save_channel(self, channel_name: str, data: np.ndarray):
        """ Save the channel to an image """
        if channel_name not in self.stain_table:
            raise ValueError(f'No stain type for channel {channel_name}')
        stain_name = self.stain_table[channel_name]
        attr = f'_{stain_name}_img'
        print(f'Found {stain_name} for channel {channel_name}')
        setattr(self, attr, data.astype(np.float64))

    def load_images(self):
        with CZIManager(self.infile) as czi:
            for outfile, metadata, data in czi.iter_images():
                channel = metadata['channel']
                if channel == 'Alexa Fluor 488':
                    self.save_channel('af488', data)
                elif channel == 'Alexa Fluor 555':
                    self.save_channel('af555', data)
                elif channel in ('Hoechst 33258', 'Alexa Fluor 350', 'Alexa Fluor 405'):
                    self.save_channel('af350', data)
                elif channel == 'Alexa Fluor 647':
                    self.save_channel('af647', data)
                else:
                    raise KeyError(f'Unknown image: {metadata} at {outfile}')

        for channel_name, stain_name in self.stain_table.items():
            if getattr(self, f'_{stain_name}_img', None) is None:
                raise ValueError(f'Missing {stain_name.upper()} in channel {channel_name.upper()} in {self.infile}')

        rows, cols = self._dapi_img.shape
        # assert (rows, cols) == self._sox2_img.shape
        assert (rows, cols) == self._tbxt_img.shape

        self.rows = rows
        self.cols = cols

    def segment_dapi_islands(self):
        dapi_img = self._dapi_img
        rows, cols = dapi_img.shape

        dapi_percentiles = np.percentile(dapi_img, [5, 25, 50, 75, 95])

        print(f'DAPI Min: {np.min(dapi_img)} Max: {np.max(dapi_img)}')
        print(f'DAPI Percentiles: {dapi_percentiles}')

        threshold_island_abs = self.threshold_dapi_per_image.get(self.infile.name, self.threshold_island_abs)
        print(f'DAPI Island threshold: {threshold_island_abs}')

        dapi_img += self._gen.random((rows, cols))

        mask = dapi_img > threshold_island_abs
        mask = binary_dilation(mask, selem=np.ones((10, 10)))
        mask = remove_small_holes(mask, self.large_island_size)
        edge_size = self.edge_size

        if not np.any(mask):
            raise ValueError(f'Failed to segment DAPI Islands: {self.infile}')

        # Split sections into major regions
        labels = label(mask)

        final_labels = np.zeros_like(labels)
        final_contours = []
        final_center_x = []
        final_center_y = []
        final_radius = []
        final_area = []
        label_inds = np.unique(labels)
        for ind in label_inds:
            if ind == 0:
                continue
            mask = labels == ind
            if np.sum(mask) < self.small_island_size:
                continue

            # Work out how much of the aggregate touches the edge
            num_edge = np.sum(mask[:edge_size, :]) + np.sum(mask[-edge_size:, :])
            num_edge += np.sum(mask[:, :edge_size]) + np.sum(mask[:, -edge_size:])

            pct_edge = num_edge / np.sum(mask)

            if pct_edge > 0.01:
                continue

            contour = contours_from_mask(mask)[0]

            cx, cy = center_of_polygon(contour)
            print(contour.shape)

            contour_radii = np.sqrt((contour[:, 0] - cx)**2 + (contour[:, 1] - cy)**2)
            contour_area = area_of_polygon(contour)

            final_center_x.append(cx)
            final_center_y.append(cy)
            final_radius.append(np.mean(contour_radii))
            final_area.append(contour_area)
            final_contours.append(contour)
            final_labels[mask] = ind

        if len(final_contours) < 1:
            raise ValueError(f'Failed to calculate DAPI island contours: {self.infile}')

        print(f'Got {len(final_contours)} sections')
        self._labels = final_labels
        self._contours = final_contours

        self._contour_center_x = final_center_x
        self._contour_center_y = final_center_y
        self._contour_area = final_area
        self._contour_radius = final_radius

    def segment_cells(self):
        """ Use the individual islands to mark the boundaries of each section """

        # Calculate peaks only inside the major regions
        label_inds = np.unique(self._labels)

        dapi_img = self._dapi_img
        rows, cols = dapi_img.shape

        label_inds = np.unique(self._labels)
        if set(label_inds) == {0} or len(label_inds) < 1:
            raise ValueError(f'No valid DAPI island labels for {self.infile}')

        for stain_name in self.stain_table.values():
            stain_img = getattr(self, f'_{stain_name}_img')
            print(f'{stain_name.upper()} Min: {np.min(stain_img)} Max: {np.max(stain_img)}')
            print(f'{stain_name.upper()} Percentiles: {np.percentile(stain_img, [5, 25, 50, 75, 95])}')

        dapi_percentiles = np.percentile(dapi_img, [5, 25, 50, 75, 95])
        print(f'DAPI percentiles: {dapi_percentiles}')
        threshold_dapi_abs = self.threshold_dapi_per_image.get(self.infile.name, self.threshold_dapi_abs)
        print(f'DAPI Cell threshold: {threshold_dapi_abs}')

        dapi_coords = []
        dapi_values = []
        stain_values = {}
        for stain_name in self.stain_table.values():
            stain_values[stain_name] = []

        for ind in label_inds:
            if ind == 0:
                continue
            mask = self._labels == ind
            dapi_mask_img = dapi_img.copy()
            dapi_mask_img[~mask] = 0.0

            dapi_peaks = peak_local_max(dapi_mask_img,
                                        min_distance=self.min_peak_distance,
                                        threshold_abs=threshold_dapi_abs)

            dapi_value = self._dapi_img[dapi_peaks[:, 0], dapi_peaks[:, 1]]

            dapi_coords.append(dapi_peaks)
            dapi_values.append(dapi_value)

            for stain_name in self.stain_table.values():
                stain_img = getattr(self, f'_{stain_name}_img')
                stain_value = stain_img[dapi_peaks[:, 0], dapi_peaks[:, 1]]

                threshold_abs = getattr(self, f'threshold_{stain_name}_abs')

                stain_mask = stain_value > threshold_abs
                stain_total = np.sum(stain_mask)
                cell_total = stain_mask.shape[0]
                stain_pct = stain_total/cell_total
                print(f'{self.infile.name}: ROI {ind}: got {stain_total}/{cell_total} {stain_name.upper()}+ cells ({stain_pct:0.2%})')

                stain_values[stain_name].append(stain_value)

        self._dapi_coords = dapi_coords
        self._dapi_values = dapi_values
        for stain_name, stain_value in stain_values.items():
            setattr(self, f'_{stain_name}_values', stain_value)

    def plot_images(self):
        """ Plot the image for the segmentation """

        vmin, vmax = np.percentile(self._tbxt_img, [2, 98])

        with set_plot_style(self.plot_style) as style:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(self._tbxt_img, cmap='gray', vmin=vmin, vmax=vmax)
            for contour in self._contours:
                ax.plot(contour[:, 0], contour[:, 1], '-')
            for i, dapi_coord in enumerate(self._dapi_coords):
                tbxt_mask = self._tbxt_values[i] > self.threshold_tbxt_abs
                ax.plot(dapi_coord[tbxt_mask, 1], dapi_coord[tbxt_mask, 0], 'o', color='red')
                ax.plot(dapi_coord[~tbxt_mask, 1], dapi_coord[~tbxt_mask, 0], '.', color='gray')
            style.set_image_axis_lims(self._dapi_img)
            ax.set_title(f'TBXT Segmentation: {self.infile.stem}')
            style.show(outfile=self.plotfile)

    def plot_distributions(self):

        for i in range(len(self._dapi_values)):

            with set_plot_style(self.plot_style) as style:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
                add_histogram(ax1, self._dapi_values[i])
                add_histogram(ax2, self._sox2_values[i])
                add_histogram(ax3, self._tbxt_values[i])

                ax1.axvline(self.threshold_dapi_abs)
                ax2.axvline(self.threshold_sox2_abs)
                ax3.axvline(self.threshold_tbxt_abs)

                fig.suptitle(f'{self.infile.name} ROI{i+1}')
                ax1.set_title('DAPI')
                ax2.set_title('SOX2')
                ax3.set_title('TBXT')
                style.show()

    def save_stats(self):
        """ Calculate the stats for each nuclei """

        assert len(self._contours) == len(self._dapi_values)

        df = {
            'Day': [],
            'Condition': [],
            'ROI': [],
            'IslandArea': [],
            'IslandRadius': [],
            'IslandCenterX': [],
            'IslandCenterY': [],
            'CellX': [],
            'CellY': [],
            'CellDAPI': [],
            'CellTBXT': [],
        }
        filename = self.infile.name.upper()

        if filename.startswith('TBXT-KD_5DOX'):
            condition = 'TBXT-KD'
        elif filename.startswith('TBXT-KD_NODOX'):
            condition = 'WT'
        elif '_NODOX_' in filename:
            condition = 'WT'
        elif '_5DOX_' in filename:
            condition = 'TBXT-KD'
        else:
            raise ValueError(f'Unknown condition in file {self.infile}')

        if '_D3_' in filename:
            day = 3
        elif '_D5_' in filename:
            day = 5
        elif '_D7_' in filename:
            day = 7
        elif '_D10_' in filename:
            day = 10
        elif filename.startswith('20210305_TBXT-KD_NODOX_'):
            day = 5
        else:
            raise ValueError(f'Unknown day in file {self.infile}')

        # if '_D1_20X_' in self.infile.name.upper():
        #     day = 1
        # elif '_D3_20X_' in self.infile.name.upper() or '_D3_10CM' in self.infile.name.upper():
        #     day = 3
        # elif '_D5_20X_' in self.infile.name.upper():
        #     day = 5
        # else:
        #     raise ValueError(f'Unknown day in file {self.infile}')
        #
        # if '_LBC_0UM_' in self.infile.name.upper():
        #     condition = '0uM'
        # elif '_LBC_4UM_' in self.infile.name.upper() or '_WTC_4UM_' in self.infile.name.upper():
        #     condition = '4uM'
        # else:
        #     raise ValueError(f'Unknown condition in file {self.infile}')

        for i in range(len(self._contours)):
            df['ROI'].extend(i+1 for _ in self._dapi_values[i])
            df['Day'].extend(day for _ in self._dapi_values[i])
            df['Condition'].extend(condition for _ in self._dapi_values[i])
            df['IslandArea'].extend(self._contour_area[i] for _ in self._dapi_values[i])
            df['IslandRadius'].extend(self._contour_radius[i] for _ in self._dapi_values[i])
            df['IslandCenterX'].extend(self._contour_center_x[i] for _ in self._dapi_values[i])
            df['IslandCenterY'].extend(self._contour_center_y[i] for _ in self._dapi_values[i])

            df['CellX'].extend(cx for cx, _ in self._dapi_coords[i])
            df['CellY'].extend(cy for _, cy in self._dapi_coords[i])
            df['CellDAPI'].extend(self._dapi_values[i])
            df['CellTBXT'].extend(self._tbxt_values[i])
            # df['CellSOX2'].extend(self._sox2_values[i])

        df = pd.DataFrame(df)
        df.to_excel(self.outfile, index=False)

# Main Function


def quant_tbxt_stains(rootdir: pathlib.Path):
    """ Quantify the TBXT staining in sections

    :param Path rootdir:
        Directory to quantify
    """
    outdir = rootdir / 'plots_tbxt'
    if outdir.is_dir():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if rootdir.name == 'CZI':
        pattern = re.compile(r'^.*_488T_555SOX2_', re.IGNORECASE)
    elif rootdir.name == '4uM D3':
        pattern = re.compile(r'^.*_488TBXT_555SOX2_', re.IGNORECASE)
    else:
        raise OSError(f'Unknown base Directory: {rootdir}')

    for infile in rootdir.iterdir():
        if infile.suffix != '.czi':
            continue
        if not pattern.match(infile.name):
            continue
        if not infile.is_file():
            continue
        print(infile)
        proc = TBXTQuant(infile, outdir)
        proc.load_images()
        proc.segment_dapi_islands()
        proc.segment_cells()
        # proc.plot_distributions()
        proc.plot_images()
        proc.save_stats()


def quant_cdx2_stains(rootdir: pathlib.Path):
    """ Quantify the CDX2 staining in sections

    :param Path rootdir:
        Directory to quantify
    """
    outdir = rootdir / 'plots_cdx2'
    if outdir.is_dir():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if rootdir.name == 'CZI':
        pattern = re.compile(r'^.*_488CDX2_555pSMAD_647e-cad_', re.IGNORECASE)
    elif rootdir.name == '4uM D3':
        pattern = re.compile(r'^.*_488CDX2_594TBXT_Dapi_', re.IGNORECASE)
    else:
        raise OSError(f'Unknown base Directory: {rootdir}')

    for infile in rootdir.iterdir():
        if infile.suffix != '.czi':
            continue
        if not pattern.match(infile.name):
            continue
        if not infile.is_file():
            continue
        print(infile)
        proc = TBXTQuant(infile, outdir)
        proc.load_images()
        proc.segment_dapi_islands()
        proc.segment_cells()
        # proc.plot_distributions()
        proc.plot_images()
        proc.save_stats()


def analyze_quant_tbxt_stains(rootdir: pathlib.Path,
                              plot_style: str = PLOT_STYLE):
    """ Analyze the TBXT quantification """

    indirs = [rootdir / 'plots_tbxt', rootdir.parent / '4uM D3' / 'plots_tbxt']

    outdir = rootdir / 'quant_tbxt'
    if outdir.is_dir():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_df = []
    for indir in indirs:
        for infile in indir.iterdir():
            if infile.name.startswith('.') or not infile.name.endswith('.xlsx'):
                continue
            prefix = infile.stem[:-len('_stats')]
            prefix = prefix.rsplit('_', 1)[0]
            print(prefix)

            df = pd.read_excel(infile)
            df['Prefix'] = prefix
            df['Filename'] = infile.stem
            all_df.append(df)
    all_df = pd.concat(all_df, ignore_index=True)

    mean_df = all_df[['Prefix', 'CellTBXT']].groupby('Prefix', as_index=False).mean()

    for i, rec in mean_df.iterrows():
        prefix = rec['Prefix']
        prefix_mean = rec['CellTBXT']

        print(f'{prefix}: {prefix_mean}')

        sub_df = all_df[all_df['Prefix'] == prefix]

        with set_plot_style('light') as style:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            add_histogram(ax, sub_df['CellTBXT'])
            ax.axvline(prefix_mean, color='r')

            ax.set_title(prefix)
            style.show(outfile=outdir / f'replicates{i:02d}.png')

    thresholds = {
        '20210217_LBC_0uM_D1_20x_488T_555SOX2_647b-cat': 2500,
        '20210217_LBC_0uM_D3_20x_488T_555SOX2_647b-cat': 2500,
        '20210217_LBC_0uM_D5_20x_488T_555SOX2_647b-cat': 2500,
        '20210217_LBC_4uM_D1_20x_488T_555SOX2_647b-cat': 1000,
        '20210217_LBC_4uM_D5_20x_488T_555SOX2_647b-cat': 2500,
        '20210220_NEEB193.2_LBC_4uM_D3_20x_488TBXT_555SOX2_Dapi': 1000,
        # '20210220_NEEB194_LBC_4uM_D3_10cm_20x_488TBXT_555SOX2_Dapi': 1000,
        # '20210220_WTC_4uM_D3_20x_488TBXT_555SOX2_Dapi': 1000,
    }
    pct_df = {
        'Condition': [],
        'Day': [],
        'PctTBXT': [],
        'Filename': [],
    }

    for prefix, threshold in thresholds.items():
        prefix_df = all_df[all_df['Prefix'] == prefix]

        for filename in np.unique(prefix_df['Filename']):
            section_df = prefix_df[prefix_df['Filename'] == filename]
            total_cells = section_df.shape[0]
            positive_cells = np.sum(section_df['CellTBXT'] >= threshold)

            pct_tbxt = positive_cells / total_cells
            pct_df['Condition'].append(section_df['Condition'].values[0])
            pct_df['Day'].append(section_df['Day'].values[0])
            pct_df['PctTBXT'].append(pct_tbxt*100)
            pct_df['Filename'].append(section_df['Filename'].values[0])
    pct_df = pd.DataFrame(pct_df)

    print(pct_df)
    significance = {}
    for day in np.unique(pct_df['Day']):
        day_df = pct_df[pct_df['Day'] == day]
        left = day_df[day_df['Condition'] == '0uM']
        right = day_df[day_df['Condition'] == '4uM']

        _, pval = ttest_ind(left['PctTBXT'].values, right['PctTBXT'].values)
        if pval < 0.05:
            significance[((day, '0uM'), (day, '4uM'))] = pval
    print(significance)

    outfile = outdir / 'tbxt_stats.png'
    savefile = outdir / 'tbxt_stats.xlsx'

    with set_plot_style('light') as style:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        add_barplot(ax, pct_df, x='Day', hue='Condition', y='PctTBXT',
                    significance=significance, savefile=savefile)
        ax.set_ylabel('% TBXT+ Cells')
        style.show(outfile=outfile)

    replot_bars(savefile,
                outfile=outdir / 'tbxt_stats_final.pdf',
                suffixes=('.png', '.pdf', '.svg'),
                xcolumn='Day',
                ycolumn='PctTBXT',
                hue_column='Condition',
                xlabel='Day',
                ylabel='% TBXT+ Cells',
                ylimits=[(0.0, 75.0)],
                palette='wheel_bluegrey')


def analyze_quant_cdx2_stains(rootdir: pathlib.Path,
                              plot_style: str = PLOT_STYLE):
    """ Analyze the CDX2 quantification """

    indirs = [rootdir / 'plots_cdx2', rootdir.parent / '4uM D3' / 'plots_cdx2']

    outdir = rootdir / 'quant_cdx2'
    if outdir.is_dir():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_df = []
    for indir in indirs:
        for infile in indir.iterdir():
            if infile.name.startswith('.') or not infile.name.endswith('.xlsx'):
                continue
            prefix = infile.stem[:-len('_stats')]
            prefix = prefix.rsplit('_', 1)[0]
            print(prefix)

            df = pd.read_excel(infile)
            df['Prefix'] = prefix
            df['Filename'] = infile.stem
            all_df.append(df)
    all_df = pd.concat(all_df, ignore_index=True)
    all_df = all_df.rename({'CellTBXT': 'CellCDX2'}, axis=1)

    mean_df = all_df[['Prefix', 'CellCDX2']].groupby('Prefix', as_index=False).mean()

    for i, rec in mean_df.iterrows():
        prefix = rec['Prefix']
        prefix_mean = rec['CellCDX2']

        print(f'{prefix}: {prefix_mean}')

        sub_df = all_df[all_df['Prefix'] == prefix]

        with set_plot_style('light') as style:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            add_histogram(ax, sub_df['CellCDX2'])
            ax.axvline(prefix_mean, color='r')

            ax.set_title(prefix)
            style.show(outfile=outdir / f'replicates{i:02d}.png')

    # [800]
    # [1500, 1500]
    # [800]
    # [1500, 1500, 500, 500]
    # [500, 500, 500]
    # [4000, 4000]

    thresholds = {
        '20210217_LBC_0uM_D1_20x_488CDX2_555pSMAD_647e-cad': 1000,
        '20210217_LBC_0uM_D3_20x_488CDX2_555pSMAD_647e-cad': 2000,
        '20210217_LBC_0uM_D5_20x_488CDX2_555pSMAD_647e-cad': 1000,
        '20210217_LBC_4uM_D1_20x_488CDX2_555pSMAD_647e-cad': 750,
        '20210217_LBC_4uM_D5_20x_488CDX2_555pSMAD_647e-cad': 750,
        '20210220_NEEB193.2_LBC_4uM_D3_20x_488CDX2_594TBXT_Dapi': 2000,
    }
    pct_df = {
        'Condition': [],
        'Day': [],
        'PctCDX2': [],
        'Filename': [],
    }

    for prefix, threshold in thresholds.items():
        prefix_df = all_df[all_df['Prefix'] == prefix]

        for filename in np.unique(prefix_df['Filename']):
            section_df = prefix_df[prefix_df['Filename'] == filename]
            total_cells = section_df.shape[0]
            positive_cells = np.sum(section_df['CellCDX2'] >= threshold)

            pct_cdx2 = positive_cells / total_cells
            pct_df['Condition'].append(section_df['Condition'].values[0])
            pct_df['Day'].append(section_df['Day'].values[0])
            pct_df['PctCDX2'].append(pct_cdx2*100)
            pct_df['Filename'].append(section_df['Filename'].values[0])
    pct_df = pd.DataFrame(pct_df)

    print(pct_df)
    significance = {}
    for day in np.unique(pct_df['Day']):
        day_df = pct_df[pct_df['Day'] == day]
        left = day_df[day_df['Condition'] == '0uM']
        right = day_df[day_df['Condition'] == '4uM']

        _, pval = ttest_ind(left['PctCDX2'].values, right['PctCDX2'].values)
        if pval < 0.05:
            significance[((day, '0uM'), (day, '4uM'))] = pval
    print(significance)

    outfile = outdir / 'cdx2_stats.png'
    savefile = outdir / 'cdx2_stats.xlsx'

    with set_plot_style('light') as style:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        add_barplot(ax, pct_df, x='Day', hue='Condition', y='PctCDX2',
                    significance=significance, savefile=savefile)
        ax.set_ylabel('% CDX2+ Cells')
        style.show(outfile=outfile)

    replot_bars(savefile,
                outfile=outdir / 'cdx2_stats_final.pdf',
                suffixes=('.png', '.pdf', '.svg'),
                xcolumn='Day',
                ycolumn='PctCDX2',
                hue_column='Condition',
                xlabel='Day',
                ylabel='% CDX2+ Cells',
                ylimits=[(0.0, 75.0)],
                palette='wheel_bluegrey')


def quant_tbxt_kd_stains(rootdir: pathlib.Path,
                         plot_style: str = PLOT_STYLE):
    outdir = rootdir / 'plots_tbxt_kd'
    if outdir.is_dir():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if rootdir.name == 'TBXT-KD (Dox and NoDox)':
        patterns = [
            re.compile(r'^.*_555TBXT_DAPi_', re.IGNORECASE),
            re.compile(r'^.*_488SOX2_555TBXT_', re.IGNORECASE),
        ]
    elif rootdir.name == '20210305_TKD_Worms':
        patterns = [
            None,
            re.compile(r'^.*_488SOX2_555TBXT_', re.IGNORECASE),
        ]
    else:
        raise OSError(f'Unknown base Directory: {rootdir}')

    for infile in rootdir.iterdir():
        if infile.suffix != '.czi':
            continue
        if '_TBXT-KD_' not in infile.name:
            continue
        is_matched = False
        for i, pattern in enumerate(patterns):
            if pattern is None:
                continue
            if not pattern.match(infile.name):
                continue
            if i == 0:
                stain_table = {
                    'af350': 'dapi',
                    'af555': 'tbxt',
                }
            elif i == 1:
                stain_table = {
                    'af350': 'dapi',
                    'af488': 'sox2',
                    'af555': 'tbxt',
                }
            else:
                stain_table = None

            is_matched = True
            break
        if not is_matched:
            continue
        if not infile.is_file():
            continue
        print(infile)
        proc = TBXTQuant(infile, outdir, stain_table=stain_table)
        proc.load_images()
        proc.segment_dapi_islands()
        proc.segment_cells()
        # proc.plot_distributions()
        proc.plot_images()
        proc.save_stats()


def analyze_quant_tbxt_kd_stains(rootdir: pathlib.Path,
                                 plot_style: str = PLOT_STYLE):
    """ Analyze the TBXT quantification """

    indirs = [
        rootdir.parent / 'TBXT-KD (Dox and NoDox)' / 'plots_tbxt_kd',
        rootdir.parent / '20210305_TKD_Worms' / 'plots_tbxt_kd',
    ]

    outdir = rootdir / 'quant_tbxt_kd'
    if outdir.is_dir():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    skipfiles = {}

    all_df = []
    for indir in indirs:
        for infile in indir.iterdir():
            if infile.name.startswith('.') or not infile.name.endswith('.xlsx'):
                continue
            if infile.name in skipfiles:
                continue
            prefix = infile.stem[:-len('_stats')]
            prefix = prefix.rsplit('_', 1)[0]
            print(prefix)

            df = pd.read_excel(infile)
            df['Prefix'] = prefix
            df['Filename'] = infile.stem
            all_df.append(df)
    all_df = pd.concat(all_df, ignore_index=True)

    mean_df = all_df[['Prefix', 'CellTBXT']].groupby('Prefix', as_index=False).mean()

    for i, rec in mean_df.iterrows():
        prefix = rec['Prefix']
        prefix_mean = rec['CellTBXT']

        print(f'{prefix}: {prefix_mean}')

        sub_df = all_df[all_df['Prefix'] == prefix]

        with set_plot_style('light') as style:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            add_histogram(ax, sub_df['CellTBXT'])
            ax.axvline(prefix_mean, color='r')

            ax.set_title(prefix)
            style.show(outfile=outdir / f'replicates{i:02d}.png')

    thresholds = {
        'TBXT-KD_NoDox_D3_slide_555TBXT_DAPi_20200706': 7000,
        'TBXT-KD_5Dox_D3_slide2_555TBXT_DAPi_20200706': 1000,
        # 'TBXT-KD_NoDox_D3_slide_555TBXT_DAPi_20200706_10x': 700,
        '20210305_TBXT-KD_5DOX_D5_350Dapi_488SOX2_555TBXT_20x': 1000,
        '20210305_TBXT-KD_NoDOX_350Dapi_488SOX2_555TBXT_20x': 1000,
        'TBXT-KD_5Dox_D7_slide7_555TBXT_DAPi_20200706': 700,
        'TBXT-KD_NoDox_D7_slide_555TBXT_DAPi_20200706': 700,
        'TBXT-KD_5Dox_D10_slide4.1_488SOX2_555TBXT_20x_20200716': 500,
        'TBXT-KD_5Dox_D10_slide4.1_488SOX2_555TBXT_5x_20200716': 500,
        'TBXT-KD_NoDOX_D10_slide2.1_488SOX2_555TBXT_20200716': 500,
    }
    pct_df = {
        'Condition': [],
        'Day': [],
        'PctTBXT': [],
        'Filename': [],
    }

    for prefix, threshold in thresholds.items():
        prefix_df = all_df[all_df['Prefix'] == prefix]

        for filename in np.unique(prefix_df['Filename']):
            section_df = prefix_df[prefix_df['Filename'] == filename]
            total_cells = section_df.shape[0]
            positive_cells = np.sum(section_df['CellTBXT'] >= threshold)

            pct_tbxt = positive_cells / total_cells
            pct_df['Condition'].append(section_df['Condition'].values[0])
            pct_df['Day'].append(section_df['Day'].values[0])
            pct_df['PctTBXT'].append(pct_tbxt*100)
            pct_df['Filename'].append(section_df['Filename'].values[0])
    pct_df = pd.DataFrame(pct_df)

    print(pct_df)
    significance = {}
    for day in np.unique(pct_df['Day']):
        day_df = pct_df[pct_df['Day'] == day]
        left = day_df[day_df['Condition'] == 'TBXT-KD']
        right = day_df[day_df['Condition'] == 'WT']

        _, pval = ttest_ind(left['PctTBXT'].values, right['PctTBXT'].values)
        if pval < 0.05:
            significance[((day, 'TBXT-KD'), (day, 'WT'))] = pval
    print(significance)

    outfile = outdir / 'tbxt_stats.png'
    savefile = outdir / 'tbxt_stats.xlsx'

    with set_plot_style('light') as style:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        add_barplot(ax, pct_df, x='Day', hue='Condition', y='PctTBXT',
                    significance=significance, savefile=savefile)
        ax.set_ylabel('% TBXT+ Cells')
        style.show(outfile=outfile)

    replot_bars(savefile,
                outfile=outdir / 'tbxt_stats_final.pdf',
                suffixes=('.png', '.pdf', '.svg'),
                xcolumn='Day',
                ycolumn='PctTBXT',
                hue_column='Condition',
                xlabel='Day',
                ylabel='% TBXT+ Cells',
                ylimits=[(0.0, 75.0)],
                palette='wheel_bluegrey')

# Command line


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('-e', '--experiment', choices=('ph3', 'edu'), default='ph3')
    parser.add_argument('rootdir', type=pathlib.Path)
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    # quant_tbxt_stains(**vars(args))
    # analyze_quant_tbxt_stains(**vars(args))

    # quant_cdx2_stains(**vars(args))
    # analyze_quant_cdx2_stains(**vars(args))

    # quant_tbxt_kd_stains(**vars(args))
    analyze_quant_tbxt_kd_stains(**vars(args))


if __name__ == '__main__':
    main()
