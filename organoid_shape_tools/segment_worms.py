""" Tools for segmenting timelapse phase images

Classes:

* :py:class:`WormSegmentation`: Segmentation pipeline for tracking objects in timelapse brightfield

Functions:

* :py:func:`background_inpaint`: Inpaint to remove holes in a background mask
* :py:func:`trunc_quad`: Fit a trunctated 2D quadratic function
* :py:func:`read_contour_file`: Read the contour files back

These functions are used by the following scripts:

* ``segment_worms.py``: Main segmentation script for timelapse images
* ``analyze_segment_worms.py``: Make the final plots for the timelapse analysis

API Documentation
-----------------

"""

# Standard lib
import pathlib
import shutil
import json
import traceback
from collections import Counter
from typing import Tuple, Optional, Dict, List

# 3rd party
import numpy as np

from PIL import Image

import pandas as pd

from scipy.optimize import curve_fit
from scipy import ndimage as ndi
from scipy.interpolate import griddata

import matplotlib.pyplot as plt

from skimage.filters import gaussian
from skimage.feature import canny
from skimage import segmentation, exposure, filters, morphology
from skimage.measure import label, regionprops, EllipseModel

# Our own imports
from .utils import (
    czi_utils, contours_from_mask, mask_from_contours, area_of_polygon,
    center_of_polygon, perimeter_of_polygon, to_json_types, save_image, load_image,
)
from .plotting import set_plot_style, add_scalebar

# Scale factor from pixels to mm
SPACE_SCALE: float = 1.3  # mm per pixel - for the 5x tortoise lens (2.5 is 2.6 um/pixel)
TIME_SCALE: float = 20.0  # minutes per frame

# Smoothing Constants
FOREGROUND_SMOOTHING: Tuple[float] = (1, 1)  # Gaussian smoothing for the foreground image
BACKGROUND_SMOOTHING: Tuple[float] = (30, 30)  # Gaussian smoothing to estimate the image background

THRESHOLD_ABS: Optional[float] = 0.05  # Threshold between background and foreground after smoothing

BG_MAX_VALUE: float = 24000  # Maximum value in the image
BG_MIN_VALUE: float = 200  # Minimum value in the image

DIFFUSION_BETA: float = 4000  # Diffusion coefficient, larger makes edges stronger

STD_REPLOT: Optional[List[int]] = None
STD_DEFAULT_THRESHOLD: float = 0.03
STD_THRESHOLDS: Dict[int, float] = {}

# BX, BY, Width, Height
BBOX_ROIS = {
    1: (813, 678, 549, 528),
    2: (180, 805, 972, 774),
    3: (786, 723, 561, 495),
    4: [(930, 620, 915, 705), (1344, 0, 564, 576)],  # Fused... ish
    5: (846, 861, 603, 462),
    6: (741, 663, 582, 579),
    7: (690, 747, 537, 465),
    8: (546, 879, 810, 645),
    9: (831, 700, 636, 870),
    10: (696, 813, 624, 690),
    11: (807, 1161, 576, 621),
    12: (771, 927, 741, 753),
    13: (822, 678, 564, 564),
    14: [(885, 786, 570, 672), (520, 1500, 900, 567)],
    15: (645, 657, 666, 531),
    16: (648, 756, 615, 720),
    17: [(810, 280, 528, 534), (843, 720, 516, 417)],
    18: (834, 552, 531, 621),
    19: (640, 795, 774, 462),
    20: (744, 618, 576, 486),
    21: (500, 594, 927, 606),
    22: (810, 732, 525, 570),
    23: (765, 645, 555, 615),
    24: (828, 612, 606, 630),
    25: (609, 880, 528, 840),
    26: [(339, 279, 663, 483), (630, 1179, 921, 618)],
    27: (798, 675, 525, 597),
    28: (894, 816, 564, 630),
    29: [(546, 249, 564, 570), (1161, 777, 630, 738)],
    30: (798, 663, 411, 534),
    31: (552, 864, 606, 858),  # Fused
    32: (657, 699, 549, 660),
    33: (411, 903, 777, 693),
    34: [(519, 963, 582, 642), (807, 663, 567, 573)],  # Fused
    35: (291, 1257, 726, 645),
    36: (780, 717, 585, 480),
    37: (135, 819, 744, 408),
    38: (609, 576, 747, 648),
    39: (741, 639, 534, 594),
    40: (456, 1164, 582, 639),
    41: [(320, 1140, 852, 549), (1212, 408, 534, 522)],
    42: [(441, 282, 993, 1293)],  # Fused and too big
    43: (1353, 513, 597, 687),
    44: (825, 792, 606, 558),
    45: (1293, 1002, 603, 519),
    46: (372, 789, 621, 567),
    47: (801, 831, 546, 663),
    48: (498, 1074, 588, 462),
    49: (810, 1307, 636, 627),
    50: (648, 822, 546, 705),
    51: (801, 1470, 576, 519),
    52: (402, 762, 636, 564),
    53: (768, 732, 477, 516),
    54: (393, 822, 579, 510),
    55: (852, 792, 558, 588),
    56: (618, 240, 711, 768),
    57: (354, 684, 702, 597),
    58: (399, 993, 471, 516),
    59: (906, 726, 540, 528),
    60: (654, 576, 654, 627),
    61: (780, 600, 606, 588),
    62: (780, 80, 657, 945),
}

# Main Class


class WormSegmentation(object):
    """ Segment the worms

    This class implements a multi-stage workflow. Run each stage and then manually
    inspect the results to see how parameters need to be adapted to your images.

    1. Initialize the pipeline and configure the parameters:

    .. code-block:: python

            with WormSegmentation(czifile,
                                  outdir=outdir,
                                  overwrite=overwrite) as proc:
                proc.set_params()
                proc.create_outdir()
                proc.load_params()
                proc.save_params()

    2. Estimate overall background and locate moving objects within the field

    .. code-block:: python

        proc.estimate_background()
        proc.find_delta_rois()

    3. Subset the images to focus only on the moving objects:

    .. code-block:: python

        proc.subset_tiles_rois()
        proc.subset_tiles()

    4a. Background subtract and segment (good for monochrome aggregates):

    .. code-block:: python

        proc.segment_all_tile_foregrounds()
        proc.segment_all_frames()

    4b. Edge enhance and segment (good for highly textured aggregates):

    .. code-block:: python

        proc.find_delta_segments()

    5. Merge and plot the final segmentations:

    .. code-block:: python

        proc.plot_final_segmentations()
        proc.gather_final_segmentations()

    See the ``scripts/segment_worms.py`` script for the current workflow

    :param Path czifile:
        The raw czi images
    :param Path outdir:
        The directory to write data to
    :param bool overwrite:
        If True, overwrite the data
    """

    def __init__(self,
                 czifile: pathlib.Path,
                 outdir: pathlib.Path,
                 overwrite: bool = False):

        # Main czi file object
        self.czifile = czifile
        self._czi = None

        self.outdir = outdir
        self.out_background_imgfile = self.outdir / 'background_image.tif'
        self.out_background_coords = self.outdir / 'background_coords.xlsx'

        self.out_maskdir = self.outdir / 'masks'

        self.out_roi_coords = self.outdir / 'roi_coords.xlsx'
        self.out_roi_frames = self.outdir / 'roi_frames'

        self.out_subset_coords = self.outdir / 'subset_coords.xlsx'
        self.out_subset_frames = self.outdir / 'subset_frames'

        self.overwrite = overwrite

        self.threshold_paramfile = self.outdir / 'threshold_params.xlsx'

        # Which tiles to segment
        self.tile_numbers = list(range(1, 21))

        # Parameters
        self.row_width = 1000
        self.col_width = 1000

        # Caches for various coordinates
        self.rows = None
        self.cols = None
        self.xx = None
        self.yy = None

        # Background image
        self.bg_img = None
        self.bg_xx = None
        self.bg_yy = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def open(self):
        """ Load the CZI file """
        return
        self._czi = czi_utils.CZIManager(self.czifile)
        self._czi.open()

    def close(self):
        """ Close the CZI file """
        if self._czi is not None:
            self._czi.close()
        self._czi = None

    @property
    def czi(self) -> czi_utils.CZIManager:
        if self._czi is None:
            self._czi = czi_utils.CZIManager(self.czifile)
            self._czi.open()
        return self._czi

    def load_params(self):
        """ Load all the parameter files """
        if not self.threshold_paramfile.is_file():
            std_thresholds = STD_THRESHOLDS
        else:
            print(f'Loading thresholds from {self.threshold_paramfile}')
            std_thresholds = {}
            df = pd.read_excel(self.threshold_paramfile)
            for _, rec in df.iterrows():
                key = int(rec['Tile'])
                value = float(rec['Threshold'])

                assert key not in std_thresholds
                std_thresholds[key] = value
        self.std_thresholds = std_thresholds

    def save_params(self):
        """ Save all the parameter files """
        # Save the threshold parameter file
        print(f'Saving thresholds to {self.threshold_paramfile}')
        df = {'Tile': [], 'Threshold': []}
        for key, value in self.std_thresholds.items():
            df['Tile'].append(key)
            df['Threshold'].append(value)
        df = pd.DataFrame(df)
        df.to_excel(self.threshold_paramfile, index=False)

    def set_params(self):
        """ Set all the initial processing parameters """
        self.czi.set_contrast_correction('raw')

    def create_outdir(self):
        """ Create the output directory """
        keep_paths = [self.threshold_paramfile]

        if self.overwrite and self.outdir.is_dir():
            print(f'Overwriting {self.outdir}')
            for p in self.outdir.iterdir():
                if p in keep_paths:
                    continue
                if p.is_file():
                    p.unlink()
                    continue
                if p.is_dir():
                    shutil.rmtree(p)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def calc_bbox(self, cx: float, cy: float) -> Tuple:
        """ Calculate the bounding box around a centroid

        :param float cx:
            The center row coordinate of the bbox
        :param float cy:
            The center column coordinate of the bbox
        :returns:
            The x and y coordinates of the box corners, in image coordinates
        """

        cx_minx = int(round(cx - self.row_width / 2))
        cx_minx = max([cx_minx, 0])
        cx_maxx = cx_minx + self.row_width
        if cx_maxx > self.rows:
            cx_maxx = self.rows
            cx_minx = cx_maxx - self.row_width

        cy_miny = int(round(cy - self.col_width / 2))
        cy_miny = max([cy_miny, 0])
        cy_maxy = cy_miny + self.col_width
        if cy_maxy > self.cols:
            cy_maxy = self.cols
            cy_miny = cy_maxy - self.col_width
        return cx_minx, cx_maxx, cy_miny, cy_maxy

    def estimate_background(self):
        """ Estimate background from the first few frames """

        # if self.out_background_coords.is_file() and self.out_background_imgfile.is_file():
        #     return

        centers = {
            'CenterX': [],
            'CenterY': [],
            'Radius': [],
            'MinX': [],
            'MaxX': [],
            'MinY': [],
            'MaxY': [],
            'Tile': [],
            'Filename': [],
        }
        center_frames = []

        multi_xx = multi_yy = None

        valid_roots = 0
        for i, (filename, metadata, frame) in enumerate(self.czi.iter_images()):
            if metadata['timepoint'] > 1:
                break
            tileno = metadata['tile']
            if tileno not in self.tile_numbers:
                print(f'Skipping tile {tileno}...')
                continue
            print(f'Frame {i}: Loading background for tile {tileno}')

            if frame.ndim == 3:
                frame = frame[:, :, 0]
            frame = frame.astype(np.float64) / BG_MAX_VALUE

            if self.rows is None and self.cols is None:
                rows, cols = frame.shape
                self.rows = rows
                self.cols = cols

                # Make the coordinates too
                xx, yy = np.meshgrid(np.arange(rows), np.arange(cols))
                self.xx = xx
                self.yy = yy

                assert xx.shape == (rows, cols)
                assert yy.shape == (rows, cols)

                multi_xx, multi_yy = np.meshgrid(np.arange(-rows, rows), np.arange(-cols, cols))
            else:
                assert frame.shape == (rows, cols)

            #print(f'Frame {i}: Inital range {np.min(frame)},{np.max(frame)}')

            # Smooth to allow the gaussian to fit more robustly
            bg_frame = filters.gaussian(frame, BACKGROUND_SMOOTHING)

            #print(f'Frame {i}: Smooth range {np.min(frame)},{np.max(frame)}')

            # Fit a quadratic surface to estimate the distortion
            x = xx.flatten()
            y = yy.flatten()
            a = np.array([x**2, y**2, x, y, x*y, x*0 + 1]).T
            b = bg_frame.flatten()
            coeff, r0, rank, s = np.linalg.lstsq(a, b, rcond=None)
            print(f'Frame {i}: initial radius {r0} with params {coeff}')

            # Look at the initial guesses
            a, b, c, d, e, _ = coeff
            init_root = 4*a*b - e**2
            print(f'Frame {i}: initial root {init_root}')
            cx = -(2*b*c - d*e)/init_root
            cy = -(2*a*d - c*e)/init_root
            print(f'Frame {i}: initial center at {cx:0.2f}, {cy:0.2f}')

            # Fit a truncated quadratic surface using the quad fit as a seed
            popt, pcov = curve_fit(f=trunc_quad,
                                   xdata=np.stack([x, y], axis=0),
                                   ydata=bg_frame.flatten(),
                                   p0=list(coeff) + [rows/2])

            # Estimate the final centers from the optimizer
            a, b, c, d, e, f, radius = popt
            root = 4*a*b - e**2
            if root <= 0:
                print(f'Frame {i}: Got invalid root: {root} at radius {radius}')
                continue
            if radius < 200:
                print(f'Frame {i}: Got invalid radius: {radius} with root {root}')
                continue

            cx = -(2*b*c - d*e)/root
            cy = -(2*a*d - c*e)/root
            print(f'Frame {i}: Center at {cx:0.2f}, {cy:0.2f}')

            # Coordinates get flipped because images vs points...
            minx = int(np.min(xx - np.floor(cy)) + rows)
            minx = max([0, minx])
            maxx = minx + rows

            miny = int(np.min(yy - np.floor(cx)) + cols)
            miny = max([0, miny])
            maxy = miny + cols

            rr = (xx - cx)**2 + (yy - cy)**2

            # Mask off any really low values and anything outside the circle
            frame[frame < BG_MIN_VALUE/BG_MAX_VALUE] = np.nan
            frame[rr > radius**2] = np.nan

            mask_frame = np.full(multi_xx.shape, fill_value=np.nan, dtype=np.float64)
            mask_frame[minx:maxx, miny:maxy] = frame

            # Metadata to save off
            centers['CenterX'].append(cx)
            centers['CenterY'].append(cy)
            centers['Radius'].append(radius)

            centers['MinX'].append(minx)
            centers['MaxX'].append(maxx)
            centers['MinY'].append(miny)
            centers['MaxY'].append(maxy)

            centers['Tile'].append(metadata['tile'])
            centers['Filename'].append(filename.name)

            center_frames.append(mask_frame)

            valid_roots += 1
            if valid_roots > 5:
                break

        # Write out the aligned image coordinates
        df = pd.DataFrame(centers)
        df.to_excel(str(self.out_background_coords))

        # Take the median to get the central background
        center_mask = np.nanmedian(center_frames, axis=0)
        center_mask[np.isnan(center_mask)] = 0

        center_mask = np.round(center_mask * BG_MAX_VALUE)
        center_mask[center_mask < 0] = 0
        center_mask[center_mask > BG_MAX_VALUE] = BG_MAX_VALUE
        center_mask = center_mask.astype(np.uint16)

        # Cache the final background image
        img = Image.fromarray(center_mask)
        img.save(str(self.out_background_imgfile))

        alt_mask = np.nanmean(center_frames, axis=0)
        alt_mask[np.isnan(alt_mask)] = 0

        alt_mask = np.round(alt_mask * BG_MAX_VALUE)
        alt_mask[alt_mask < 0] = 0
        alt_mask[alt_mask > BG_MAX_VALUE] = BG_MAX_VALUE
        alt_mask = alt_mask.astype(np.uint16)

        # Cache the final background image
        img = Image.fromarray(alt_mask)
        img.save(str(self.outdir / 'background_mean.tif'))

    def find_background_mask(self,
                             exponent: float = 2.0,
                             fg_sigma: float = 2.0,
                             bg_sigma: float = 60.0,
                             threshold: float = 0.001,
                             min_hole_size: int = 200,
                             min_object_size: int = 2000,
                             min_contour_size: int = 10000):
        """ For each tile, extract a background/foreground mask """

        outdir = self.out_roi_frames
        if self.overwrite and outdir.is_dir():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        thresholds = {
            1: 0.001,
            2: 0.0001,
            3: 0.001,
            4: 0.001,
            5: 0.001,
            6: 0.001,
            7: 0.0001,
            8: 0.00005,
            9: 0.0001,
            10: 0.001,
            11: 0.00005,
            12: 0.0001,
            13: 0.001,
            14: 0.0005,
            15: 0.001,
            16: 0.001,
            17: 0.001,
            18: 0.001,
            19: 0.00005,
            20: 0.0001,
        }

        rows, cols = None, None

        all_segments = Counter()

        selem = morphology.disk(radius=6)

        for i, (filename, metadata, frame) in enumerate(self.czi.iter_images()):
            tileno = metadata['tile']
            timepoint = metadata['timepoint']

            if tileno not in self.tile_numbers:
                print(f'Skipping delta estimation for tile {tileno}...')
                continue

            out_maskfile = outdir / 'masks' / f'mask_s{tileno:02d}.tif'
            out_bgfile = outdir / 'backgrounds' / f'bg_s{tileno:02d}.tif'

            if out_maskfile.is_file() and out_bgfile.is_file():
                continue
            if timepoint > 1:
                break
            print(f'Segmenting the background for s{tileno:02d}')

            key = (tileno, timepoint)
            rep = all_segments[key] + 1
            all_segments[key] += 1

            threshold = thresholds[tileno]

            if rows is None and cols is None:
                rows, cols = frame.shape[:2]
            else:
                assert (rows, cols) == frame.shape[:2]

            tile_outdir = outdir / f's{tileno:02d}-{rep:d}'
            tile_outdir.mkdir(exist_ok=True, parents=True)

            outfile = tile_outdir / f'delta_s{tileno:02d}-{rep:d}t{timepoint:03d}.tif'

            # Background subtract the raw frame and then sharpen the edges
            fg_frame = gaussian(frame, fg_sigma)
            bg_frame = gaussian(frame, bg_sigma)
            cur_frame = np.abs(fg_frame - bg_frame)**exponent
            cur_frame = cur_frame + canny(cur_frame)*0.5

            low_mask = cur_frame > threshold
            low_mask = morphology.binary_dilation(low_mask, selem=selem)
            low_mask = morphology.remove_small_holes(low_mask, min_hole_size)
            low_mask = morphology.remove_small_objects(low_mask, min_object_size)
            low_mask = morphology.binary_erosion(low_mask, selem=selem)

            if np.any(low_mask):
                low_seg = contours_from_mask(low_mask, 0.5)
            else:
                low_seg = []

            final_mask = np.zeros_like(low_mask, dtype=np.bool)

            for i, seg in enumerate(low_seg):
                area = area_of_polygon(seg)
                if area < min_contour_size:
                    continue
                print(f'Area of poly {i}: {area}')
                mask = mask_from_contours([seg], final_mask.shape)
                final_mask[mask] = True

            if np.any(final_mask):
                final_seg = contours_from_mask(final_mask, 0.5)
            else:
                final_seg = []

            # Use the mask to estimate a background field
            final_bg = frame.copy()
            for seg in final_seg:
                mask = mask_from_contours([seg], final_mask.shape)
                ring = morphology.binary_dilation(mask, selem=selem)
                ring = np.logical_and(ring, ~mask)
                val = np.nanmedian(frame[ring])
                final_bg[mask] = val

            with set_plot_style('light') as style:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
                ax1.imshow(frame, cmap='gray')
                for seg in final_seg:
                    ax1.plot(seg[:, 0], seg[:, 1], '-r', linewidth=1)

                ax1.set_xlim([0, cols])
                ax1.set_ylim([rows, 0])
                ax1.set_xticks([])
                ax1.set_yticks([])

                ax2.imshow(cur_frame, cmap='inferno', vmin=0, vmax=3*threshold)
                ax2.set_xlim([0, cols])
                ax2.set_ylim([rows, 0])
                ax2.set_xticks([])
                ax2.set_yticks([])

                ax3.imshow(final_mask, cmap='tab20b')
                ax3.set_xlim([0, cols])
                ax3.set_ylim([rows, 0])
                ax3.set_xticks([])
                ax3.set_yticks([])

                ax4.imshow(final_bg, cmap='gray')
                ax4.set_xlim([0, cols])
                ax4.set_ylim([rows, 0])
                ax4.set_xticks([])
                ax4.set_yticks([])

                style.show(outfile=outfile)

            save_image(out_maskfile, final_mask, cmin=0, cmax=1)
            save_image(out_bgfile, final_bg)

    def find_delta_segments(self,
                            default_exponent: float = 0.5,
                            default_threshold: float = 0.25,
                            min_hole_size: int = 200,
                            min_object_size: int = 2000,
                            min_contour_size: int = 10000,
                            erosion_rounds: int = 7,
                            backward_window: int = 3,
                            forward_window: int = 3):
        """ Use the background mask to find deltas

        :param float default_exponent:
            Exponent for the delta image curve (0.5 works good to start)
        :param float default_threshold:
            Threshold for the delta image to segment foreground (high) from background (low)
        :param int erosion_rounds:
            How many rounds to erode the initial segmentation mask
        :param int backward_window:
            How many time steps to look backwards to find an initial segmentation mask
        :param int forward_window:
            How many time steps to look forwards to find an initial segmentation mask
        """

        maskdir = self.out_maskdir
        outdir = self.out_roi_frames
        if self.overwrite and outdir.is_dir():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        thresholds = {
            1: 0.2,
            2: 0.1,
            3: 0.2,
            4: 0.15,
            5: 0.2,
            6: 0.2,
            7: 0.15,
            8: 0.15,
            9: 0.15,
            10: 0.2,
            11: 0.15,
            12: 0.10,
            13: 0.2,
            14: 0.15,
            15: 0.25,
            16: 0.17,
            17: 0.25,
            18: 0.2,
            19: 0.15,
            20: 0.15,
        }
        exponents = {
            2: 0.5,
        }

        rows, cols = None, None

        selem = morphology.disk(radius=6)

        # Clear any incomplete timepoint records
        print('Clearing old incomplete records...')
        for tileno in self.tile_numbers:
            out_plotdir = outdir / f's{tileno:02d}' / 'plots'
            out_maskdir = outdir / f's{tileno:02d}' / 'masks'
            out_contourdir = outdir / f's{tileno:02d}' / 'contours'
            for timepoint in range(1, 146):
                out_maskfile = out_maskdir / f'mask_s{tileno:02d}s{timepoint:03d}.tif'
                out_contourfile = out_contourdir / f'coords_s{tileno:02d}s{timepoint:03d}.csv'
                out_plotfile = out_plotdir / f'plot_s{tileno:02d}s{timepoint:03d}.tif'

                if all([p.is_file() for p in (out_maskfile, out_contourfile, out_plotfile)]):
                    continue
                if any([p.is_file() for p in (out_maskfile, out_contourfile, out_plotfile)]):
                    print(f'Clearing bad record s{tileno:02d}t{timepoint:03d}')
                if out_maskfile.is_file():
                    out_maskfile.unlink()
                if out_contourfile.is_file():
                    out_contourfile.unlink()
                if out_plotfile.is_file():
                    out_plotfile.unlink()

        # Now segment missing records
        for filename, metadata, frame in self.czi.iter_images():
            tileno = metadata['tile']
            timepoint = metadata['timepoint']

            if tileno not in self.tile_numbers:
                print(f'Skipping delta estimation for tile {tileno}...')
                continue

            out_plotdir = outdir / f's{tileno:02d}' / 'plots'
            out_maskdir = outdir / f's{tileno:02d}' / 'masks'
            out_contourdir = outdir / f's{tileno:02d}' / 'contours'

            out_maskdir.mkdir(parents=True, exist_ok=True)
            out_plotdir.mkdir(parents=True, exist_ok=True)
            out_contourdir.mkdir(parents=True, exist_ok=True)

            out_maskfile = out_maskdir / f'mask_s{tileno:02d}s{timepoint:03d}.tif'
            out_contourfile = out_contourdir / f'coords_s{tileno:02d}s{timepoint:03d}.csv'
            out_plotfile = out_plotdir / f'plot_s{tileno:02d}s{timepoint:03d}.tif'

            if all([p.is_file() for p in (out_maskfile, out_contourfile, out_plotfile)]):
                print(f'Skipping already processed frame s{tileno:02d}t{timepoint:03d}')
                continue

            print(f'Segmenting frame s{tileno:02d}t{timepoint:03d}...')

            # Use the masks from the last few frames
            bg_mask = np.zeros_like(frame, dtype=np.bool)
            for j in range(-backward_window, forward_window+1):
                maskfile = out_maskdir / f'mask_s{tileno:02d}s{timepoint+j:03d}.tif'
                if not maskfile.is_file():
                    continue
                bg_mask = np.logical_or(bg_mask, load_image(maskfile) > 100)

            # No valid masks yet, seed with the background mask
            if not np.any(bg_mask):
                maskfile = maskdir / f'mask_s{tileno:02d}.tif'
                bg_mask = load_image(maskfile) > 100

            # Dilate the mask to allow for movement
            for j in range(erosion_rounds):
                bg_mask = morphology.binary_dilation(bg_mask, selem=selem)

            frame_min = np.min(frame)
            frame_max = np.max(frame)
            frame_range = frame_max - frame_min

            frame = (frame - frame_min) / frame_range
            bg_inpaint = background_inpaint(frame, bg_mask)

            exponent = exponents.get(tileno, default_exponent)
            threshold = thresholds.get(tileno, default_threshold)

            delta_img = np.abs(frame - bg_inpaint)**exponent

            low_mask = delta_img > threshold

            # Seem to be getting a lot of shot noise
            low_mask = morphology.remove_small_objects(low_mask, 50)

            # Inflate, close holes, remove noise, erode
            low_mask = morphology.binary_dilation(low_mask, selem=selem)
            low_mask = morphology.remove_small_holes(low_mask, min_hole_size)
            low_mask = morphology.remove_small_objects(low_mask, min_object_size)
            low_mask = morphology.binary_erosion(low_mask, selem=selem)

            if np.any(low_mask):
                low_seg = contours_from_mask(low_mask, 0.5)
            else:
                low_seg = []

            final_mask = np.zeros_like(low_mask, dtype=np.bool)

            for j, seg in enumerate(low_seg):
                area = area_of_polygon(seg)
                if area < min_contour_size:
                    continue
                print(f'Area of poly {j+1}: {area}')
                mask = mask_from_contours([seg], final_mask.shape)
                final_mask[mask] = True

            if np.any(final_mask):
                final_seg = contours_from_mask(final_mask, 0.5)
            else:
                final_seg = []

            # Write the mask out
            save_image(out_maskfile, final_mask)

            # Write the contours out
            with out_contourfile.open('wt') as fp:
                for j, seg in enumerate(final_seg):
                    fp.write(f'Track{j+1}\n')
                    fp.write('XCoords,' + ','.join(f'{x:0.4f}' for x in seg[:, 0]) + '\n')
                    fp.write('YCoords,' + ','.join(f'{y:0.4f}' for y in seg[:, 1]) + '\n')

            # Make a debugging plot
            with set_plot_style('light') as style:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

                ax1.imshow(frame, cmap='gray')
                for j, seg in enumerate(final_seg):
                    ax1.plot(seg[:, 0], seg[:, 1], '-r')
                    cx, cy = center_of_polygon(seg)
                    ax1.text(cx, cy, f'Track{j+1}', fontsize=24, weight='bold', color='k',
                             horizontalalignment='center', verticalalignment='center')
                    ax1.text(cx, cy, f'Track{j+1}', fontsize=24, color='w',
                             horizontalalignment='center', verticalalignment='center')
                ax1.set_xlim([0, cols])
                ax1.set_ylim([rows, 0])
                ax1.set_xticks([])
                ax1.set_yticks([])

                ax2.imshow(delta_img, cmap='inferno', vmin=0, vmax=3*threshold)
                ax2.set_xlim([0, cols])
                ax2.set_ylim([rows, 0])
                ax2.set_xticks([])
                ax2.set_yticks([])

                ax3.imshow(final_mask, cmap='gray')
                ax3.set_xlim([0, cols])
                ax3.set_ylim([rows, 0])
                ax3.set_xticks([])
                ax3.set_yticks([])

                ax4.imshow(bg_inpaint, cmap='gray')
                for seg in final_seg:
                    ax4.plot(seg[:, 0], seg[:, 1], '-r')
                ax4.set_xlim([0, cols])
                ax4.set_ylim([rows, 0])
                ax4.set_xticks([])
                ax4.set_yticks([])

                fig.suptitle(f's{tileno:02d}t{timepoint:03d}: Threshold {threshold} Exp {exponent}')

                style.show(outfile=out_plotfile)

    def plot_final_segmentations(self):
        """ Plot the final segmentations for each tile """

        outdir = self.out_roi_frames
        outdir.mkdir(parents=True, exist_ok=True)

        rows = cols = None

        self.czi.dump_metadata(self.outdir / 'metadata.xml')

        # Now segment missing records
        for filename, metadata, frame in self.czi.iter_images():
            tileno = metadata['tile']
            timepoint = metadata['timepoint']

            if tileno not in (6, ):
                print(f'Skipping delta estimation for tile {tileno}...')
                continue

            if rows is None and cols is None:
                rows, cols = frame.shape
            else:
                assert frame.shape == (rows, cols)

            out_contourdir = outdir / f's{tileno:02d}' / 'contours'
            out_contourdir.mkdir(parents=True, exist_ok=True)

            out_plotdir = outdir / f's{tileno:02d}-final' / 'final_plots'
            out_plotdir.mkdir(parents=True, exist_ok=True)

            out_contourfile = out_contourdir / f'coords_s{tileno:02d}s{timepoint:03d}.csv'
            out_plotfile = out_plotdir / f'final_plot_s{tileno:02d}s{timepoint:03d}.tif'

            segments = read_contour_file(out_contourfile)

            largest_coords = None
            largest_area = -1

            print(f'Got {len(segments)} tracks')
            for key, coords in segments.items():
                area = area_of_polygon(coords)
                print(f'{key} with {coords.shape[0]} points has area {area}')
                if area > largest_area:
                    largest_area = area
                    largest_coords = coords
            assert largest_coords is not None

            frame_min = np.min(frame)
            frame_max = np.max(frame)
            frame_range = frame_max - frame_min

            frame = (frame - frame_min) / frame_range

            with set_plot_style('light') as style:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))

                ax.imshow(frame, cmap='gray')
                ax.plot(largest_coords[:, 0], largest_coords[:, 1], '-r')
                add_scalebar(ax, (rows, cols), space_scale=SPACE_SCALE, bar_len=500)

                ax.set_xlim([0, cols])
                ax.set_ylim([rows, 0])
                ax.set_xticks([])
                ax.set_yticks([])

                for suffix in ('.png', '.svg'):
                    out_plotfile = out_plotfile.parent / f'{out_plotfile.stem}{suffix}'
                    style.show(out_plotfile, close=False)
                plt.close()

    def gather_final_segmentations(self):
        """ Collect the final segmentation data and convert them to stats """

        outdir = self.out_roi_frames
        outdir.mkdir(parents=True, exist_ok=True)

        df = {
            'Tile': [],
            'Timepoint': [],
            'CenterX': [],
            'CenterY': [],
            'Area': [],
            'Perimeter': [],
            'Class': [],
            'MinRadius': [],
            'MeanRadius': [],
            'MaxRadius': [],
            'RadiusRatio': [],
            'Circularity': [],
            'SemiMajorAxis': [],
            'SemiMinorAxis': [],
            'EllipseAxisRatio': [],
        }

        for tiledir in sorted(outdir.iterdir()):
            if not tiledir.is_dir():
                continue

            tileno = int(tiledir.name[1:])

            # Bad samples (doublets)
            if tileno in (4, 14):
                continue

            out_contourdir = outdir / f's{tileno:02d}' / 'contours'
            out_contourdir.mkdir(parents=True, exist_ok=True)

            if not out_contourdir.is_dir():
                continue
            for timepoint_file in sorted(out_contourdir.iterdir()):
                if not timepoint_file.is_file():
                    continue
                if not timepoint_file.name.startswith('coords_s'):
                    continue
                if timepoint_file.suffix != '.csv':
                    continue
                timepoint_name = timepoint_file.stem[len('coords_s'):]
                tile, timepoint = timepoint_name.split('s')
                assert int(tile) == tileno
                timepoint = int(timepoint)

                print(f'Analyzing s{tileno:02d}t{timepoint:03d}')

                segments = read_contour_file(timepoint_file)

                largest_coords = None
                largest_area = -1

                print(f'Got {len(segments)} tracks')
                for key, coords in segments.items():
                    area = area_of_polygon(coords)
                    print(f'{key} with {coords.shape[0]} points has area {area}')
                    if area > largest_area:
                        largest_area = area
                        largest_coords = coords
                assert largest_coords is not None

                # Convert from pixels to um
                largest_coords = largest_coords * SPACE_SCALE

                # Extract stats
                area = area_of_polygon(largest_coords)
                perimeter = perimeter_of_polygon(largest_coords)

                cx, cy = center_of_polygon(largest_coords)

                radius = np.sqrt((largest_coords[:, 0]-cx)**2 + (largest_coords[:, 1]-cy)**2)
                min_radius = np.min(radius)
                max_radius = np.max(radius)
                mean_radius = np.mean(radius)

                radius_ratio = max_radius / (min_radius + 1e-5)

                circularity = 4 * np.pi * area / perimeter**2

                # Fit an ellipse
                # params are cx, cy, a, b, theta
                model = EllipseModel()
                model.estimate(largest_coords)
                semi_major = model.params[2]
                semi_minor = model.params[3]

                if semi_minor > semi_major:
                    semi_minor, semi_major = semi_major, semi_minor

                ellipse_axis_ratio = semi_major / (semi_minor + 1e-5)

                # Split into classes
                if radius_ratio < 2:
                    worm_class = 'non-elongating'
                elif radius_ratio < 3:
                    worm_class = 'partially elongating'
                else:
                    worm_class = 'elongating'

                df['Tile'].append(tileno)
                df['Timepoint'].append(timepoint*TIME_SCALE)
                df['CenterX'].append(cx)
                df['CenterY'].append(cy)
                df['Area'].append(area)
                df['Perimeter'].append(perimeter)
                df['MinRadius'].append(min_radius)
                df['MaxRadius'].append(max_radius)
                df['MeanRadius'].append(mean_radius)
                df['RadiusRatio'].append(radius_ratio)
                df['Circularity'].append(circularity)
                df['SemiMajorAxis'].append(semi_major)
                df['SemiMinorAxis'].append(semi_minor)
                df['EllipseAxisRatio'].append(ellipse_axis_ratio)
                df['Class'].append(worm_class)

        # Bank the final data frame
        df = pd.DataFrame(df)
        df = df.sort_values(['Tile', 'Timepoint'])
        df.to_excel(outdir / 'worm_summary_data.xlsx', index=False)

    def find_delta_rois(self, exponent: float = 2.0,
                        fg_sigma: float = 1.0,
                        bg_sigma: float = 30.0):
        """ For each tile, use the delta image to segment the aggregate """

        rows = cols = None

        outdir = self.out_roi_frames
        if self.overwrite and outdir.is_dir():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        cur_frames = {}
        delta_frames = {}

        tiles, timepoints = None, None

        for i, (filename, metadata, frame) in enumerate(self.czi.iter_images()):
            tileno = metadata['tile']
            timepoint = metadata['timepoint']

            if tileno not in self.tile_numbers:
                print(f'Skipping delta estimation for tile {tileno}...')
                continue

            if tiles is None and timepoints is None:
                tiles, timepoints = self.czi.tile_shape[:2]

            if rows is None and cols is None:
                rows, cols = frame.shape[:2]
            else:
                assert (rows, cols) == frame.shape[:2]

            # Background subtract the raw frame and then sharpen the edges
            fg_frame = gaussian(frame, fg_sigma)
            bg_frame = gaussian(frame, bg_sigma)
            cur_frame = np.abs(fg_frame - bg_frame)**exponent
            cur_frame = cur_frame + canny(cur_frame)*0.5

            print(f'Segmenting {filename.name}: tile {tileno} timepoint {timepoint}')
            key1 = (tileno, timepoint, timepoint+1)
            key2 = (tileno, timepoint-1, timepoint)
            if timepoint == 1:
                cur_frames.setdefault(key1, [None, None])
                cur_frames[key1][0] = cur_frame
            elif timepoint == timepoints:
                cur_frames.setdefault(key2, [None, None])
                cur_frames[key2][1] = cur_frame
            else:
                cur_frames.setdefault(key1, [None, None])
                cur_frames[key1][0] = cur_frame
                cur_frames.setdefault(key2, [None, None])
                cur_frames[key2][1] = cur_frame

            # Now try and calculate a delta
            left = cur_frames.get(key1)
            if left is not None:
                l0, l1 = left
                if l0 is not None and l1 is not None:
                    delta = delta_frames.setdefault(tileno, np.zeros((rows, cols)))
                    delta += np.abs(l0 - l1)
                    cur_frames[key1] = None
            right = cur_frames.get(key2)
            if right is not None:
                r0, r1 = right
                if r0 is not None and r1 is not None:
                    delta = delta_frames.setdefault(tileno, np.zeros((rows, cols)))
                    delta += np.abs(r0 - r1)
                    cur_frames[key2] = None

        for tileno, delta in delta_frames.items():
            if tileno not in self.tile_numbers:
                print(f'Skipping delta composite for tile {tileno}...')
                continue

            outfile = outdir / f'delta_s{tileno:02d}.tif'
            print(f'Tile {tileno:02d}: {np.min(delta)} to {np.max(delta)}')

            dmin, dmax = np.percentile(delta, [2, 98])
            drange = dmax - dmin

            if np.abs(drange) < 1e-5:
                fix_delta = np.zeros_like(delta, dtype=np.uint8)
            else:
                fix_delta = np.round((delta-dmin)/drange*255)
                fix_delta[fix_delta < 0] = 0
                fix_delta[fix_delta > 255] = 255
                fix_delta = fix_delta.astype(np.uint8)

            fix_delta = np.stack([fix_delta, fix_delta, fix_delta], axis=2)

            img = Image.fromarray(fix_delta)
            img.save(str(outfile))

    def subset_tiles_rois(self):
        """ Subset the tiles with manual ROIs """
        bbox_rois = BBOX_ROIS

        rows = cols = None

        outdir = self.out_subset_frames
        if self.overwrite and outdir.is_dir():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        for _, (filename, metadata, frame) in enumerate(self.czi.iter_images()):
            tileno = metadata['tile']
            tile_name = metadata['tile_name']
            if tileno not in bbox_rois:
                continue
            if tileno not in self.tile_numbers:
                continue

            # timepoint = metadata['timepoint']
            # if timepoint not in (1, 100, 145):
            #     continue

            # Force the image to be the right shape
            if rows is None and cols is None:
                rows, cols = frame.shape[:2]
                self.rows = rows
                self.cols = cols
            else:
                if frame.shape[:2] != (rows, cols):
                    err = f'Got shape {frame.shape[:2]}, expected {rows, cols}'
                    print(err)
                    raise ValueError(err)
            if frame.ndim == 3:
                frame = np.min(frame, axis=2)
            assert frame.ndim == 2

            # Subset multiple worms per image
            all_bboxes = bbox_rois[tileno]
            if isinstance(all_bboxes, tuple):
                all_bboxes = [all_bboxes]
            for i, bbox in enumerate(all_bboxes):
                assert len(bbox) == 4

                print(f'Subsetting {filename.name}: {bbox}')

                cy, cx, width, height = bbox
                cy = cy + height/2.0
                cx = cx + width/2.0
                cx_minx, cx_maxx, cy_miny, cy_maxy = self.calc_bbox(cx, cy)

                sub_frame = frame[cx_minx:cx_maxx, cy_miny:cy_maxy]

                # Histogram equalize
                eq_frame = exposure.equalize_adapthist(sub_frame)

                eq_frame = np.round(eq_frame*255)
                eq_frame[eq_frame < 0] = 0
                eq_frame[eq_frame > 255] = 255
                eq_frame = eq_frame.astype(np.uint8)

                outfile = outdir / f's{tileno:02d}-{tile_name}-{i+1:d}' / filename.name
                outfile.parent.mkdir(parents=True, exist_ok=True)

                print(f'Writing {outfile}')
                img = Image.fromarray(eq_frame)
                img.save(str(outfile))

    def find_rois(self):
        """ For each image, remove the background field to get the aggregate """

        rows = cols = None

        outdir = self.out_roi_frames
        if outdir.is_dir():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True)

        roi_coords = {
            'Filename': [],
            'Tile': [],
            'CenterX': [],
            'CenterY': [],
            'ROIMinX': [],
            'ROIMaxX': [],
            'ROIMinY': [],
            'ROIMaxY': [],
            'BBoxMinX': [],
            'BBoxMaxX': [],
            'BBoxMinY': [],
            'BBoxMaxY': [],
        }

        for i, (filename, metadata, frame) in enumerate(self.czi.iter_images()):
            if metadata['timepoint'] > 1:
                break
            if metadata['tile'] not in self.tile_numbers:
                continue

            print(f'Segmenting {filename.name}')

            if frame.ndim == 3:
                frame = frame[:, :, 0]
            if rows is None or cols is None:
                rows, cols = frame.shape
                self.rows = rows
                self.cols = cols
            else:
                assert frame.shape == (rows, cols)
            frame = frame.astype(np.float64) / BG_MAX_VALUE
            frame = filters.gaussian(frame, FOREGROUND_SMOOTHING)

            # Get the seed with a pair of thresholds
            frame_mask = np.logical_or(frame < BG_MIN_VALUE/BG_MAX_VALUE,
                                       frame > 0.03)

            frame_mask = morphology.binary_dilation(frame_mask, selem=np.ones((9, 9)))

            label_image = label(frame_mask, connectivity=2, background=1)

            eccentricity = {}
            centroids = {}
            props = []
            for prop in regionprops(label_image):
                is_good = prop.area > 2000 and prop.area < self.row_width*self.col_width
                is_good = is_good and prop.solidity > 0.4
                is_good = is_good and prop.eccentricity > 0.1
                if not is_good:
                    label_image[label_image == prop.label] = 0
                else:
                    props.append(prop)
                    eccentricity[prop.label] = prop.eccentricity
                    centroids[prop.label] = (int(round(prop.centroid[0])), int(round(prop.centroid[1])))

            # Pick the most circular remaining sample
            best_label = [label_idx for label_idx in sorted(eccentricity, key=lambda x: eccentricity[x])]
            if len(best_label) < 1:
                continue

            best_label = best_label[0]
            best_label_image = label_image == best_label

            # Flood fill to try and expand the segmentation
            frame_edges = filters.sobel(frame, mask=np.logical_and(frame > BG_MIN_VALUE/BG_MAX_VALUE,
                                                                   frame < 0.1))
            seg = segmentation.flood(frame_edges, centroids[best_label], tolerance=0.01)

            # Merge the flood fill with our ROI of choice
            seg = ~morphology.binary_erosion(seg, selem=np.ones((3, 3)))
            seg = np.logical_or(seg, best_label_image)

            seg = morphology.binary_dilation(seg, selem=np.ones((21, 21)))
            seg = morphology.remove_small_holes(seg, area_threshold=12000)
            seg = morphology.binary_erosion(seg, selem=np.ones((21, 21)))

            seg_label = label(seg, connectivity=2, background=0)
            seg = seg_label == np.max(seg_label[best_label_image])

            prop = list(regionprops(seg.astype(np.int)))[0]
            minx, miny, maxx, maxy = prop.bbox

            # Make a large ROI around the region
            cx, cy = prop.centroid[0], prop.centroid[1]

            cx_minx, cx_maxx, cy_miny, cy_maxy = self.calc_bbox(cx, cy)

            # Stash the coordinates for the next stage
            roi_coords['CenterX'].append(cx)
            roi_coords['CenterY'].append(cy)

            roi_coords['ROIMinX'].append(minx)
            roi_coords['ROIMaxX'].append(maxx)
            roi_coords['ROIMinY'].append(miny)
            roi_coords['ROIMaxY'].append(maxy)

            roi_coords['BBoxMinX'].append(cx_minx)
            roi_coords['BBoxMaxX'].append(cx_maxx)
            roi_coords['BBoxMinY'].append(cy_miny)
            roi_coords['BBoxMaxY'].append(cy_maxy)

            roi_coords['Tile'].append(metadata['tile'])
            roi_coords['Filename'].append(filename.name)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            ax1.imshow(frame[cx_minx:cx_maxx, cy_miny:cy_maxy])
            ax2.imshow(best_label_image[cx_minx:cx_maxx, cy_miny:cy_maxy])
            ax3.imshow(frame_edges[cx_minx:cx_maxx, cy_miny:cy_maxy])
            ax4.imshow(seg[cx_minx:cx_maxx, cy_miny:cy_maxy])

            outfile = outdir / (filename.stem + '.png')
            fig.savefig(str(outfile))
            plt.close()

        roi_coords = pd.DataFrame(roi_coords)
        roi_coords.to_excel(str(self.out_roi_coords))

    def subset_tiles(self):
        """ Actually subset the tiles """

        rows = cols = None
        roi_coords = pd.read_excel(str(self.out_subset_coords), index_col=0)

        for i, (filename, metadata, frame) in enumerate(self.czi.iter_images()):
            if metadata['tile'] not in self.tile_numbers:
                continue

            print(f'Subsetting {filename.name}')

            if frame.ndim == 3:
                frame = frame[:, :, 0]
            if rows is None or cols is None:
                rows, cols = frame.shape
                self.rows = rows
                self.cols = cols
            else:
                assert frame.shape == (rows, cols)
            # frame = frame.astype(np.float64) / BG_MAX_VALUE

            row = roi_coords[roi_coords['Tile'] == metadata['tile']]
            cx = row['CenterX']
            cy = row['CenterY']

            cx_minx, cx_maxx, cy_miny, cy_maxy = self.calc_bbox(cx, cy)
            frame = frame[cx_minx:cx_maxx, cy_miny:cy_maxy]

            eq_frame = exposure.equalize_adapthist(frame)

            # frame_mask = np.logical_and(frame > BG_MIN_VALUE, frame < 0.1*BG_MAX_VALUE)
            frame_mask = frame > BG_MIN_VALUE
            frame_edges = filters.sobel(eq_frame, mask=frame_mask)

            eq_frame[~frame_mask] = 0
            frame_edges[~frame_mask] = 0

            fg_scale = np.max(eq_frame[frame_mask]) - np.min(eq_frame[frame_mask])
            bg_scale = np.max(frame_edges[frame_mask]) - np.min(eq_frame[frame_mask])

            enh_frame = eq_frame + frame_edges * fg_scale / bg_scale * 0.5

            outdir = self.outdir / 'segments' / 's{:02d}'.format(metadata['tile'])
            outdir.mkdir(parents=True, exist_ok=True)

            outfile = outdir / (filename.stem + '.tif')

            enh_frame = np.round(enh_frame * 500)
            enh_frame[enh_frame < 0] = 0
            enh_frame[enh_frame > 255] = 255
            enh_frame = enh_frame.astype(np.uint8)

            enh_img = Image.fromarray(enh_frame)
            enh_img.save(str(outfile))

    def segment_tile_foreground(self, tile: int):
        """ Now segment the individual tile

        :param int tile:
            The tile number to segment
        """
        print(f'Segmenting foreground for s{tile:02d}')

        tiledir = self.outdir / 'segments' / f's{tile:02d}'

        if not tiledir.is_dir():
            raise OSError(f'Cannot find tile {tile} directory: {tiledir}')

        prev_img = None
        curr_img = None
        mask_imgs = []

        for i, image_file in enumerate(sorted(tiledir.iterdir())):
            if image_file.suffix not in ('.tif', ):
                continue

            curr_img = np.asarray(Image.open(str(image_file)))
            curr_img = curr_img.astype(np.float) / 255

            if prev_img is None:
                prev_img = curr_img
                continue

            delta_img = np.abs(curr_img - prev_img)
            mask_imgs.append(delta_img)

            prev_img = curr_img

        erosion_size = np.ones((13, 13))

        std_threshold = self.std_thresholds.get(tile, STD_DEFAULT_THRESHOLD)

        all_delta_std = np.std(mask_imgs, axis=0)
        all_delta_mask = all_delta_std > std_threshold
        all_delta_mask = morphology.binary_erosion(all_delta_mask, selem=erosion_size)

        all_delta_labels = label(all_delta_mask, connectivity=2, background=0)
        areas = {}
        for prop in regionprops(all_delta_labels):
            areas[prop.label] = prop.area
        best_label = [label_idx for label_idx in sorted(areas, key=lambda x: areas[x])][-1]
        all_delta_mask = morphology.binary_dilation(all_delta_labels == best_label, selem=erosion_size)

        all_delta_filled = ndi.binary_fill_holes(all_delta_mask)
        all_delta_center = np.logical_xor(all_delta_filled, all_delta_mask)
        # all_delta_center = morphology.binary_erosion(all_delta_filled, selem=disk(100))

        outdir = self.outdir / 'background_foreground' / f's{tile:02d}'
        if outdir.is_dir():
            shutil.rmtree(str(outdir))
        outdir.mkdir(parents=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.imshow(all_delta_std)
        ax2.imshow(all_delta_labels)
        ax3.imshow(all_delta_filled)
        ax4.imshow(all_delta_center)
        fig.savefig(str(outdir / f'bg_plot_s{tile:02d}.png'), transparent=True)
        plt.close()

        outfile = outdir / f'bg_mask_s{tile:02d}.tif'
        img = Image.fromarray((~all_delta_filled).astype(np.uint8) * 255)
        img.save(str(outfile))

        outfile = outdir / f'fg_mask_s{tile:02d}.tif'
        img = Image.fromarray((all_delta_center).astype(np.uint8) * 255)
        img.save(str(outfile))

    def segment_all_tile_foregrounds(self, tiles: Optional[List[int]] = None):
        """ Segment all tiles

        :param list tiles:
            The tiles to segment the foreground of
        """

        if tiles is None:
            tiles = self.tile_numbers
        for tile in tiles:
            self.segment_tile_foreground(tile)

    def segment_all_frames(self, tiles: Optional[List[int]] = None):
        """ Segment all tiles

        :param list tiles:
            The tiles to segment the frames of
        """

        if tiles is None:
            tiles = self.tile_numbers

        errors = []
        for tile in tiles:
            try:
                self.segment_frame(tile)
            except Exception:
                print(f'Error segmenting {tile}')
                traceback.print_exc()
                errors.append(tile)

        if len(errors) > 0:
            print(f'Failed to segment {len(errors)} tiles:')
            for error in errors:
                print(f'* {error}')

    def segment_frame(self, tile: int):
        """ Segment a frame, using the background foreground masks

        :param int tile:
            The tile number to segment
        """

        segdir = self.outdir / 'segments' / f's{tile:02d}'
        maskdir = self.outdir / 'background_foreground' / f's{tile:02d}'

        if not segdir.is_dir():
            raise OSError(f'Cannot find tile {tile} directory: {segdir}')

        if not maskdir.is_dir():
            raise OSError(f'Cannot find mask {tile} directory: {maskdir}')

        # Load the foreground and background segments
        fg_file = maskdir / f'fg_mask_s{tile:02d}.tif'
        fg_mask = np.asarray(Image.open(str(fg_file))) > 1
        fg_mask = morphology.binary_erosion(fg_mask, selem=np.ones((15, 15)))
        fg_mask = morphology.remove_small_objects(fg_mask, 200)

        bg_file = maskdir / f'bg_mask_s{tile:02d}.tif'
        bg_mask = np.asarray(Image.open(str(bg_file))) > 1
        bg_mask = morphology.binary_erosion(bg_mask, selem=np.ones((100, 100)))
        bg_mask = morphology.remove_small_objects(bg_mask, 200)

        # Initialize the labels
        label_mask = np.zeros_like(bg_mask, dtype=np.int)
        label_mask[fg_mask] = 1
        label_mask[bg_mask] = 2

        rows, cols = label_mask.shape

        outdir = self.outdir / 'contours' / f's{tile:02d}'
        if outdir.is_dir():
            shutil.rmtree(str(outdir))
        outdir.mkdir(parents=True)

        outfile = outdir / 'segments.json'

        with outfile.open('wt') as fp:
            for i, image_file in enumerate(sorted(segdir.iterdir())):
                if image_file.suffix not in ('.tif', ):
                    continue

                print(f'Segmenting {image_file}')

                curr_img = np.asarray(Image.open(str(image_file)))
                curr_img = curr_img.astype(np.float) / 255
                delta_img = curr_img - filters.gaussian(curr_img, 30)
                curr_img += delta_img*0.1

                seg = segmentation.random_walker(curr_img, label_mask, beta=DIFFUSION_BETA)

                fg_mask = seg == 1
                fg_mask = morphology.remove_small_holes(fg_mask, 200)
                fg_mask = morphology.binary_erosion(fg_mask)
                fg_mask = morphology.binary_erosion(fg_mask)
                contours = contours_from_mask(fg_mask, tolerance=0.5)

                if len(contours) < 1:
                    raise ValueError('Segmentation failed')

                # Find the biggest polygon
                if len(contours) > 1:
                    best_index = -1
                    best_area = 0
                    for idx, contour in enumerate(contours):
                        area = area_of_polygon(contour)
                        if area > best_area:
                            best_area = area
                            best_index = idx
                else:
                    best_index = 0

                # Pull out the main contour and centroid
                contour = contours[best_index]
                cx, cy = center_of_polygon(contour)
                area = area_of_polygon(contour)

                # Re-seed the labels for the next round by shrinking and inflating
                fg_contour = np.stack([
                    (contour[:, 0] - cx) * 0.5 + cx,
                    (contour[:, 1] - cy) * 0.5 + cy,
                ], axis=1)

                bg_contour = np.stack([
                    (contour[:, 0] - cx) * 2.0 + cx,
                    (contour[:, 1] - cy) * 2.0 + cy,
                ], axis=1)

                with set_plot_style('poster') as style:
                    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                    ax.imshow(curr_img, cmap='gray')

                    ax.plot(contour[:, 0], contour[:, 1], '-r', linewidth=2)

                    #ax.plot(fg_contour[:, 0], fg_contour[:, 1], '-g', linewidth=2)
                    #ax.plot(bg_contour[:, 0], bg_contour[:, 1], '--g', linewidth=2)

                    outfile = outdir / (image_file.stem + '.tif')
                    ax.set_xlim([0, cols])
                    ax.set_ylim([rows, 0])
                    style.set_axis_off(ax)
                    style.show(outfile, transparent=True)

                data = {
                    'outfile': outfile.name,
                    'contour': contour,
                    'area': area,
                    'center_x': cx,
                    'center_y': cy,
                }
                fp.write(json.dumps(to_json_types(data)) + '\n')

                # And generate the masks for the next frame
                fg_mask = mask_from_contours([fg_contour], curr_img.shape)
                bg_mask = ~mask_from_contours([bg_contour], curr_img.shape)

                label_mask[...] = 0
                label_mask[fg_mask] = 1
                label_mask[bg_mask] = 2

# Functions


def background_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ Inpaint the values inside a mask

    :param ndarray image:
        The image to inpaint
    :param ndarray mask:
        The mask to use for inpainting
    :returns:
        A new image, with all masked areas interpolated over
    """
    rows, cols = image.shape[:2]

    x = np.arange(rows)
    y = np.arange(cols)

    xx, yy = np.meshgrid(x, y, indexing='ij')

    selem = morphology.disk(radius=5)

    labels = ndi.label(mask)[0]
    background = image.copy()
    for i in np.unique(labels):
        if i < 1:
            continue
        label_mask = labels == i
        label_ring = morphology.binary_dilation(label_mask, selem=selem)
        label_ring = np.logical_and(label_ring, ~label_mask)

        # Extract the points inside and around the mask
        points_in = np.stack([xx[label_ring], yy[label_ring]], axis=1)
        vals_in = image[label_ring]
        points_out = np.stack([xx[label_mask], yy[label_mask]], axis=1)

        # Interpolate inside the mask
        vals_out = griddata(points_in, vals_in, points_out, method='linear')
        background[label_mask] = vals_out

    return background


def trunc_quad(xdata: Tuple[np.ndarray],
               a: float, b: float, c: float, d: float, e: float, f: float,
               r: float) -> np.ndarray:
    """ Truncated quadratic function

    :param ndarray xdata:
        The x, y coordinates for the points
    :param float r:
        Clip all values greater than this radius to 0.0
    :returns:
        The model truncated quadratic
    """
    x, y = xdata

    # Normal quadratic
    ydata = a*x**2 + b*y**2 + c*x + d*y + e*x*y + f

    # If we have a real root, remove the radius
    root = 4*a*b - e**2
    if root > 0:
        # Solve for the peak
        cx = -(2*b*c - d*e)/root
        cy = -(2*a*d - c*e)/root
        rr = (x - cx)**2 + (y - cy)**2
        ydata[rr > r**2] = 0
    return ydata


def read_contour_file(infile: pathlib.Path) -> Dict[str, np.ndarray]:
    """ Read in the contour file

    :param Path infile:
        Input contour coordinate file
    :returns:
        A dictionary of trackname: 2D coordinates
    """
    segments = {}

    with infile.open('rt') as fp:
        trackname = None
        xcoords = None
        ycoords = None
        for rec in fp:
            rec = rec.strip()
            if rec == '':
                continue
            if rec.startswith('Track'):
                if all([c is not None for c in (trackname, xcoords, ycoords)]):
                    assert trackname not in segments
                    segments[trackname] = np.stack([xcoords, ycoords], axis=1)
                    trackname = None
                    xcoords = None
                    ycoords = None
                trackname = rec
            elif rec.startswith('XCoords,'):
                assert xcoords is None
                xcoords = np.array([float(x.strip()) for x in rec.split(',')[1:]])
            elif rec.startswith('YCoords,'):
                assert ycoords is None
                ycoords = np.array([float(y.strip()) for y in rec.split(',')[1:]])

    if all([c is not None for c in (trackname, xcoords, ycoords)]):
        segments[trackname] = np.stack([xcoords, ycoords], axis=1)
    return segments
