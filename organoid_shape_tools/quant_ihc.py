""" Immunohistochemistry quantification tools

Functions:

* :py:func:`load_single_channel`: Load the best channel from an image
* :py:func:`group_infiles`: Find the DAPI and PH3 images for each input for each channel
* :py:func:`plot_lines_fit`: Plot the radial distribution of cells with several fit lines

API Documentation
-----------------

"""

# Imports
import re
import pathlib
from typing import Optional

# 3rd party
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

# our own imports
from .utils import load_image
from .plotting import set_plot_style, add_lineplot

# Constants
PLOT_STYLE = 'figure'

# Extract group info from the file name
reFILENAME = re.compile(r'''^
    daj_(diff[0-9]_)?(?P<day>d[0-9]+)_lbc_2hredu_ph3_[0-9]-image.export-[0-9]+_(?P<channel>c[0-9]+)
$''', re.VERBOSE | re.IGNORECASE)

# Functions


def plot_lines_fit(data: pd.DataFrame, x: str, y: str, plotfile: pathlib.Path,
                   hue: Optional[str] = None,
                   max_extent_fit: float = 0.9,
                   ylabel: str = '',
                   plot_style: str = PLOT_STYLE):
    """ Plot the data lines with fit curves on top """

    xs = data[x].values
    ys = data[y].values

    mask = np.logical_and(xs > -max_extent_fit, xs < max_extent_fit)
    xs = xs[mask]
    ys = ys[mask]
    mask_data = data[mask]

    formula_linear = f'{y} ~ {x} + 1'

    print(f'Linear model: "{formula_linear}"')
    model_linear = smf.ols(formula=formula_linear, data=data)
    res = model_linear.fit()
    print(res.summary())
    print('\n')

    xfit = np.linspace(-1, 1, 100)
    yfit_linear = xfit * res.params[1] + res.params[0]

    formula_quad = f'{y} ~ np.power({x}, 2) + {x} + 1'

    print(f'Quad model: "{formula_quad}"')
    model_quad = smf.ols(formula=formula_quad, data=data)
    res = model_quad.fit()
    print(res.summary())
    print('\n')

    # NOTE THAT THE PARAMETERS COME BACK IN A STUPID ORDER!!
    yfit_quad = xfit**2 * res.params[1] + xfit * res.params[2] + res.params[0]

    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        add_lineplot(ax, mask_data, x=x, y=y, yscale=100, hue=hue)

        ax.plot(xfit, yfit_linear*100, '-', color='gray', linewidth=5, label='linear model')
        ax.plot(xfit, yfit_quad*100, '-', color='cyan', linewidth=5, label='quadratic model')

        ax.set_xlabel('Position along semi major axis')
        ax.set_ylabel(ylabel)

        ax.legend()

        style.show(outfile=plotfile, suffixes=('.svg', '.png', '.pdf'))


def group_infiles(rootdir: pathlib.Path,
                  dapi_channel: str = '_c2',
                  ph3_channel: str = '_c1'):
    """ Group the input files and split them into DAPI/ph3 classes

    :param Path rootdir:
        Directory to process
    """

    dapi_suffix = f'{dapi_channel}.tif'
    ph3_suffix = f'{ph3_channel}.tif'

    for subdir in sorted(rootdir.iterdir()):
        if not subdir.is_dir():
            continue
        if not subdir.name.startswith('DAJ_'):
            continue
        dapi_file = None
        ph3_file = None

        for infile in sorted(subdir.iterdir()):
            if not infile.is_file():
                continue
            if dapi_file is not None and ph3_file is not None:
                break
            if infile.name.endswith(dapi_suffix):
                assert dapi_file is None
                dapi_file = infile
                continue
            if infile.name.endswith(ph3_suffix):
                assert ph3_file is None
                ph3_file = infile
                continue

        if dapi_file is None or ph3_file is None:
            raise OSError(f'Failed to pair DAPI and PH3 under {subdir}')

        instem = dapi_file.name[:-len(dapi_suffix)].strip('_').strip('-').strip()

        match = reFILENAME.match(dapi_file.stem)
        if not match:
            raise ValueError(f'Cannot parse {dapi_file.name}')

        day = match.group('day')

        yield dapi_file, ph3_file, instem, day


def load_single_channel(infile: pathlib.Path) -> np.ndarray:
    """ Load the best single channel from an image

    :param Path infile:
        Input file to load
    :returns:
        The maximum intensity color from each channel
    """
    print(f'Loading {infile}...')

    img = load_image(infile, ctype='color') / 255.0
    if img.shape[2] == 4:
        img = img[:, :, :3] * img[:, :, 3:4]
    assert img.shape[2] == 3
    img_flat = img.reshape((-1, 3))

    chan_min = np.min(img_flat, axis=0)
    chan_max = np.max(img_flat, axis=0)

    chan_rng = chan_max - chan_min

    out_img = np.zeros(img.shape[:2])
    if np.sum(chan_rng) < 1e-5:
        return out_img

    # Make channel weights sum to 1.0
    chan_rng /= np.sum(chan_rng)

    for i in range(3):
        if chan_rng[i] < 1e-5:
            continue
        out_img += (img[:, :, i] - chan_min[i]) / chan_rng[i]
    return out_img
