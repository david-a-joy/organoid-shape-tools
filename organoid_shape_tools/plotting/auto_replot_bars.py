""" Replotting system for regenerating publication-quality plots

Main function:

* :py:func:`replot_bars`: Replot things with bars/boxes/lines/etc

For the command-line version, see ``replot_bars.py``
"""

# Imports
import re
import pathlib
from typing import Optional, Tuple, List, Dict
import inspect

# 3rd party
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# Our own imports
from . import (
    set_plot_style, SplitAxes, colorwheel, add_single_boxplot, add_single_barplot,
)

# Constants

PLOT_STYLE = 'light'
PLOT_TYPE = 'bars'
ORIENT = 'vertical'
PALETTE = 'wheel_greywhite'

SUFFIXES = ('.png', '.svg')

FIGURE_SCALE: float = 10  # Height of the panel in inches

# Filename, xcolumn, ycolumn, hue_column, palette
DEFAULT_COLUMNS = {}

# Names for the neural nets
DETECTOR_NAMES = {
    'countception': 'Count-ception',
    'count_ception': 'Count-ception',
    'countception-r003-n50000': 'Count-ception',
    'fcrn_a_wide-r003-n75000': 'FCRN-A',
    'fcrn_a_wide': 'FCRN-A',
    'fcrn_a': 'FCRN-A',
    'fcrn_b_wide-r003-n75000': 'FCRN-B',
    'fcrn_b_wide': 'FCRN-B',
    'fcrn-b': 'FCRN-B',
    'residual_unet-r004-n25000': 'Residual U-net',
    'residual_unet': 'Residual U-net',
    'res_u_net': 'Residual U-net',
    'unet-r001-n50000': 'U-net',
    'unet': 'U-net',
    'u_net': 'U-net',
    'composite-d3-final': 'Net Ensemble',
    'composite': 'Net Ensemble',
}
DETECTOR_NAMES = {k.lower().replace(' ', '_').replace('-', '_'): v
                  for k, v in DETECTOR_NAMES.items()}
RADIUS_NAMES = {
    '0.0-0.6': 'Center',
    '0.6-0.8': 'Middle',
    '0.8-1.0': 'Edge',
}


def grouper(n, iterable):
    """ Group a list of objects into tuples of those objects """
    args = [iter(iterable)] * n
    return zip(*args)


def parse_bbox(bbox: Optional[str]) -> Optional[Tuple[Tuple[float]]]:
    """ Parse a bounding box string

    :param str bbox:
        If not None, a string of comma separated bounds of the form: start,end
    :returns:
        A tuple of tuples, one for each (start, end) pair, in order
    """
    if bbox is None:
        return None
    # Parse the string
    bbox = [float(b.strip()) for b in bbox.split(',')]
    if len(bbox) % 2 != 0:
        raise ValueError('Expected sets of (start, end) bounds, got: {}'.format(bbox))
    return tuple(grouper(2, bbox))


def parse_order(order: Optional[str]) -> Optional[List[str]]:
    """ Parse order strings (comma separated) into a list

    :param str order:
        If not None, the names of xcolumns, in order
    :returns:
        A list containing those columns in order
    """
    if order is None:
        return None
    return [o.strip() for o in order.split(',')]


# Classes


class StrInt(object):
    """ Sort ints before strings, in numerical order """

    def __init__(self, val: str):
        try:
            val = int(val)
            is_num = True
        except (TypeError, ValueError):
            is_num = False
        self.is_num = is_num
        self.val = val

    def __lt__(self, other):
        if self.is_num:
            if other.is_num:
                return self.val < other.val
            return True
        else:
            if other.is_num:
                return False
            return self.val < other.val


class BarPlotter(object):
    """ Load the plot data and regenerate plots """

    def __init__(self,
                 datafile: pathlib.Path,
                 xcolumn: str,
                 ycolumn: str,
                 hue_column: Optional[str] = None,
                 outfile: Optional[pathlib.Path] = None,
                 plot_style: str = PLOT_STYLE,
                 plot_type: str = PLOT_TYPE,
                 palette: str = PALETTE,
                 orient: str = ORIENT,
                 xlimits: Optional[Tuple[Tuple[float]]] = None,
                 ylimits: Optional[Tuple[Tuple[float]]] = None,
                 scale: Optional[float] = None,
                 xlabel: Optional[str] = None,
                 xticklabel_rotation: Optional[float] = None,
                 ylabel: Optional[str] = None,
                 order: Optional[List[str]] = None,
                 hue_order: Optional[List[str]] = None,
                 figure_scale: float = FIGURE_SCALE,
                 yscale_type: str = 'linear',
                 bar_baseline: Optional[float] = None,
                 bar_errorbar_side: Optional[str] = None,
                 split_regex: Optional[str] = None,
                 bar_width: Optional[float] = None,
                 bar_padding: Optional[float] = None,
                 bar_cat_padding: Optional[float] = None,
                 plot_individual_samples: bool = False,
                 sample_marker_color: Optional[str] = None,
                 sample_marker_size: Optional[float] = None,
                 significance_barcolor: Optional[str] = None,
                 plot_num_points: bool = False,
                 tight_layout: bool = False,
                 suffixes: Tuple[str] = None):
        # Paths
        self.datafile = datafile
        self.outfile = outfile

        # Column selectors
        self.xcolumn = xcolumn
        self.ycolumn = ycolumn
        self.hue_column = hue_column
        self.split_regex = split_regex

        # Styles
        self.plot_type = plot_type
        self.plot_style = plot_style
        self.palette = palette
        orient = orient.lower()[0]
        if orient not in ('v', 'h'):
            raise ValueError('Orient must be one of [h]orizontal or [v]ertical')
        self.orient = orient
        if suffixes is None:
            suffixes = SUFFIXES
        if isinstance(suffixes, str):
            suffixes = (suffixes, )
        self.suffixes = tuple(str(s) for s in suffixes)

        self._color_table = None

        # Bar layout
        self.figure_y = figure_scale
        self.figure_x = None
        self.tight_layout = tight_layout

        self.bar_width = 1.0 if bar_width is None else bar_width
        self.bar_padding = 0.15 if bar_padding is None else bar_padding
        self.bar_cat_padding = 0.4 if bar_cat_padding is None else bar_cat_padding

        self.box_width = 1.0
        self.box_padding = 0.15
        self.box_cat_padding = 0.4

        # Bar styles
        self.bar_linewidth = 3
        self.bar_error_linewidth = 3
        self.bar_error_capsize = 15 * self.bar_width
        self.bar_edgecolor = {
            'dark': '#FFFFFF',
            'dark_poster': '#FFFFFF',
        }.get(self.plot_style, '#000000')
        self.bar_error_linecolor = {
            'dark': '#FFFFFF',
            'dark_poster': '#FFFFFF',
        }.get(self.plot_style, '#000000')

        # Sample plotting
        self.plot_individual_samples = plot_individual_samples
        self.marker_facecolor = sample_marker_color
        self.marker_size = sample_marker_size if sample_marker_size else 12.0*self.bar_width
        self.marker_edgecolor = {
            'dark': '#FFFFFF',
            'dark_poster': '#FFFFFF',
        }.get(self.plot_style, '#000000')

        # Text and colors for annotations
        self.num_points_fontsize = 24
        self.num_points_color = {
            'dark': '#FFFFFF',
            'dark_poster': '#FFFFFF',
        }.get(self.plot_style, '#000000')
        self.plot_num_points = plot_num_points

        if significance_barcolor is None:
            significance_barcolor = {
                'dark': '#AAAAAA',
                'dark_poster': '#AAAAAA',
            }.get(self.plot_style, '#666666')
        self.significance_barcolor = significance_barcolor
        self.significance_linewidth = 3 * self.bar_width
        self.significance_fontsize = 28

        # Derived
        self._bar_baseline = bar_baseline
        self._bar_errorbar_side = bar_errorbar_side
        self._yscale_type = yscale_type

        self._data = None
        self._distdata = None
        self._sigdata = None
        self._sampledata = None

        self._norm_xcolumn = None
        self._norm_ycolumn = None

        self._order = order
        self._hue_order = hue_order

        self._xcoords = None
        self._xtick_coords = None
        self._xticklabel_rotation = xticklabel_rotation
        self._ycoords = None
        self._yerr = None
        self._ynum = None
        self._ysig = None
        self._yscale = scale

        self._bar_hcat = None
        self._bar_color = None

        self._xlimits = xlimits
        self._ylimits = ylimits
        self._xlabel = xlabel
        self._ylabel = ylabel

    @classmethod
    def get_plot_methods(cls) -> List[str]:
        """ Get a list of ``plot_`` methods for this class """
        return [n.split('_', 1)[1]
                for n, _ in inspect.getmembers(cls, inspect.isfunction)
                if n.startswith('plot_')]

    @property
    def norm_xcolumn(self):
        """ Normalize the x-column name """
        if self._norm_xcolumn is not None:
            return self._norm_xcolumn
        self._norm_xcolumn = self.xcolumn.lower().strip().replace('_', '')
        return self._norm_xcolumn

    @property
    def norm_ycolumn(self):
        """ Normalize the y-column name """
        if self._norm_ycolumn is not None:
            return self._norm_ycolumn
        self._norm_ycolumn = self.ycolumn.lower().strip().replace('_', '')
        return self._norm_ycolumn

    @property
    def data_distfile(self):
        """ Distribution data associated with a graph """
        datadir = self.datafile.parent
        suffix = self.datafile.suffix
        stem = self.datafile.stem
        return datadir / f'{stem}_dist{suffix}'

    @property
    def data_samplefile(self):
        """ Sample data associated with a graph """
        datadir = self.datafile.parent
        suffix = self.datafile.suffix
        stem = self.datafile.stem
        return datadir / f'{stem}_samples{suffix}'

    @property
    def data_sigfile(self):
        """ Significance data associated with a graph """
        datadir = self.datafile.parent
        suffix = self.datafile.suffix
        stem = self.datafile.stem
        return datadir / f'{stem}_sig{suffix}'

    def read_data_frame(self, datafile: pathlib.Path) -> pd.DataFrame:
        """ Read in a data frame in a data type agnostic way """
        if datafile.suffix == '.csv':
            data = pd.read_csv(str(datafile))
        elif datafile.suffix == '.tsv':
            data = pd.read_csv(str(datafile), sep='\t')
        elif self.datafile.suffix in ('.xls', '.xlsx'):
            data = pd.read_excel(str(datafile))
        else:
            raise ValueError(f'Unknown data file type: {datafile}')
        return data

    def load_data(self):
        """ Load the raw data in """
        # Load the original data frame
        data = self.read_data_frame(self.datafile)
        if self.split_regex not in (None, ''):
            split_regex = self.split_regex
            if not split_regex.startswith('^'):
                split_regex = '^' + split_regex
            if not split_regex.endswith('$'):
                split_regex = split_regex + '$'
            split_regex = re.compile(split_regex, re.IGNORECASE)
            new_xcolumn = []
            new_hue_column = []
            for _, rec in data.iterrows():
                key = str(rec[self.xcolumn])
                match = split_regex.match(key)
                if not match:
                    raise ValueError(f'Failed to split record: {key}')
                new_x = None
                new_hue = None
                groupdict = match.groupdict()
                groups = match.groups()

                if new_x is None and 'x' in groupdict:
                    new_x = groupdict['x']
                if new_x is None and 'xcolumn' in groupdict:
                    new_x = groupdict['xcolumn']
                if new_x is None and len(groups) > 0:
                    new_x = groups[0]

                if new_hue is None and 'hue' in groupdict:
                    new_hue = groupdict['hue']
                if new_hue is None and 'huecolumn' in groupdict:
                    new_hue = groupdict['huecolumn']
                if new_hue is None and 'hue_column' in groupdict:
                    new_hue = groupdict['hue_column']
                if new_hue is None and len(groups) > 1:
                    new_hue = groups[1]

                # Make sure we got an x and a hue
                assert new_x is not None
                new_xcolumn.append(new_x)
                if new_hue is not None:
                    new_hue_column.append(new_hue)

            assert len(new_xcolumn) > 0
            if len(new_hue_column) == 0:
                data['NewX'] = new_xcolumn
                self._norm_xcolumn = self.norm_xcolumn
                self.xcolumn = 'NewX'
            else:
                assert len(new_hue_column) == len(new_xcolumn)
                data['NewX'] = new_xcolumn
                self._norm_xcolumn = self.norm_xcolumn
                self.xcolumn = 'NewX'
                data['NewHue'] = new_hue_column
                self.hue_column = 'NewHue'

        # Convert the categories to strings for compatibility
        data[self.xcolumn] = data[self.xcolumn].apply(str)
        if self.hue_column is not None:
            data[self.hue_column] = data[self.hue_column].apply(str)
        self._data = data

        # Load the raw sample data, if any
        if self.data_samplefile.is_file():
            sampledata = self.read_data_frame(self.data_samplefile)
            # Convert the categories to strings for compatibility
            sampledata[self.xcolumn] = sampledata[self.xcolumn].apply(str)
            if self.hue_column is not None:
                sampledata[self.hue_column] = sampledata[self.hue_column].apply(str)
            self._sampledata = sampledata

        # Load the distribution data, if any
        if self.data_distfile.is_file():
            self._distdata = self.read_data_frame(self.data_distfile)

        # Load the significance data
        if self.data_sigfile.is_file():
            sigdata = self.read_data_frame(self.data_sigfile)
            xcol1 = self.xcolumn + '1'
            xcol2 = self.xcolumn + '2'
            if xcol1 in sigdata.columns and xcol2 in sigdata.columns:
                sigdata[xcol1] = sigdata[xcol1].apply(str)
                sigdata[xcol2] = sigdata[xcol2].apply(str)
                self._sigdata = sigdata

        if self._order is None:
            if self.orient == 'v':
                self._order = sorted(np.unique(data[self.xcolumn]), key=StrInt)
            else:
                self._order = sorted(np.unique(data[self.xcolumn]), reverse=True, key=StrInt)
        # Make sure there are any columns to plot
        in_order = set(self._order)
        data_order = set(np.unique(data[self.xcolumn]))
        if len(in_order & data_order) == 0:
            raise ValueError(f'No intersection between order categories {in_order} and data categories {data_order}')

        if self.hue_column is None:
            self._color_table = colorwheel(self.palette, n_colors=len(self._order))
        else:
            if self._hue_order is None:
                self._hue_order = sorted(np.unique(data[self.hue_column]), key=StrInt)
            in_hue_order = set(self._hue_order)
            hue_data_order = set(np.unique(data[self.hue_column]))
            if len(in_hue_order & hue_data_order) == 0:
                raise ValueError(f'No intersection between hue order categories {in_hue_order} and hue data categories {hue_data_order}')
            self._color_table = colorwheel(self.palette, n_colors=len(self._hue_order))

    def load_metadata(self):
        """ Load the metadata for this modality """

        # Y-axis attributes
        if self._ylabel is None:
            self._ylabel = {
                'pctifr': '% Matches Between Frames',
                'pctirr': '% Matches To Consensus',
                'ifr': '% Matches Between Frames',
                'irr': '% Matches To Consensus',
                'cellarea': 'Cell Area ($\\mu m^2$)',
                'curl': 'Curl ($rads/min * 10^{-2}$)',
                'density': 'Cell Density ($\\mu m^{-2} * 10^{-3}$)',
                'displacement': 'Cell Displacement ($\\mu m$)',
                'distance': 'Cell Distance ($\\mu m$)',
                'divergence': 'Divergence ($area/min * 10^{-2}$)',
                'persistence': 'Cell Persistence',
                'velocity': 'Velocity Magnitude ($\\mu m/min$)',
                'activeperiod': 'Active Period (min)',
                'quiescentperiod': 'Quiescent Period (min)',
                'pearsoncorrelation': 'Pearson Correlation',
            }.get(self.norm_ycolumn)
        if self._yscale is None:
            self._yscale = 1.0

        # X-axis attributes
        if self._xlabel is None:
            self._xlabel = {
                'percentlabeled': 'Percent Labeled',
                'annotator': 'Annotator',
                'detector': 'Detector',
                'radius': 'Colony Radius',
                'media': 'Culture Media',
                'substrate': 'Substrate',
                'colonysize': 'Colony Size',
                'bbox': 'Region of Interest',
                'distance': 'Neighborhood Distance',
            }[self.norm_xcolumn]
        self._xticklabels = {
            'percentlabeled': lambda o: '{}%'.format(o),
            'detector': lambda o: DETECTOR_NAMES[o.lower().replace(' ', '_').replace('-', '_')],
            'radius': lambda o: RADIUS_NAMES[o.lower()],
        }.get(self.norm_xcolumn, lambda o: o)
        if self._xticklabel_rotation is None:
            self._xticklabel_rotation = {
                'detector': 90,
                'radius': 90,
            }.get(self.norm_xcolumn, 0)

        if self._bar_baseline is None:
            self._bar_baseline = 0.0

        # Force an ordering for specific x-columns
        if self._order is None:
            self._order = {
                'detector': [
                    'countception-r003-n50000',
                    'fcrn_a_wide-r003-n75000',
                    'fcrn_b_wide-r003-n75000',
                    'residual_unet-r004-n25000',
                    'unet-r001-n50000',
                    'composite-d3-final',
                ],
                'radius': [
                    '0.0-0.6',
                    '0.6-0.8',
                    '0.8-1.0',
                ],
                'media': [
                    'mTeSR',
                    'E8',
                ],
                'substrate': [
                    'Matrigel',
                    'Vitronectin',
                    'Laminin',
                ],
                'colonysize': [
                    '100 Cell',
                    '500 Cell',
                ]
            }.get(self.norm_xcolumn, self._order)

    def calc_significance_coords(self, coord_map: Dict) -> List:
        """ Work out the coordinates for the bars

        :param dict coord_map:
            The mapping of (column1, column2): significance for all the columns
        :returns:
            A list of (x1, x2, significance) coordinates for plotting
        """
        ysignificance = []
        if self._sigdata is None:
            return ysignificance

        xkey1 = self.xcolumn + '1'
        xkey2 = self.xcolumn + '2'
        if self.hue_column is None:
            hkey1 = hkey2 = None
        else:
            hkey1 = self.hue_column + '1'
            hkey2 = self.hue_column + '2'
        for i, row in self._sigdata.iterrows():
            if hkey1 is None:
                ckey1 = row[xkey1]
            else:
                ckey1 = (row[xkey1], row[hkey1])

            # Skip significance calls not included in the total ordering
            if ckey1 not in coord_map:
                continue
            coord1 = coord_map[ckey1]

            if hkey2 is None:
                ckey2 = row[xkey2]
            else:
                ckey2 = (row[xkey2], row[hkey2])

            # Skip significance calls not included in the total ordering
            if ckey2 not in coord_map:
                continue
            coord2 = coord_map[ckey2]
            pvalue = row['P-value']
            ysignificance.append((coord1[0], coord2[0], pvalue))
        return ysignificance

    def calc_bars(self):
        """ Calculate the coordinates needed for bars """

        bar_width = self.bar_width
        bar_padding = self.bar_padding
        cat_padding = self.bar_cat_padding
        bar_baseline = self._bar_baseline
        if bar_baseline is None:
            bar_baseline = 0

        # Work out the number of bars
        if self._hue_order is None:
            hue_order = [None]
        else:
            hue_order = self._hue_order
        num_hue_cats = len(hue_order)

        # Look at the data
        data = self._data
        sample_data = self._sampledata
        xcolumn = self.xcolumn
        ycolumn = self.ycolumn
        hue_column = self.hue_column

        # Lay out the data in x/y
        xcoords = []
        xtick_coords = []
        ycoords = []
        ylows = []
        yhighs = []
        ynum_samples = []
        samples_x = []
        samples_y = []
        samples_color = []
        coord_map = {}
        bar_hcat = []
        bar_color = []

        # Load data, calculate positions
        x = 0.0
        for i, xcat in enumerate(self._order):
            x += cat_padding
            for j, hcat in enumerate(hue_order):
                mask = data[xcolumn] == xcat
                if hcat is not None:
                    mask = np.logical_and(mask, data[hue_column] == hcat)
                if ~np.any(mask):
                    if hcat is None:
                        raise ValueError(f'Got empty category "{xcat}"')
                    else:
                        raise ValueError(f'Got empty dual category "{xcat}" "{hcat}"')

                if self._hue_order is None:
                    xcolor = self._color_table[i]
                else:
                    xcolor = self._color_table[j]

                if self.plot_individual_samples:
                    sample_mask = sample_data[xcolumn] == xcat
                    if hcat is not None:
                        sample_mask = np.logical_and(sample_mask, sample_data[hue_column] == hcat)
                    if np.any(sample_mask):
                        sample_vals = sample_data.loc[sample_mask, ycolumn].values * self._yscale
                        samples_x.extend([x + bar_width/2.0]*sample_vals.shape[0])
                        samples_y.extend(sample_vals)
                        if self.marker_facecolor is None:
                            marker_facecolor = xcolor
                        else:
                            marker_facecolor = self.marker_facecolor
                        samples_color.extend([marker_facecolor]*sample_vals.shape[0])

                subdata = data[mask]

                ymean = subdata[ycolumn + ' mean'].values * self._yscale - bar_baseline
                ylow = subdata[ycolumn + ' ci low'].values * self._yscale - bar_baseline
                yhigh = subdata[ycolumn + ' ci high'].values * self._yscale - bar_baseline
                ynum = subdata[ycolumn + ' n'].values

                if not np.isfinite(ylow):
                    ylow = ymean
                if not np.isfinite(yhigh):
                    yhigh = ymean

                xcoords.append(x + bar_width/2.0)
                ycoords.extend(ymean)
                ylows.extend(ymean - ylow)
                yhighs.extend(yhigh - ymean)
                ynum_samples.extend(ynum)

                if hcat is None:
                    coord_map[xcat] = (xcoords[-1], ycoords[-1])
                else:
                    coord_map[(xcat, hcat)] = (xcoords[-1], ycoords[-1])

                bar_hcat.append(hcat)
                bar_color.append(xcolor)

                x += bar_width
                if j < len(hue_order)-1:
                    x += bar_padding

            # Work out where the ticks go
            xtick_coords.append(np.mean(xcoords[-num_hue_cats:]))

        # Work out where the significance bars go
        ysignificance = self.calc_significance_coords(coord_map)

        self.figure_x = x + cat_padding
        self._ycoords = np.array(ycoords)
        self._samples_y = np.array(samples_y)
        self._samples_x = np.array(samples_x)
        self._samples_color = samples_color
        self._yerr = np.stack([np.array(ylows),
                               np.array(yhighs)], axis=0)
        self._ynum = np.array(ynum_samples)
        if ysignificance != []:
            self._ysig = np.array(ysignificance)
        self._xcoords = np.array(xcoords)
        self._xtick_coords = np.array(xtick_coords)
        self._bar_hcat = bar_hcat
        self._bar_color = bar_color

    def calc_boxes(self):
        """ Calculate the coordinates needed for boxes """

        box_width = self.box_width
        box_padding = self.box_padding
        cat_padding = self.box_cat_padding

        # Work out the number of bars
        if self._hue_order is None:
            hue_order = [None]
        else:
            hue_order = self._hue_order
        num_hue_cats = len(hue_order)

        # Look at the data
        data = self._data
        xcolumn = self.xcolumn
        ycolumn = self.ycolumn
        hue_column = self.hue_column

        # Lay out the data in x/y
        xcoords = []
        xtick_coords = []
        yp5s = []
        yp25s = []
        yp50s = []
        yp75s = []
        yp95s = []
        yerr = []
        ynum_samples = []
        coord_map = {}
        box_hcat = []
        box_color = []

        # Load data, calculate positions
        x = 0.0
        for i, xcat in enumerate(self._order):
            x += cat_padding
            for j, hcat in enumerate(hue_order):
                mask = data[xcolumn] == xcat
                if hcat is not None:
                    mask = np.logical_and(mask, data[hue_column] == hcat)
                if ~np.any(mask):
                    if hcat is None:
                        raise ValueError(f'Got empty category {xcat}')
                    else:
                        raise ValueError(f'Got empty dual category {xcat} {hcat}')
                subdata = data[mask]

                yp5 = subdata[ycolumn + ' p5'].values * self._yscale
                yp25 = subdata[ycolumn + ' p25'].values * self._yscale
                yp50 = subdata[ycolumn + ' p50'].values * self._yscale
                yp75 = subdata[ycolumn + ' p75'].values * self._yscale
                yp95 = subdata[ycolumn + ' p95'].values * self._yscale

                yerr.append(yp95-yp5)

                ynum = subdata[ycolumn + ' n'].values

                xcoords.append(x + box_width/2.0)
                yp5s.append(yp5)
                yp25s.append(yp25)
                yp50s.append(yp50)
                yp75s.append(yp75)
                yp95s.append(yp95)
                ynum_samples.extend(ynum)

                if hcat is None:
                    coord_map[xcat] = (xcoords[-1], yp95s[-1])
                else:
                    coord_map[(xcat, hcat)] = (xcoords[-1], yp95s[-1])

                box_hcat.append(hcat)
                if self._hue_order is None:
                    box_color.append(self._color_table[i])
                else:
                    box_color.append(self._color_table[j])

                x += box_width
                if j < len(hue_order)-1:
                    x += box_padding

            # Work out where the ticks go
            xtick_coords.append(np.mean(xcoords[-num_hue_cats:]))

        # Work out where the significance bars go
        ysignificance = self.calc_significance_coords(coord_map)

        self.figure_x = x + cat_padding
        self._ycoords = np.stack([yp5s, yp25s, yp50s, yp75s, yp95s], axis=1)
        self._yerr = yerr
        self._ynum = np.array(ynum_samples)
        if ysignificance != []:
            self._ysig = np.array(ysignificance)
        self._xcoords = np.array(xcoords)
        self._xtick_coords = np.array(xtick_coords)
        self._box_hcat = box_hcat
        self._box_color = box_color

    def add_significance(self, ax, markers=None):
        """ Plot significance bars over the bars """

        if self._ysig is None:
            return

        # Sort the significance markers
        if markers is None:
            markers = {
                0.05: '*',
                0.01: '**',
                0.001: '***',
            }
        else:
            markers = dict(markers)
        sorted_markers = list(sorted(markers.items(), key=lambda x: x[0]))

        barcolor = self.significance_barcolor
        linewidth = self.significance_linewidth
        fontsize = self.significance_fontsize

        if self._ylimits is None:
            ymax = np.max(self._ycoords) + np.max(self._yerr)
            ymin = np.min(self._ycoords) - np.min(self._yerr)
        else:
            ymin = np.min(self._ylimits)
            ymax = np.max(self._ylimits)

        yrange = ymax - ymin
        ystep = yrange * 0.02
        ylevel = yrange * 0.9 + ymin

        # ymax = (np.max(self._ycoords) + np.max(self._yerr))
        # ystep = (np.max(self._ycoords) - np.min(self._ycoords)) * 0.05

        for xcoord1, xcoord2, pvalue in self._ysig:
            psymbol = None
            for pthreshold, symbol in sorted_markers:
                if pvalue <= pthreshold:
                    psymbol = symbol
                    break
            if psymbol is None:
                continue
            ylevel += ystep
            if self.orient == 'v':
                ax.plot([xcoord1, xcoord2], [ylevel, ylevel],
                        color=barcolor, linewidth=linewidth, clip_on=False)
                ax.text((xcoord1+xcoord2)/2, ylevel+ystep, psymbol,
                        color=barcolor, fontsize=fontsize,
                        horizontalalignment='center',
                        verticalalignment='top',
                        clip_on=False)
            else:
                ax.plot([ylevel, ylevel], [xcoord1, xcoord2],
                        color=barcolor, linewidth=linewidth, clip_on=False)
                ax.text(ylevel+ystep, (xcoord1+xcoord2)/2, psymbol,
                        color=barcolor, fontsize=fontsize,
                        rotation=90,
                        horizontalalignment='right',
                        verticalalignment='center',
                        clip_on=False)

    def add_number_of_samples(self, ax):
        """ Plot the number of samples over the bars """
        if not self.plot_num_points:
            return

        color = self.num_points_color
        fontsize = self.num_points_fontsize

        ymax = (np.max(self._ycoords) + np.max(self._yerr))*1.05
        for i, xcoord in enumerate(self._xcoords):
            num_points = self._ynum[i]

            # Label the number of samples
            if self.orient == 'v':
                ax.text(xcoord, ymax, f'n={num_points:,}',
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=fontsize,
                        color=color,
                        clip_on=False)
            else:
                ax.text(ymax, xcoord, f'n={num_points:,}',
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=fontsize,
                        color=color,
                        clip_on=False)

    def add_samples(self, ax):
        """ Add the individual samples to the axis """
        if not self.plot_individual_samples:
            return

        for x, y, color in zip(self._samples_x, self._samples_y, self._samples_color):
            # Plot the individual samples
            if self.orient == 'v':
                ax.plot(x, y, color=color, marker='o', linestyle='',
                        markeredgecolor=self.marker_edgecolor,
                        markeredgewidth=1.0,
                        markersize=self.marker_size)
            else:
                ax.plot(y, x, color=color, marker='o', linestyle='',
                        markeredgecolor=self.marker_edgecolor,
                        markeredgewidth=1.0,
                        markersize=self.marker_size)

    def add_axis_labels(self, ax):
        """ Add the axis labels and orient them correctly """

        if abs(self._xticklabel_rotation) < 5.0:
            horizontalalignment = 'center'
        else:
            horizontalalignment = 'right'
        verticalalignment = 'top'

        style = set_plot_style(self.plot_style)

        if self.orient == 'v':
            ax.set_xticks(self._xtick_coords)
            ax.set_xticklabels([self._xticklabels(o) for o in self._order])
            ax.set_xlabel(self._xlabel)
            ax.set_ylabel(self._ylabel)
            style.rotate_xticklabels(ax, self._xticklabel_rotation,
                                     horizontalalignment=horizontalalignment,
                                     verticalalignment=verticalalignment)
        else:
            ax.set_yticks(self._xtick_coords)
            ax.set_yticklabels([self._xticklabels(o) for o in self._order])
            ax.set_ylabel(self._xlabel)
            ax.set_xlabel(self._ylabel)
            # FIXME: not sure if these anchors are right
            style.rotate_yticklabels(ax, self._xticklabel_rotation,
                                     horizontalalignment=horizontalalignment,
                                     verticalalignment=verticalalignment)

    def plot(self):
        """ Generate the correct plot """
        return getattr(self, f'plot_{self.plot_type}')()

    def plot_boxes(self):
        """ Plot categories as boxes """

        self.calc_boxes()

        if self.orient == 'v':
            xlimits = [(0, self.figure_x)]
            ylimits = self._ylimits
            figsize = (self.figure_x, self.figure_y)
        else:
            xlimits = self._ylimits
            ylimits = [(0, self.figure_x)]
            figsize = (self.figure_y, self.figure_x)

        with set_plot_style(self.plot_style) as style:
            with SplitAxes(xlimits=xlimits,
                           ylimits=ylimits,
                           figsize=figsize) as ax:
                labeled = set()
                for i in range(self._ycoords.shape[0]):
                    if self._box_hcat[i] not in labeled:
                        label = self._box_hcat[i]
                        labeled.add(label)
                    else:
                        label = None
                    add_single_boxplot(
                        ax=ax,
                        xcoord=self._xcoords[i],
                        ycoords=self._ycoords[i, :],
                        width=self.bar_width,
                        color=self._box_color[i],
                        edgecolor=self.bar_edgecolor,
                        linewidth=self.bar_linewidth,
                        orient=self.orient,
                        label=label, error_kw={
                            'ecolor': self.bar_error_linecolor,
                            'capthick': self.bar_error_linewidth,
                            'capsize': self.bar_error_capsize,
                            'elinewidth': self.bar_error_linewidth,
                        })
                self.add_significance(ax)
                self.add_number_of_samples(ax)
                self.add_axis_labels(ax)

                if self._hue_order is not None:
                    ax.legend()
                if self._yscale_type is not None:
                    ax.set_yscale(self._yscale_type)
            for suffix in self.suffixes:
                outfile = (self.outfile.parent / f'{self.outfile.stem}{suffix}')
                style.show(outfile=outfile, close=False, tight_layout=self.tight_layout)
            plt.close()

    def plot_bars(self):
        """ Plot categories as bars """

        self.calc_bars()

        if self.orient == 'v':
            xlimits = [(0, self.figure_x)]
            ylimits = self._ylimits
            figsize = (self.figure_x, self.figure_y)
        else:
            xlimits = self._ylimits
            ylimits = [(0, self.figure_x)]
            figsize = (self.figure_y, self.figure_x)

        with set_plot_style(self.plot_style) as style:
            with SplitAxes(xlimits=xlimits,
                           ylimits=ylimits,
                           figsize=figsize) as ax:
                labeled = set()
                for i in range(self._ycoords.shape[0]):
                    if self._bar_hcat[i] not in labeled:
                        label = self._bar_hcat[i]
                        labeled.add(label)
                    else:
                        label = None

                    add_single_barplot(
                        ax=ax,
                        bottom=self._bar_baseline,
                        xcoord=self._xcoords[i],
                        ycoords=[self._ycoords[i], self._yerr[0, i], self._yerr[1, i]],
                        width=self.bar_width,
                        color=self._bar_color[i],
                        edgecolor=self.bar_edgecolor,
                        linewidth=self.bar_linewidth,
                        orient=self.orient,
                        label=label,
                        errorbar_side=self._bar_errorbar_side,
                        error_kw={
                            'ecolor': self.bar_error_linecolor,
                            'capthick': self.bar_error_linewidth,
                            'capsize': self.bar_error_capsize,
                            'elinewidth': self.bar_error_linewidth,
                        })
                self.add_significance(ax)
                self.add_samples(ax)
                self.add_number_of_samples(ax)
                self.add_axis_labels(ax)

                if self._hue_order is not None:
                    ax.legend()

                if self._yscale_type is not None:
                    ax.set_yscale(self._yscale_type)

            for suffix in self.suffixes:
                outfile = (self.outfile.parent / f'{self.outfile.stem}{suffix}')
                style.show(outfile=outfile, close=False, tight_layout=self.tight_layout)
            plt.close()

    def plot_traces_bars(self):
        """ Plot the categories as a trace with bar errors """
        return self.plot_traces(err_style='bars')

    def plot_traces_bands(self):
        """ Plot the categories as a trace with band errors """
        return self.plot_traces(err_style='bands')

    def plot_traces(self, err_style: str = 'bars'):
        """ Plot categories as a trace

        :param str err_style:
            What style to use for the error bars, one of "bars" or "bands"
        """

        print('Traces...')

        if err_style in ('bar', 'bars'):
            figure_x = self.bar_width * len(self._order) + self.bar_width
        else:
            figure_x = self.figure_y

        figsize = (figure_x, self.figure_y)

        data = self._data
        xcolumn = self.xcolumn
        ycolumn = self.ycolumn
        hue_column = self.hue_column

        if self._hue_order is None:
            hue_order = [None]
        else:
            hue_order = self._hue_order

        new_columns = {c: c.lower() for c in data.columns}
        data.rename(columns=new_columns, inplace=True)

        if self._xlimits is None:
            try:
                num_order = np.array([float(o) for o in self._order])
                xmin = np.min(num_order)
                xmax = np.max(num_order)
                self._xlimits = ((xmin, xmax), )
            except ValueError:
                num_order = np.arange(len(self._order))
                self._xlimits = ((num_order[0] - 1.0, num_order[-1] + 1.0), )

        with set_plot_style(self.plot_style) as style:
            with SplitAxes(ylimits=self._ylimits,
                           xlimits=self._xlimits,
                           figsize=figsize) as ax:
                for j, hcat in enumerate(hue_order):
                    ymeans = []
                    ylows = []
                    yhighs = []
                    xcoords = []
                    xmeans = []

                    for i, xcat in enumerate(self._order):
                        if hcat is None:
                            mask = data[xcolumn.lower()] == xcat
                        else:
                            mask = np.logical_and(data[xcolumn.lower()] == xcat,
                                                  data[hue_column.lower()] == hcat)
                        if ~np.any(mask):
                            continue
                        subdata = data[mask]

                        ymean = subdata[ycolumn.lower() + ' mean'].values * self._yscale
                        ylow = subdata[ycolumn.lower() + ' ci low'].values * self._yscale
                        yhigh = subdata[ycolumn.lower() + ' ci high'].values * self._yscale

                        xcoords.append(i)
                        xmeans.append(xcat)
                        ymeans.extend(ymean)
                        ylows.extend(ylow)
                        yhighs.extend(yhigh)

                    ymeans = np.array(ymeans)
                    ylows = np.array(ylows)
                    yhighs = np.array(yhighs)
                    yerr = np.stack([ymeans - ylows, yhighs - ymeans], axis=0)
                    xcoords = np.array(xcoords)
                    color = self._color_table[j]

                    if err_style in ('bar', 'bars'):
                        ax.errorbar(
                            xcoords, ymeans, yerr,
                            capsize=self.bar_error_capsize,
                            capthick=self.bar_error_linewidth,
                            elinewidth=self.bar_error_linewidth,
                            linewidth=self.bar_error_linewidth,
                            label=hcat,
                            color=color,
                            ecolor=color)
                    elif err_style in ('band', 'bands'):

                        xmeans = np.array([float(x) for x in xmeans])

                        ax.fill_between(xmeans, ylows, yhighs, facecolor=color, alpha=0.5)
                        ax.plot(xmeans, ymeans, '-', color=color, label=hcat,
                                linewidth=self.bar_linewidth)
                    else:
                        raise KeyError(f'Unknown error style: "{err_style}"')

                if err_style in ('bar', 'bars'):
                    ax.set_xticks(range(len(self._order)))
                    ax.set_xticklabels([self._xticklabels(o) for o in self._order])
                else:
                    xnum = int(round(self.figure_y))

                    xmin = min([min(limit) for limit in self._xlimits])
                    xmax = max([max(limit) for limit in self._xlimits])

                    xticks = np.linspace(xmin, xmax, xnum)

                    ax.set_xticks(xticks)
                    ax.set_xticklabels([f'{x:0.0f}' for x in xticks])
                ax.set_xlabel(self._xlabel)
                ax.set_ylabel(self._ylabel)
                ax.legend()

                if self._yscale_type is not None:
                    ax.set_yscale(self._yscale_type)

            for suffix in self.suffixes:
                outfile = (self.outfile.parent / f'{self.outfile.stem}{suffix}')
                style.show(outfile=outfile, close=False, tight_layout=self.tight_layout)
            plt.close()


# Main Function


def replot_bars(datafile: pathlib.Path,
                xcolumn: Optional[str] = None,
                ycolumn: Optional[str] = None,
                hue_column: Optional[str] = None,
                palette: str = PALETTE,
                **kwargs):
    """ Replot bar data as a new plot

    :param Path datafile:
        The file to read to regenerate the plots
    :param str xcolumn:
        The column for categorical data
    :param str ycolumn:
        The column for value data
    :param str hue_column:
        The secondary category column
    :param str palette:
        The palette to use to plot
    """
    if xcolumn is None and ycolumn is None:
        xcolumn, ycolumn, hue_column, palette = DEFAULT_COLUMNS[datafile.name]
    kwargs.update({
        'datafile': datafile,
        'xcolumn': xcolumn,
        'ycolumn': ycolumn,
        'hue_column': hue_column,
        'palette': palette,
    })

    plotter = BarPlotter(**kwargs)
    plotter.load_data()
    plotter.load_metadata()
    plotter.plot()
