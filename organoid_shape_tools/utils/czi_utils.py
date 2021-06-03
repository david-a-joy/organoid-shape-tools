""" Utilities for extracting metadata from CZI files

Main class for extracting tiled images from CZI files:

* :py:class:`CZIManager`: Converts a CZI block to a nested directory of TIFF files

Helper class for holding information about tile position and name:

* :py:class:`TileData`: Create a mapping between user assigned tile data, tile index in the stack, and tile position

Utility functions to read metadata properties of the tiles:

* :py:func:`print_tile_metadata`: Pretty print tile metadata
* :py:func:`extract_seg_coordinates`: Find the individual x,y,z coordinates of a raw sub-block
* :py:func:`extract_tile_names`: Find the name and position of an individual tile
* :py:func:`extract_channel_names`: Find the name and index for the channel for each tile

I/O helpers to find the metadata in the directories:

* :py:func:`find_metadata`: Find the metadata.xml file in the tree
* :py:func:`read_metadata`: Read the metadata.xml file from the tree
* :py:func:`write_metadata`: Write the metadata.xml file to the tree

API Documentation
-----------------

"""

# Imports
import re
import pathlib
import shutil
import datetime
import traceback
from collections import Counter
from typing import Generator, Tuple, Dict, Optional

try:
    from lxml import etree
except ImportError:
    from xml.etree import ElementTree as etree

# 3rd party
import numpy as np

from PIL import Image

from sklearn.neighbors import BallTree

from czifile.czifile import CziFile, DECOMPRESS

# Our own imports
from . import image_utils, parallel_utils

# Constants

reTILENAME = re.compile(r'[^a-z0-9_-]+', re.IGNORECASE)  # replace any characters not in this list

# Space scale xpath
SPACE_SCALE_XPATH = '/'.join([
    '.', 'Metadata', 'Scaling', 'Items', 'Distance',
])

# Xpath strings for when tiles are embedded in timeseries
SINGLE_TILE_XPATH_TS = '/'.join([
    '.', 'Metadata', 'Experiment', 'ExperimentBlocks', 'AcquisitionBlock',
    'SubDimensionSetups', 'TimeSeriesSetup', 'SubDimensionSetups',
    'RegionsSetup', 'SampleHolder', 'SingleTileRegions', 'SingleTileRegion',
])
MULTI_TILE_XPATH_TS = '/'.join([
    '.', 'Metadata', 'Experiment', 'ExperimentBlocks', 'AcquisitionBlock',
    'SubDimensionSetups', 'TimeSeriesSetup', 'SubDimensionSetups',
    'RegionsSetup', 'SampleHolder', 'TileRegions', 'TileRegion',
])

# Xpath strings for when tiles are not in timeseries
SINGLE_TILE_XPATH = '/'.join([
    '.', 'Metadata', 'Experiment', 'ExperimentBlocks', 'AcquisitionBlock',
    'SubDimensionSetups', 'RegionsSetup', 'SampleHolder', 'SingleTileRegions', 'SingleTileRegion',
])
MULTI_TILE_XPATH = '/'.join([
    '.', 'Metadata', 'Experiment', 'ExperimentBlocks', 'AcquisitionBlock',
    'SubDimensionSetups', 'RegionsSetup', 'SampleHolder', 'TileRegions', 'TileRegion',
])

# Z-stack settings
SINGLE_ZSTACK_XPATH = '/'.join([
    '.', 'Metadata', 'Experiment', 'ExperimentBlocks', 'AcquisitionBlock',
    'SubDimensionSetups', 'ZStackSetup', 'Interval', 'Distance', 'Value',
])
MULTI_TILE_ZSTACK_XPATH = '/'.join([
    '.', 'Metadata', 'Experiment', 'ExperimentBlocks', 'AcquisitionBlock',
    'SubDimensionSetups', 'TilesSetup', 'SubDimensionSetups', 'ZStackSetup',
    'Interval', 'Distance', 'Value',
])

# Multi-tile comb pattern
MULTI_TILE_COMB_XPATH = '/'.join([
    '.', 'Metadata', 'Experiment', 'ExperimentBlocks', 'AcquisitionBlock',
    'SubDimensionSetups', 'RegionsSetup', 'SampleHolder', 'PositionedRegionsScanMode',
])
MULTI_TILE_COMB_XPATH_TS = '/'.join([
    '.', 'Metadata', 'Experiment', 'ExperimentBlocks', 'AcquisitionBlock',
    'SubDimensionSetups', 'TimeSeriesSetup', 'SubDimensionSetups',
    'RegionsSetup', 'SampleHolder', 'PositionedRegionsScanMode',
])

# Mercifully, channel data seems to be independent of whether or not you have timelapses on
CHANNEL_XPATH = '/'.join([
    '.', 'Metadata', 'Information', 'Image', 'Dimensions', 'Channels',
    'Channel',
])


# Classes


class TileData(object):
    """ Metadata for tile sets

    Create a mapping between user assigned tile data, tile index in the stack,
    tile position, and number of rows and columns of subtiles for a multi-tile image.

    :param int index:
        The index for the tile in the tile metadata (numbered from 0). This is
        the order the tiles were **ASSIGNED** in the tile manager, not the order
        they were acquired in.
    :param str name:
        The name for the tile in the tile metadata (user assigned)
    :param int tileid:
        The index for the tile in imaging order (numbered from 0). This is the
        order that the tiles were **ACQUIRED** in, not the order they were
        assigned in the tile manager.
    :param float x:
        The x coordinate of the tile center (um)
    :param float y:
        The y coordinate of the tile center (um)
    :param float z:
        The z coordinate of the tile focus plane (um)
    :param int rows:
        Number of rows in the multi-tiled image (**NOT THE IMAGE SHAPE**)
    :param int cols:
        Number of columns in the multi-tiled image (**NOT THE IMAGE SHAPE**)
    """

    def __init__(self, index: int, name: str, tileid: int,
                 x: float, y: float, z: float,
                 rows: int = 1, cols: int = 1):
        # Map tile index, tile name, position in the CZI file
        self.index = int(index)
        self.name = name
        self.tileid = int(tileid)

        # Tile position parameters
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

        # How many subtiles in each direction for this tile
        self.rows = int(rows)  # Tiles in the up-down direction
        self.cols = int(cols)  # Tiles in the left right direction

    def __str__(self):
        return 's{:02d}-{}({:0.2f},{:0.2f},{:0.2f})'.format(
            self.index+1, self.name, self.x, self.y, self.z)

    __repr__ = __str__

    def get_name(self) -> str:
        """ Get a directory safe version of the name """
        return reTILENAME.sub('_', self.name)

    def get_dirname(self) -> str:
        """ Get the name of the directory for this tile """
        if self.name in (None, ''):
            return 's{:02d}'.format(self.index+1)
        return 's{:02d}-{}'.format(self.index+1, self.get_name())


class CZIManager(object):
    """ Manage the weird and nightmarish CZI file format

    Convert the CZI file to a stack of uncorrected tiffs:

    .. code-block:: python

        with czi_utils.CZIManager(infile) as czi:
            czi.dump_all(outroot)

    Apply a contrast correction and subset the images while converting, using 8 cores:

    .. code-block:: python

        with czi_utils.CZIManager(infile) as czi:
            czi.set_crop(bbox.x0, bbox.x1, bbox.y0, bbox.y1)
            czi.set_contrast_correction('equalize_adapthist')
            czi.dump_all(outroot, processes=8)

    :param Path infile:
        The path to the CZI file
    """

    def __init__(self, infile: pathlib.Path,
                 ignore_names: bool = False):

        self.infile = pathlib.Path(infile).resolve()
        self.ignore_names = ignore_names

        self.tile_shape = None

        self._frame_count = 0
        self._crop_x_st, self._crop_x_ed = 0, -1
        self._crop_y_st, self._crop_y_ed = 0, -1
        self._max_frames = -1
        self._fix_contrast = None

        self._czi = None
        self._czi_metadata = None

        self._czi_axes = None
        self._czi_shape = None
        self._tile_names = None
        self._channel_names = None

        self._tile_index = None
        self._slice_index = None
        self._timepoint_index = None
        self._channel_index = None
        self._x_index = None
        self._y_index = None

        self._tile_coords = None
        self._tile_coord_tree = None
        self._tile_sample_index = None

        # Position tables
        self._positions = None
        self._position_index = None
        self._num_segments = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    @property
    def czi_metadata(self):
        """ Lazy load the metadata from the CZI file """
        if self._czi_metadata is None:
            if hasattr(self._czi.metadata, '__call__'):
                metadata = self._czi.metadata()
            else:
                metadata = self._czi.metadata
            self._czi_metadata = etree.fromstring(metadata)
        return self._czi_metadata

    @property
    def position_segments(self):
        """ Iterator over valid segments """
        for seg, _, _ in self._get_segment_iterator():
            yield np.squeeze(seg.data(resize=False))

    def _load_axis_indices(self):
        # Reverse engineer the indicies that map raw tile coordinates to human-readable attributes
        czi_axes = self._czi_axes

        # Not sure what it means, but here are the codes
        # B,S,T,C,Y,X,0
        x_index = czi_axes.index('X')
        y_index = czi_axes.index('Y')
        if 'S' in czi_axes:
            tile_index = czi_axes.index('S')
        else:
            tile_index = None
        if 'Z' in czi_axes:
            slice_index = czi_axes.index('Z')
        else:
            slice_index = None
        if 'T' in czi_axes:
            timepoint_index = czi_axes.index('T')
        else:
            timepoint_index = None
        if 'M' in czi_axes:
            subtile_index = czi_axes.index('M')
        else:
            subtile_index = None

        if 'C' in czi_axes:
            channel_index = czi_axes.index('C')
        else:
            channel_index = None

        # Handle multi-view volumes
        if 'V' in czi_axes:
            multi_view_index = czi_axes.index('V')
        else:
            multi_view_index = None

        self._tile_index = tile_index
        self._subtile_index = subtile_index
        self._timepoint_index = timepoint_index
        self._channel_index = channel_index
        self._slice_index = slice_index
        self._x_index = x_index
        self._y_index = y_index
        self._multi_view_index = multi_view_index

    def _load_frame_tables(self):
        # Load the tables for the individual image frames
        czi = self._czi

        print(czi._fh.name.capitalize())
        print("(Carl Zeiss Image File)")
        print(str(czi.header))
        print("MetadataSegment")
        czi_axes = str(czi.axes)
        print(czi_axes)

        czi_shape = tuple(int(s) for s in czi.shape)
        print(czi_shape)

        print(str(czi.dtype))
        print('Got {} subblocks'.format(len(czi.subblock_directory)))

        metadata = self.czi_metadata

        # Extract some useful keys from the metadata
        tile_names = extract_tile_names(metadata)
        channel_names = extract_channel_names(metadata)

        print('Got tiles: {}'.format(tile_names))
        print('Got channels: {}'.format(channel_names))

        self._czi_axes = czi_axes
        self._load_axis_indices()

        # Load the official tile coordinates
        if tile_names:
            tile_coords = np.array([(t.x, t.y) for _, t in sorted(tile_names.items())])
        else:
            tile_coords = np.zeros((0, 2))
        assert tile_coords.shape[1] == 2

        # Make sure the tiles and timepint
        if self._tile_index is not None and tile_names != {}:
            if len(tile_names) != czi.shape[self._tile_index]:
                if self.ignore_names:
                    {i: f'Tile{i:02d}' for i in range(czi.shape[self._tile_index])}
                else:
                    raise ValueError('Expected tiles {}, got {} tiles'.format(len(tile_names), czi.shape[self._tile_index]))
        if self._channel_index is None:
            channel_scale = czi_shape[-1]
        else:
            channel_scale = 1.0
            if len(channel_names) != czi.shape[self._channel_index]:
                if self.ignore_names:
                    channel_names = {i: f'Channel{i:02d}' for i in range(czi.shape[self._channel_index])}
                else:
                    raise ValueError('Expected channels {}, got {} channels'.format(channel_names, czi.shape[self._channel_index]))

        # Figure out how many segments there are
        all_indices = [self._tile_index, self._slice_index, self._timepoint_index, self._channel_index, self._subtile_index, self._multi_view_index]
        found_segments = np.prod([czi_shape[i] for i in all_indices if i is not None]) * channel_scale
        total_segments = np.prod([czi_shape[i] for i in range(len(czi_shape)) if i not in (self._x_index, self._y_index)])

        if found_segments != total_segments:
            print('Cannot read axis specification!!!')
            print(f'Spec:  {czi_axes}')
            print(f'Shape: {czi_shape}')
            raise ValueError('Got invalid segment count, expected {} blocks found {}'.format(
                total_segments, found_segments))

        # Stash all these indices for later
        self._czi_axes = czi_axes
        self._czi_shape = czi_shape

        self._tile_names = tile_names
        self._channel_names = channel_names

        self._tile_coords = tile_coords
        if tile_coords.shape[0] > 0:
            self._tile_coord_tree = BallTree(tile_coords)

    def _get_segment_iterator(self) -> Generator:
        """ Iterate over the raw position info for the file """

        for seg in self._czi.segments(kind='ZISRAWSUBBLOCK'):
            meta = seg.metadata()
            # Throw out segments that we don't want
            if meta is not None and meta.get('AttachmentSchema') is not None:
                continue

            # Filter out the weird pyramid tiles
            if seg.pyramid_type == 2:
                continue

            directory = seg.directory_entry
            shape = tuple(s for s in directory.stored_shape if s != 1)

            if len(shape) == 2:
                rows, cols = shape
                colors = 1
            elif len(shape) == 3:
                rows, cols, colors = shape
            else:
                print('Got non-2D and non-3D segment: {}'.format(shape))
                continue

            # Throw out invalid chunks
            if rows < 20 or cols < 20:
                continue

            yield seg, meta, directory

    def _load_position_table(self):
        # Index the positions for each item in the table
        positions = []
        position_timestamps = []
        position_index = []

        tile_shape = tuple(self._czi_shape[i] for i in range(len(self._czi_shape))
                           if i not in (self._x_index, self._y_index))
        x_shape = self._czi_shape[self._x_index]
        y_shape = self._czi_shape[self._y_index]

        max_shape = np.prod(tile_shape)
        print('Tile shape: {}'.format(tile_shape))
        self.tile_shape = tile_shape

        num_tiles = {}
        tile_sample_index = {}
        czi_axes = self._czi_axes

        for seg, seg_meta, directory in self._get_segment_iterator():
            # Match the shape to the expected shape
            shape = tuple(s for s in directory.stored_shape if s != 1)
            rows, cols = shape[:2]

            # Look up the directory database to sort the channels
            directory_coords = {}
            for dimension_entry in directory.dimension_entries:
                if dimension_entry.dimension not in ('X', 'Y'):
                    directory_coords[dimension_entry.dimension] = dimension_entry.start
                    # FIXME, we probably need to reload all the segments when we update this
                    if dimension_entry.dimension not in czi_axes:
                        czi_axes += dimension_entry.dimension
            directory_coords = list(directory_coords.get(c, 0) for c in czi_axes)

            xpos, ypos, zpos = extract_seg_coordinates(seg_meta)
            timestamp = extract_seg_timestamp(seg_meta)

            # Find the real tile for the coordinates
            if self._tile_index is not None and self._tile_coord_tree is not None:
                _, tile_idx = self._tile_coord_tree.query(np.array([[xpos, ypos]]), k=1)
                tile_idx = int(tile_idx[0, 0])
                tile_sample_idx = directory_coords[self._tile_index]

                # Enforce that the tile index mapping is unique
                tile_sample_index.setdefault(tile_idx, []).append(tile_sample_idx)

            # Image formatted matrix so y then x
            num_x, num_y = num_tiles.get((xpos, ypos), (None, None))
            if num_y is None:
                num_y = int(np.ceil(y_shape / rows))
            else:
                if num_y != int(np.ceil(y_shape / rows)):
                    raise ValueError('Expected {} y-tiles, but got rows {} for shape {}'.format(num_y, rows, y_shape))
            if num_x is None:
                num_x = int(np.ceil(x_shape / cols))
            else:
                if num_x != int(np.ceil(x_shape / cols)):
                    raise ValueError('Expected {} x-tiles, but got cols {} for shape {}'.format(num_x, cols, x_shape))
            num_tiles[(xpos, ypos)] = (num_x, num_y)

            positions.append((xpos, ypos, zpos))
            position_timestamps.append(timestamp)
            position_index.append(tuple(directory_coords))

        # Make sure we got unique sample-tile mapping back
        final_tile_sample_index = {}
        bad_sample_index = []
        for tile_idx, tile_sample_indices in tile_sample_index.items():
            if len(tile_sample_indices) < 1:
                print('{}: No sample indicies'.format(tile_idx))
                bad_sample_index.append(tile_idx)
                continue

            # Histogram of tiles
            tile_dist = Counter(tile_sample_indices).most_common()
            top_index = tile_dist[0][0]

            if len(tile_dist) > 1 and any([c > 2 for _, c in tile_dist[1:]]):
                # Duplicates
                print('{}: Not unique {}'.format(tile_idx, tile_dist))
                bad_sample_index.append(tile_idx)
                continue

            # Just take the top index
            final_tile_sample_index[tile_idx] = top_index

        # Make sure we fit all the tile indices
        if bad_sample_index:
            raise ValueError('Got {} invalid tile indices: {}'.format(
                len(bad_sample_index), bad_sample_index))

        # Make sure the mapped indices are unique
        if len(final_tile_sample_index.values()) != len(set(final_tile_sample_index.values())):
            raise ValueError('Got non-unique tile indices: {}'.format(
                Counter(final_tile_sample_index.values())))

        # Remap the tile indices now that we know them uniquely
        position_index = np.array(position_index)
        if self._tile_index is not None:
            final_position_index = np.copy(position_index)
            for k, v in final_tile_sample_index.items():
                mask = position_index[:, self._tile_index] == k
                final_position_index[mask, self._tile_index] = v
            position_index = final_position_index

        if self._tile_index is not None:
            self._tile_sample_index = final_tile_sample_index
            position_counts = np.bincount(position_index[:, self._tile_index])
            print(f'Got sorted tile counts: {position_counts}')

        # Now that we've scanned the segments, reload the axes
        self._czi_axes = czi_axes
        self._load_axis_indices()

        # For some reason, the sub-tiles don't show up in the image shape??
        not_subtile = [i for i in range(position_index.shape[1])
                       if i not in (self._subtile_index, self._x_index, self._y_index)]
        unique_position_index = np.unique(position_index[:, not_subtile], axis=0)

        print('Got {} positions'.format(position_index.shape[0]))
        print('Got {} unique positions'.format(unique_position_index.shape[0]))

        if unique_position_index.shape[0] != max_shape:
            if not self.ignore_names:
                raise ValueError('Got weird tile counts {}, expected {}'.format(
                    unique_position_index.shape[0], max_shape))

        self._positions = np.array(positions)
        self._position_timestamps = position_timestamps
        self._position_index = position_index
        self._num_segments = len(positions)

    def _process_image(self, data: np.ndarray,
                       outfile: Optional[pathlib.Path] = None) -> np.ndarray:
        """ Process the image before writing it out

        :param ndarray data:
            The 2D greyscale image data
        :param Path outfile:
            If not None, the name of the output file (used for debug info only)
        :returns:
            The cropped, possibly contrast corrected image
        """

        if data.ndim != 2:
            raise ValueError('Expected 2D image, got {}: {}'.format(data.shape, outfile))

        # if self._frame_count == 0:
        #     print('Dtype: {}'.format(data.dtype))
        #     print('Rows, Cols: {},{}'.format(*data.shape))

        # Crop the image
        data = data[self._crop_y_st:self._crop_y_ed, self._crop_x_st:self._crop_x_ed]
        if data.shape[0] == 0 or data.shape[1] == 0:
            err = 'Invalid image shape after bbox x({},{}) y({},{}): {}x{}: {}'
            err = err.format(self._crop_x_st,
                             self._crop_x_ed,
                             self._crop_y_st,
                             self._crop_y_ed,
                             data.shape[0],
                             data.shape[1],
                             outfile)
            raise ValueError(err)

        # Fix the contrast
        if self._fix_contrast not in (None, 'raw'):
            data = image_utils.fix_contrast(data, mode=self._fix_contrast)
            # Force the final image to be 8-bit
            data = np.round(data*255)
            data[data < 0] = 0
            data[data > 255] = 255
            data = data.astype(np.uint8)

            # Convert to RGB
            data = np.stack([data, data, data], axis=2)

        if data.ndim not in (2, 3):
            raise ValueError('Invalid segment after processing {}: {}'.format(data.shape, outfile))
        return data

    def _save_image(self, outfile: pathlib.Path, data: np.ndarray):
        # Write the actual image
        outdir = outfile.parent
        try:
            outdir.mkdir(exist_ok=True, parents=True)
        except OSError:
            pass

        if outfile.is_file():
            raise OSError('Got duplicate image file: {}'.format(outfile))

        data = self._process_image(data)

        # Write the image to a file
        img = Image.fromarray(data)
        img.save(str(outfile))

    def format_image_name(self, index: int,
                          outdir: Optional[pathlib.Path] = None) -> Tuple[pathlib.Path, Dict]:
        """ Format the image name for a given tile index

        :param int index:
            The index of the tile in the CZI file
        :param Path outdir:
            If not None, the directory to write images under
        :returns:
            The outfile and the metadata dictionary for this index
        """
        # Work out the proper name for this file
        real_index = self._position_index[index, :]

        if self._subtile_index is None:
            multi_tiles = False
        else:
            multi_tiles = np.any(self._position_index[:, self._subtile_index] > 0)

        # Build up a format string for the channel dir, tile dir, and image file name
        channel_format = []
        tile_format = []
        image_format = []
        values = {}
        if self._channel_index is not None:
            channel_format.append('{channel:s}')
            image_format.append('{channel:s}-')
            values['channel'] = self._channel_names[real_index[self._channel_index]]

        if self._tile_index is not None:
            if self._czi_shape[self._tile_index] > 1:
                tile_format.append('s{tile:02d}')
                image_format.append('s{tile:02d}')
                values['tile'] = real_index[self._tile_index] + 1
                values['sample_tile'] = self._tile_sample_index[real_index[self._tile_index]] + 1
                # Try to label the tile with the name from ZEN
                tile_name = self._tile_names.get(real_index[self._tile_index])
                if tile_name not in (None, ''):
                    tile_format.append('-{tile_name:s}')
                    if hasattr(tile_name, 'get_name'):
                        tile_name = tile_name.get_name()
                    values['tile_name'] = str(tile_name)

        if self._slice_index is not None:
            if self._czi_shape[self._slice_index] > 1:
                image_format.append('z{slice:03d}')
                values['slice'] = real_index[self._slice_index] + 1

        if self._timepoint_index is not None:
            if self._czi_shape[self._timepoint_index] > 1:
                image_format.append('t{timepoint:03d}')
                values['timepoint'] = real_index[self._timepoint_index] + 1

        if self._multi_view_index is not None:
            if self._czi_shape[self._multi_view_index] > 1:
                image_format.append('v{multi_view:03d}')
                values['multi_view'] = real_index[self._multi_view_index] + 1

        if multi_tiles:
            image_format.append('m{multi_tile:02d}')
            values['multi_tile'] = real_index[self._subtile_index] + 1

        image_format.append('.tif')

        if outdir is None:
            outdir = pathlib.Path('.')
        if channel_format:
            outdir = outdir / ''.join(channel_format).format(**values)
        if tile_format:
            outdir = outdir / ''.join(tile_format).format(**values)
        outfile = outdir / ''.join(image_format).format(**values)

        return outfile, values

    def open(self):
        """ Create the output directories and paths """

        print('Reading from: {}'.format(self.infile))
        self._czi = CziFile(str(self.infile)).__enter__()

        self._frame_count = 0

    def close(self):
        """ Close the open file handles and zero all the metadata """

        if self._czi is not None:
            self._czi.__exit__(None, None, None)
            self._czi = None

    def set_crop(self, x_st: int, x_ed: int, y_st: int, y_ed: int):
        """ Set the crop parameters

        :param int x_st:
            The index of the column (dim1) to start the crop at (inclusive)
        :param int x_ed:
            The index of the column (dim1) to end the crop at (exclusive)
        :param int y_st:
            The index of the row (dim0) to start the crop at (inclusive)
        :param int y_ed:
            The index of the row (dim0) to end the crop at (exclusive)
        """
        self._crop_x_st = x_st
        self._crop_x_ed = x_ed

        self._crop_y_st = y_st
        self._crop_y_ed = y_ed

    def set_max_frames(self, max_frames: int):
        """ Set the maximum number of frames to dump

        :param int max_frames:
            If not None, the maximum frames to save
        """
        if max_frames is None or max_frames < 0:
            max_frames = -1
        self._max_frames = max_frames

    def set_contrast_correction(self, fix_contrast: str):
        """ Set the contrast fixing method

        :param str fix_contrast:
            The contrast mode to use (as accepted by :py:func:`~agg_dyn.utils.image_utils.fix_contrast`)
        """
        self._fix_contrast = fix_contrast

    def dump_all(self, rootdir: pathlib.Path,
                 outdir: str = 'RawData',
                 metadata_file: str = 'metadata.xml',
                 processes: int = 1):
        """ Dump all the data to rootdir

        :param Path rootdir:
            The directory to dump all the metadata and actual data to
        :param str outdir:
            The name of the output image directory to write under rootdir
        :param str metadata_file:
            The name of the metadata file to write under rootdir
        :param int processes:
            Number of parallel processes to run when dumping files
        """
        rootdir = pathlib.Path(rootdir).resolve()

        metadata_file = rootdir / metadata_file
        outdir = rootdir / outdir

        self.dump_metadata(metadata_file)
        self.dump_images(outdir, processes=processes)

    def dump_metadata(self, metadata_file: pathlib.Path):
        """ Dump the metadata from the czi file to an XML file

        :param Path metadata_file:
            The path to write the metadata to
        """
        # Dump the metadata
        print("MetadataSegment")
        write_metadata(metadata_file, self.czi_metadata)

    def dump_position_table(self, outfile: pathlib.Path):
        """ Write out the original space coordinates of each position in the CZI

        :param Path outfile:
            The CSV file to dump the positions to
        """
        column_order = [
            'index', 'filename', 'channel', 'tile', 'sample_tile', 'tile_name',
            'multi_tile', 'slice', 'timepoint', 'xpos', 'ypos', 'zpos', 'timestamp',
        ]

        def to_str(val):
            if val is None:
                return ''
            elif isinstance(val, int):
                return str(val)
            elif isinstance(val, float):
                return '{:0.4f}'.format(val)
            elif isinstance(val, datetime.datetime):
                return val.strftime('%Y-%m-%dT%H:%M:%S.%f')
            else:
                return str(val)

        # Make the directory (only exists sometimes)
        outfile.parent.mkdir(parents=True, exist_ok=True)

        # Write out an easy to understand position table so we never have to look at the metadata again
        print('Writing position table to: {}'.format(outfile))
        with outfile.open('wt') as fp:
            fp.write(','.join(column_order) + '\n')
            for i in range(len(self._positions)):
                filename, values = self.format_image_name(index=i, outdir=None)
                xpos, ypos, zpos = self._positions[i]
                timestamp = self._position_timestamps[i]
                values.update({
                    'index': i,
                    'filename': filename,
                    'xpos': xpos,
                    'ypos': ypos,
                    'zpos': zpos,
                    'timestamp': timestamp,
                })
                fp.write(','.join(to_str(values.get(c)) for c in column_order) + '\n')

    def _maybe_save_image(self, item: Tuple) -> bool:
        # Dump a single segment to the filesystem, inside the parallel process
        i, data, outdir = item
        try:
            outfile, _ = self.format_image_name(index=i, outdir=outdir)
            print(outfile)
            self._save_image(outfile, data)
        except Exception:
            print('Error processing frame {}'.format(i))
            traceback.print_exc()
            return False
        return True

    def iter_images(self) -> Generator:
        """ Return an interator of images """

        self._load_frame_tables()
        self._load_position_table()

        for i, seg in enumerate(self.position_segments):
            if self._max_frames > 0 and i >= self._max_frames:
                break
            outfile, metadata = self.format_image_name(i)
            data = self._process_image(seg, outfile=outfile)
            yield outfile, metadata, data

    def dump_images(self, outdir: pathlib.Path, processes: int = 1):
        """ Dump all the images to a nested directory heirarchy

        :param Path outdir:
            The directory to save everything under
        :param int processes:
            Number of parallel processes to use to dump the image
        """
        if outdir.is_dir():
            print('Overwriting {}'.format(outdir))
            shutil.rmtree(str(outdir))
        outdir.mkdir(parents=True)

        self._load_frame_tables()
        self._load_position_table()
        self.dump_position_table(outdir.parent / 'positions.csv')

        if self._max_frames == 0:
            print('Skipping processing frames...')
            return

        # Clear the file handles and caches so they don't get copied
        # Use a lazy iterator to prevent crashy crashy
        items = ((i, seg, outdir) for (i, seg) in enumerate(self.position_segments)
                 if self._max_frames < 0 or i < self._max_frames)

        def initializer(self):
            if processes > 1:
                self._czi = None

        # Dump the files in parallel
        print('Processing {} images'.format(self._num_segments))
        with parallel_utils.Hypermap(processes=processes,
                                     lazy=False,
                                     initializer=initializer,
                                     initargs=(self, )) as proc:
            res = proc.map(self._maybe_save_image, items)
        print('Processed {} images successfully'.format(sum(res)))

    def print_tile_metadata(self):
        """ Dump out the tile metadata """

        assert self._num_segments == self._position_index.shape[0]

        # Count up how many frames are under each tile for each channel
        tile_metadata = {}
        for i in range(self._num_segments):
            real_index = self._position_index[i, :]
            if self._channel_index is None:
                channel_name = 'DefaultChannel'
            else:
                channel_name = self._channel_names[real_index[self._channel_index]]
            if self._tile_index is None:
                tile_name = 's00'
            else:
                tile_index = real_index[self._tile_index]
                tile_name = 's{:02d}'.format(tile_index + 1)
                # Try to label the tile with the name from ZEN
                cond_name = self._tile_names.get(tile_index)
                if cond_name not in (None, ''):
                    tile_name += '-{}'.format(cond_name.get_name())
            tile_metadata.setdefault(channel_name, {}).setdefault(tile_name, 0)
            tile_metadata[channel_name][tile_name] += 1

        print('\n\n##### Final tiles #####')
        for channel in sorted(tile_metadata):
            print('* {}'.format(channel))
            for tile in sorted(tile_metadata[channel]):
                print('\t* {}: {}'.format(tile, tile_metadata[channel][tile]))

        print('\n\n##### Tile Order #####')
        for old_tile_index in sorted(self._tile_names):
            print('* {}: {}'.format(old_tile_index+1, self._tile_names[old_tile_index]))


# Functions


def write_metadata(metadata_file: pathlib.Path, metadata: etree):
    """ Write the CZI metadata to an XML file

    :param Path metadata_file:
        Path to the metadata file to write
    :param ETree metadata:
        The XML object to write (from ``czi.metadata``)
    """
    metadata_file.parent.mkdir(exist_ok=True, parents=True)
    with metadata_file.open('wb') as fp:
        fp.write(etree.tostring(metadata))


def read_metadata(metadata_file: pathlib.Path) -> etree:
    """ Read in the metadata for this CZI file

    :param Path metadata_file:
        Path to the metadata file to read
    :returns:
        An ETree with the CZI metadata
    """
    return etree.parse(str(metadata_file))


def find_metadata(rootdir: pathlib.Path,
                  filename: str = 'metadata.xml') -> pathlib.Path:
    """ Find the metadata file

    :param Path rootdir:
        Path to the root directory to load
    :param str filename:
        Name of the metadata file that was written
    :returns:
        The path to the metadata file
    """
    if (rootdir / filename).is_file():
        return rootdir / filename
    for subdir in rootdir.iterdir():
        if not subdir.is_dir():
            continue
        if (subdir / filename).is_file():
            return subdir / filename
    raise OSError(f'No {filename} file found under {rootdir}')


def extract_space_scale(metadata: etree) -> Dict[str, float]:
    """ Extract the space scale for the images

    :param metadata:
        The metadata etree object (rooted at the ImageDocument node)
    :returns:
        The image scale factor for microns per pixel for each dimension
    """
    nodes = metadata.findall(SPACE_SCALE_XPATH)
    space_scale = {}
    for node in nodes:
        node_id = node.attrib['Id']
        node_val = None
        for child_node in node:
            if child_node.tag == 'Value':
                assert node_val is None
                node_val = float(child_node.text.strip())*1e6
        assert node_val is not None
        space_scale[node_id] = node_val
    return space_scale


def extract_channel_names(metadata: etree) -> Dict[int, str]:
    """ Find the channel names in the metadata soup

    :param metadata:
        The metadata etree object (rooted at the ImageDocument node)
    :returns:
        A dictionary mapping channel number to channel name
    """
    nodes = metadata.findall(CHANNEL_XPATH)
    channel_names = {}
    for node in nodes:
        node_id = node.attrib['Id']
        node_name = node.attrib.get('Name', node_id)

        if not node_id.startswith('Channel:'):
            continue
            raise ValueError(f'Got bad node id: {node_id}')
        node_id = int(node_id[len('Channel:'):])
        if node_id in channel_names:
            raise ValueError(f'Duplicate node id: {node_id}')
        channel_names[node_id] = node_name
    return channel_names


def extract_tile_names(metadata: etree) -> Dict[int, TileData]:
    """ Find the tile names in the metadata soup

    :param metadata:
        The metadata etree object (rooted at the ImageDocument node), like that
        returned by :py:func:`read_metadata`
    :returns:
        A dictionary mapping tile number to :py:class:`TileData` objects
    """
    tile_names = {}

    # Load all the single tile data points
    for single_tile_xpath in (SINGLE_TILE_XPATH, SINGLE_TILE_XPATH_TS):
        offset = len(tile_names)
        nodes = metadata.findall(single_tile_xpath)
        for i, node in enumerate(nodes):
            tileid = node.attrib['Id']
            name = node.attrib.get('Name', tileid)

            if node.find('IsUsedForAcquisition').text.lower().strip() != 'true':
                continue

            # Simple, normal tiles with no tricky bits
            if node.find('X') is None:
                x = 0.0
            else:
                x = float(node.find('X').text)
            if node.find('Y') is None:
                y = 0.0
            else:
                y = float(node.find('Y').text)
            if node.find('Z') is None:
                z = 0.0
            else:
                z = float(node.find('Z').text)

            tile_names[i+offset] = TileData(i+offset, name, tileid, x, y, z, rows=1, cols=1)

    # FIXME: We probably also need to look at
    # '.', 'Metadata', 'Experiment', 'ExperimentBlocks', 'AcquisitionBlock',
    # 'SubDimensionSetups', 'TimeSeriesSetup', 'SubDimensionSetups',
    # 'RegionsSetup', 'SampleHolder', 'PositionedRegionsScanMode'

    # Load all the multi-tile data points
    for multi_tile_xpath in (MULTI_TILE_XPATH, MULTI_TILE_XPATH_TS):
        offset = len(tile_names)
        nodes = metadata.findall(multi_tile_xpath)
        for i, node in enumerate(nodes):
            tileid = node.attrib['Id']
            name = node.attrib.get('Name', tileid)

            if node.find('IsUsedForAcquisition').text.lower().strip() != 'true':
                continue

            # Unpack the tile arrangement
            if node.find('CenterPosition') is None:
                x = y = 0.0
            else:
                x, y = node.find('CenterPosition').text.split(',')
                x = float(x.strip())
                y = float(y.strip())

            if node.find('Z') is None:
                z = 0.0
            else:
                z = float(node.find('Z').text)

            if node.find('Rows') is None:
                rows = 1
            else:
                rows = int(node.find('Rows').text)  # How many tiles in y are there
            if node.find('Columns') is None:
                cols = 1
            else:
                cols = int(node.find('Columns').text)  # How many tiles in x are there

            tile_names[i+offset] = TileData(i+offset, name, tileid, x, y, z, rows=rows, cols=cols)

    return tile_names


def extract_seg_coordinates(seg_metadata):
    """ Extract the segment coordinates

    :param seg_metadata:
        The etree object for the segment metadata
    :returns:
        The x, y, z coordinates of the tile
    """
    if seg_metadata is None:
        return 0.0, 0.0, 0.0

    if isinstance(seg_metadata, str):
        seg_metadata = etree.fromstring(seg_metadata)
    if isinstance(seg_metadata, dict):
        if 'Tags' in seg_metadata:
            seg_metadata = seg_metadata['Tags']
        xpos = seg_metadata.get('StageXPosition')
        ypos = seg_metadata.get('StageYPosition')
        zpos = seg_metadata.get('FocusPosition')
        xoff = seg_metadata.get('RoiCenterOffsetX')
        yoff = seg_metadata.get('RoiCenterOffsetY')
    else:
        xpos = seg_metadata.find('./Tags/StageXPosition')
        ypos = seg_metadata.find('./Tags/StageYPosition')
        zpos = seg_metadata.find('./Tags/FocusPosition')
        xoff = seg_metadata.find('./Tags/RoiCenterOffsetX')
        yoff = seg_metadata.find('./Tags/RoiCenterOffsetY')

        if xpos is not None:
            xpos = float(xpos.text)
        if ypos is not None:
            ypos = float(ypos.text)
        if zpos is not None:
            zpos = float(zpos.text)
        if xoff is not None:
            xpos += float(xoff.text)
        if yoff is not None:
            ypos += float(yoff.text)
    if xpos is not None and xoff is not None:
        xpos += xoff
    if ypos is not None and yoff is not None:
        ypos += yoff
    return xpos, ypos, zpos


def extract_seg_timestamp(seg_metadata):
    """ Extract the segment acquisition time

    :param seg_metadata:
        The etree object for the segment metadata
    :returns:
        The timestamp for the tile
    """
    if seg_metadata is None:
        return None
    if isinstance(seg_metadata, str):
        seg_metadata = etree.fromstring(seg_metadata)
    if isinstance(seg_metadata, dict):
        if 'Tags' in seg_metadata:
            seg_metadata = seg_metadata['Tags']
        timestamp = seg_metadata.get('AcquisitionTime')
    else:
        timestamp = seg_metadata.find('./Tags/AcquisitionTime')
        if timestamp is not None:
            timestamp = timestamp.text
    # Convert the timestamp to a datetime
    if timestamp is not None:
        timestamp = timestamp.strip()
        if '.' in timestamp:
            timestamp, microseconds = timestamp.rsplit('.', 1)
            if len(microseconds) < 6:
                microseconds += '0' * (6 - len(microseconds))
            microseconds = microseconds[:6]
            timestamp += '.' + microseconds
        timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')
    return timestamp


def extract_comb_pattern(metadata):
    """ Extract the tile comb pattern

    :param metadata:
        The metadata etree object (rooted at the ImageDocument node), like that
        returned by :py:func:`read_metadata`
    :returns:
        The comb pattern for multi-tile acquisitions
    """
    node = metadata.find(MULTI_TILE_COMB_XPATH)
    if node is not None:
        return node.text
    node = metadata.find(MULTI_TILE_COMB_XPATH_TS)
    if node is not None:
        return node.text
    return None


def decompress_raw_formats(raw_img):
    """ Decompress an undefined RAW format

    :param bytes raw_img:
        The bytes of a raw image
    :returns:
        A numpy array with the image properly decompressed
    """
    return np.frombuffer(raw_img, dtype=np.uint16)


# Monkey Patches

DECOMPRESS[1000] = decompress_raw_formats
DECOMPRESS[1001] = decompress_raw_formats
