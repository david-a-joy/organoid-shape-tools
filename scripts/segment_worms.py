#!/usr/bin/env python3

""" Segment the growing neural worms

Segment all the worms in a CZI file:

.. code-block:: bash

    $ ./segment_worms.py /data/NeuroWorm-48hrs-2019-08-31.czi

Segment the worms in the newer CZI tracking file:

.. code-block:: bash

    $ ./segment_worms.py ~/data/2020-01-22-Tracking/2020-01-22/2020-01-22-NeuroWorm-48hr-d3-d5.czi

To analyze the results, see ``analyze_segment_worms.py``

"""

# Standard lib
import argparse
import pathlib
import sys
from typing import Optional

THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
if (BASEDIR / 'organoid_shape_tools').is_dir():
    sys.path.insert(0, str(BASEDIR))

# Our own imports
from organoid_shape_tools import WormSegmentation

# Main Function


def segment_worms(czifile: pathlib.Path,
                  outdir: Optional[pathlib.Path] = None,
                  overwrite: bool = False):
    """ Segment the worms from brightfield imaging

    :param Path czifile:
        Path to the raw CZI file to segment
    """
    if outdir is None:
        outdir = czifile.parent / czifile.stem

    # Track using background segmentation
    with WormSegmentation(czifile,
                          outdir=outdir,
                          overwrite=overwrite) as proc:
        proc.set_params()
        proc.create_outdir()

        proc.load_params()
        proc.save_params()

        # proc.estimate_background()
        # proc.find_delta_rois()
        # proc.subset_tiles_rois()
        # proc.subset_tiles()
        # proc.segment_all_tile_foregrounds(STD_REPLOT)

        #proc.segment_all_frames(range(49, 61))

        # New New Segmentation
        # proc.find_delta_segments()
        proc.plot_final_segmentations()
        # proc.gather_final_segmentations()

# Command line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type=pathlib.Path,
                        help='Directory to write the files to')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite the cached data')
    parser.add_argument('czifile', type=pathlib.Path)
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    segment_worms(**vars(args))


if __name__ == '__main__':
    main()
