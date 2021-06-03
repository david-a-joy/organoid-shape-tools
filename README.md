# Organoid Shape and Tracking Toolbox

Track the morphology of elongating organoids using brightfield microscopy.

If you find this code useful, please cite:

> Libby*, A., Joy*, D., Elder, N., Bulger, E., Krakora, M., Gaylord, E.,
> Mendoza-Camacho, F., Butts, J., and McDevitt, T. (2020). Axial Elongation of
> Caudalized Human Organoids Mimics Neural Tube Development. BioRxiv.

## Installing

This script requires Python 3.8 or greater and several additional python packages.
This code has been tested on Ubuntu 20.04, but may work with modifications on
other systems.

It is recommended to install and test the code in a virtual environment for
maximum reproducibility:

```{bash}
# Create the virtual environment
python3 -m venv ~/shape_env
source ~/shape_env/bin/activate
```

All commands below assume that `python3` and `pip3` refer to the binaries installed in
the virtual environment. Commands are executed from the base of the git repository
unless otherwise specified.

```{bash}
pip3 install --upgrade pip wheel setuptools
pip3 install numpy cython

# Install the required packages
pip3 install -r requirements.txt

# Build and install all files in the deep hiPSC tracking package
python3 setup.py build_ext --inplace
cd scripts
```

Commands can then be run from the ``scripts`` directory of the git repository.

## Scripts provided by the package

This repository provides two independent aggregate segmentation pipelines, one
for time lapse phase images of elongating organoids, and one for segmenting
static phase images of elongating organoids.

### Time lapse segmentation

Time lapse movies of elongating organoids such as those shown in Figures 1D and 6A
can be segmented using the main script:

* `segment_worms.py`: Run a multi-stage aggregate segmentation pipeline over time lapse phase imaging

Temporal shape analysis for these segmented movies can then be performed using:

* `analyze_segment_worms.py`: Analyze the shape changes of individual aggregates over time

Time lapse images were acquired in an incubated Zeiss Axio Observer microscope over 48 hours. See Libby et al for details.

### Static snapshot segmentation

Static snapshots of elongating organoids such as those shown in Figures 1C, 3A, 8D, and 8F
can be segmented using the main script:

* `analyze_aggregate_shape.py`: Run a single stage aggregate segmentation pipeline over static images

The subsequent shape analysis used in the paper can then be performed using:

* `merge_analyze_aggregate_shape.py`: Merge the statistics for folders of segmented aggregates
* `plot_analyze_aggregate_shape.py`: Plot the statistics for aggregate elongation
* `plot_analyze_aggregate_hands.py`: Plot the convexity analysis from Figure 8

Phase images were acquired on an EVOS Cell Imaging system as described in Libby et al.

Additionally, the following scripts can be used to reproduce the segmentation of histology
sections such as those shown in Figure 4C, 6D, 6E, and 8H:

* `quant_edu_stains.py`: Quantify the percent of Edu or PH3+ cells in a section
* `quant_tbxt_stains.py`: Quantify the percent of TBXT+/SOX2+ cells in a section

Sections were stained and imaged as described in Libby et al.

## Testing

The modules defined in `organoid_shape_tools` have a test suite that can be run
using the `pytest` package.

```{bash}
python3 -m pytest tests
```

## Documentation

Documentation for the scripts and individual modules can be built using the
`sphinx` package.

```{bash}
cd docs
make html
```

Documentation will then be available under `docs/_build/html/index.html`
