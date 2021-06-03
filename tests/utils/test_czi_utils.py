# Imports
import unittest
import textwrap

# 3rd party
try:
    from lxml import etree
except ImportError:
    from xml.etree import ElementTree as etree

# Our own imports
from organoid_shape_tools.utils import czi_utils
from ..helpers import FileSystemTestCase

# Tests


class TestFindMetadata(FileSystemTestCase):
    """ Make sure the metadata finder works """

    def test_fails_on_empty_dir(self):

        with self.assertRaises(OSError):
            czi_utils.find_metadata(self.tempdir)

    def test_finds_rootdir(self):

        metadata_file = self.tempdir / 'metadata.xml'
        metadata_file.touch()

        res = czi_utils.find_metadata(self.tempdir)
        self.assertEqual(res, metadata_file)

    def test_finds_subdir(self):

        d1 = self.tempdir / '1'
        d2 = self.tempdir / '2'

        d1.mkdir()
        d2.mkdir()

        metadata_file = d2 / 'metadata.xml'
        metadata_file.touch()

        res = czi_utils.find_metadata(self.tempdir)
        self.assertEqual(res, metadata_file)


class TestExtractTileNames(unittest.TestCase):

    def test_works(self):
        self.assertIsNotNone(czi_utils.extract_tile_names)


class TestExtractChannelNames(unittest.TestCase):

    def test_works(self):
        self.assertIsNotNone(czi_utils.extract_channel_names)


class TestExtractSpaceScale(unittest.TestCase):

    def test_extracts_from_metadata(self):

        metadata = textwrap.dedent("""
            <ImageDocument>
            <Metadata>
            <Scaling>
              <AutoScaling>
                <Type>Measured</Type>
                <Objective>Objective.000000-1020-863</Objective>
                <Optovar>Aquila.Tubelens 1.0x ext.</Optovar>
                <Reflector>Reflector.none</Reflector>
                <CameraAdapter>YokogawaCameraAdapter.1.2x</CameraAdapter>
                <ObjectiveName>A-Plan 10x/0.25 Ph 1 Var 1</ObjectiveName>
                <OptovarMagnification>1</OptovarMagnification>
                <ReflectorMagnification>1</ReflectorMagnification>
                <CameraName>95B</CameraName>
                <CameraAdapterMagnification>1.2</CameraAdapterMagnification>
                <CameraPixelDistance>11,11</CameraPixelDistance>
                <CreationDateTime>06/07/2020 16:01:43</CreationDateTime>
              </AutoScaling>
              <Items>
                <Distance Id="X">
                  <Value>9.1666666666666664E-07</Value>
                  <DefaultUnitFormat>&#181;m</DefaultUnitFormat>
                </Distance>
                <Distance Id="Y">
                  <Value>9.1666666666666664E-07</Value>
                  <DefaultUnitFormat>&#181;m</DefaultUnitFormat>
                </Distance>
              </Items>
            </Scaling>
            </Metadata>
            </ImageDocument>
        """).strip()
        metadata = etree.fromstring(metadata)

        res = czi_utils.extract_space_scale(metadata)
        exp = {'X': 0.91666666, 'Y': 0.91666666}

        self.assertEqual(res.keys(), exp.keys())
        for key, val in res.items():
            self.assertAlmostEqual(val, exp[key])
