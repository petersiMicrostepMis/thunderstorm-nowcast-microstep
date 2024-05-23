# -*- coding: utf-8 -*-
import unittest
import ai4eosc_thunder_nowcast_ml.models.api as api

class TestModelMethods(unittest.TestCase):

    def setUp(self):
        self.meta = api.get_metadata()

    def test_model_metadata_type(self):
        """
        Test that get_metadata() returns dict
        """
        self.assertTrue(type(self.meta) is dict)

    def test_model_metadata_values(self):
        """
        Test that get_metadata() returns right values (subset)
        """
        self.assertEqual(self.meta['name'].lower().replace('-','_'),
                        'ai4eosc_thunder_nowcast_ml'.lower().replace('-','_'))
        self.assertEqual(self.meta['author'].lower(),
                         'MicroStep-MIS'.lower())
        self.assertEqual(self.meta['license'].lower(),
                         'MIT'.lower())


if __name__ == '__main__':
    unittest.main()
