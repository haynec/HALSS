import unittest

import numpy as np

from HALSS.classes import halss_data_packet, parameters


class LandingSiteCoordinatesTest(unittest.TestCase):
    def setUp(self):
        self.packet = halss_data_packet()
        self.packet.num_sites = 1
        self.packet.surf_norm = np.zeros((21, 21, 3), dtype=np.uint8)
        self.packet.safety_map = np.zeros_like(self.packet.surf_norm)
        self.packet.safety_map[6:15, 6:15] = 255

        # A planar support cloud gives unit scale in both map directions.
        x, y = np.meshgrid(np.arange(22.0), np.arange(22.0))
        self.packet.pcd_global = np.column_stack(
            (x.ravel(), y.ravel(), np.zeros(x.size))
        )
        self.packet.find_NED_origin_uv()

    def test_returned_circle_excludes_unsafe_pixel_centers(self):
        self.packet.find_landing_site()
        center = self.packet.center_coords_uv[0]
        unsafe = np.argwhere(self.packet.safety_map[:, :, 0] == 0)
        clearance = np.linalg.norm(unsafe - center, axis=1).min()

        self.assertGreaterEqual(clearance, self.packet.radii_uv[0])
        np.testing.assert_array_equal(center, [10, 10])

    def test_radius_is_unchanged_when_the_safety_map_is_unchanged(self):
        self.packet.find_landing_site()
        self.assertEqual(self.packet.radii_ned[0], 5.0)

        self.packet.update_landing_site(parameters())

        self.assertEqual(self.packet.radii_ned[0], 5.0)

    def test_all_unsafe_map_preserves_the_zero_radius_placeholder(self):
        self.packet.safety_map.fill(0)

        self.packet.find_landing_site()

        self.assertEqual(self.packet.radii_uv[0], 0)
        np.testing.assert_array_equal(self.packet.center_coords_uv[0], [0, 0])
