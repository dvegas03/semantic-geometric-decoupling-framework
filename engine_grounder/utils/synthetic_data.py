# Synthetic flat/bowl-shaped depth map generator for pipeline tests without real sensor data.

import numpy as np


class SyntheticDataGenerator:
    def __init__(self, size=(100, 100), true_depth=0.5):
        if true_depth <= 0:
            raise ValueError(f"true_depth must be positive, got {true_depth}")
        self.size = size
        self.true_depth = true_depth

    def generate_noisy_crop(self, void_ratio=0.60, noise_std=0.1, seed=None):
        if not 0.0 <= void_ratio <= 1.0:
            raise ValueError(f"void_ratio must be in [0, 1], got {void_ratio}")

        rng = np.random.default_rng(seed)

        noise = rng.normal(0, noise_std, self.size)
        depth_map = np.full(self.size, self.true_depth) + noise

        np.clip(depth_map, 0.0, None, out=depth_map)

        void_pixels = int(self.size[0] * self.size[1] * void_ratio)
        if void_pixels > 0:
            flat_map = depth_map.flatten()
            void_indices = rng.choice(flat_map.size, void_pixels, replace=False)
            flat_map[void_indices] = 0.0
            depth_map = flat_map.reshape(self.size)

        return depth_map
