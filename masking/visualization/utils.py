import numpy as np


def make_blended_heatmap(image, heatmap):

    heatmap_clipped = np.clip(heatmap, 0.0, 1.0)
    heatmap_normalized = (heatmap_clipped - np.min(heatmap_clipped)) / (np.max(heatmap_clipped) - np.min(heatmap_clipped))
    heatmap_stacked = np.stack([heatmap_normalized, heatmap_normalized, heatmap_normalized], axis=2)
    superposed = image.astype(np.float32) * heatmap_stacked + np.array([255, 255, 255], np.float32) * (1.0 - heatmap_stacked)
    superposed = superposed.astype(np.uint8)

    return superposed
