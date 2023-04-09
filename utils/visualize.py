import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import constants

def visualize_segmentation(
    img_3d: np.ndarray,
    seg_3d: np.ndarray,
    slice_idx: int,
    colormap: str="tab20"
) -> np.ndarray:
    img_2d = img_3d[:,:,slice_idx] / img_3d.max()
    seg_2d = seg_3d[:,:,slice_idx]

    cmap = mpl.colormaps[colormap].resampled(constants.NUM_CLASSES-1)
    colors = np.vstack([np.ones(3), cmap.colors[:,:3]])

    overlayed = img_2d[:,:,None] * colors[seg_2d.astype(int),:3]

    return overlayed
