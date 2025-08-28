
import numpy as np
from scipy.ndimage import distance_transform_edt

def hd95(binary_pred, binary_gt, spacing=(1.0,1.0,1.0)):
    # binary_pred, binary_gt: numpy arrays (D,H,W) of {0,1}
    # compute surface distances both ways
    def surface_dist(a, b):
        a = a.astype(bool); b = b.astype(bool)
        if a.sum()==0 and b.sum()==0: return 0.0
        if a.sum()==0 or b.sum()==0: return np.inf
        a_edge = a ^ binary_erosion(a)
        b_edge = b ^ binary_erosion(b)
        dt = distance_transform_edt(~b, sampling=spacing)
        dists = dt[a_edge]
        return dists
    from scipy.ndimage import binary_erosion
    da = surface_dist(binary_pred, binary_gt)
    db = surface_dist(binary_gt, binary_pred)
    if np.isinf(da).any() or np.isinf(db).any(): return float("inf")
    all_d = np.concatenate([da, db])
    if all_d.size==0: return 0.0
    return np.percentile(all_d, 95)
