from numpy import linag as LA
import numpy as np
import math

def p2pmerror(gtld, pld):
    """normalized point to point mean square error,
    from this paper: The Menpo Benchmark for Multi-pose 2D and 3D Facial Landmark Localisation and Tracking"""

    # gth: ground truth landmarks
    # pld: predicted landmarks
    n = gtld.shape[0]
    dscale = LA.norm([np.amax(gtld[:, 0]) - np.amin(gtld[:, 0]),
                      np.amax(gtld[:, 1]) - np.amin(gtld[:, 1])])
    dis = LA.norm(gtld - pld) / math.sqrt(n) / dscale
    return dis