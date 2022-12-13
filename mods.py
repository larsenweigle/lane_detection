""" mods.py

Implements simple modifications to the Canny edge detector to speed up the algorithm
by only considering a certain subset of pixels that match colors required for road 
lane markings. 
"""
import numpy as np

from edge import * 

def modified_canny(img, lane_indicies, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    kernel = gaussian_kernel(kernel_size, sigma) 
    smoothed = conv(img, kernel)
    G, theta = gradient(smoothed)
    nms = modified_non_maximum_suppression(G, theta, lane_indicies)
    strong_edges, weak_edges = modified_double_thresholding(nms, high, low, lane_indicies)
    edge = link_edges(strong_edges, weak_edges)

    return edge

def modified_non_maximum_suppression(G, theta, lane_indicies):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    theta = (theta % 360.0).astype(np.int32)
    
    def get_grad_values(grad_dir, G, i, j):
        if grad_dir == 0 or grad_dir == 180 or grad_dir == 360:
            return G[i,j-1], G[i,j+1]
        elif grad_dir == 45 or grad_dir == 225:
            return G[i-1,j-1], G[i+1,j+1]
        elif grad_dir == 90 or grad_dir == 270:
            return G[i-1,j], G[i+1,j]
        else:
            return G[i-1,j+1], G[i+1,j-1]

    padded_G = np.pad(G, 1, mode='constant')

    visited_indices = set()
    for i in range(len(lane_indicies)):
        ys, xs = lane_indicies[i]
        for x, y in zip(xs, ys):
            if (x, y) not in visited_indices:
                grad_dir = theta[y,x]
                val = G[y,x]
                pos_grad_val, neg_grad_val = get_grad_values(grad_dir, padded_G, y+1, x+1)
                
                if val > pos_grad_val and val > neg_grad_val:
                    out[y,x] = val
            
            visited_indices.add((x, y))

    return out

def modified_double_thresholding(img, high, low, lane_indicies):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool) #np.bool
    weak_edges = np.zeros(img.shape, dtype=np.bool) # np.bool

    visited_indices = set()
    for i in range(len(lane_indicies)):
        ys, xs = lane_indicies[i]
        for x, y in zip(xs, ys):
            if (x, y) not in visited_indices:
                val = img[y,x]
                if val > high:
                    strong_edges[y,x] = 1
                elif val > low:
                    weak_edges[y,x] = 1

        visited_indices.add((x, y))

    return strong_edges, weak_edges