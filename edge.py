""" edge.py

Builds out the necessary functions to implement a Canny edge filter and Hough Transform 
function for straight lines. Together, they can be used to build a lane line detection 
algorithm as demonstrated in detection.py. 
"""

import numpy as np
import matplotlib.pyplot as plt


def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    kernel = np.flip(kernel)

    for r in range(Hi):
        for c in range(Wi):
            # calculate convolution at location (r, c) in the output image
            weighted_neighbors = kernel * padded[r:r+Hk,c:c+Wk]
            out[r,c] = np.sum(weighted_neighbors)

    return out


def zero_pad(image, pad_height, pad_width):
    """ Pads an input image with zeros for processing. 

    Args:
        image: input image to be padded of shape (H, W).
        pad_height: the number of zero filled rows we will stack to the 
        bottom on top of the input image.
        pad_width: the number of zero filled columns we will appened to the 
        left and right of the input image. 

    Returns:
        out: the resulting padded image.
    """
    H, W = image.shape
    out = None

    col_pad, row_pad = np.zeros((H, pad_width)), np.zeros((pad_height, W + 2 * pad_width))
    image_col_padded = np.concatenate([np.concatenate([col_pad, image], axis=1), col_pad], axis=1)
    out = np.concatenate([np.concatenate([row_pad, image_col_padded], axis=0), row_pad], axis=0)

    return out


def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """
    kernel = np.zeros((size, size))

    k = (size) // 2
    for i in range(size):
        for j in range(size):
            kernel[i,j] = (1/(2*np.pi*sigma**2)) * np.exp(-((i-k)**2+(j-k)**2)/(2*sigma**2))

    return kernel


def partial_x(img):
    """ Computes partial x-derivative of input img.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """
    x_dir_kernel = 1/2 * np.array([[1,0 ,-1]])
    out = conv(img, x_dir_kernel)
    
    return out


def partial_y(img):
    """ Computes partial y-derivative of input img.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """
    y_dir_kernel = 1/2 * np.array([[1,0 ,-1]]).T
    out = conv(img, y_dir_kernel)

    return out


def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    G_x, G_y = partial_x(img), partial_y(img)
    G = np.sqrt(np.add(G_x**2, G_y**2))
    theta = np.arctan2(G_y, G_x) * (180/np.pi)
    theta %= 360 

    return G, theta


def non_maximum_suppression(G, theta):
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

    for i in range(H):
        for j in range(W):
            grad_dir = theta[i,j]
            val = G[i,j]
            pos_grad_val, neg_grad_val = get_grad_values(grad_dir, padded_G, i+1, j+1)

            if val > pos_grad_val and val > neg_grad_val:
                out[i,j] = val

    return out


def double_thresholding(img, high, low):
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

    strong_edges = np.zeros(img.shape, dtype=np.bool) 
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            val = img[i,j]
            if val > high:
                strong_edges[i,j] = 1
            elif val > low:
                weak_edges[i,j] = 1

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors


def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    visited_edges = set()
    for strong_indices in indices:
        queue = []
        queue.append(strong_indices)
        # bfs pn weak edges starting with a strong edge
        while len(queue): 
            curr_row, curr_col = queue.pop(0)
            edges[curr_row, curr_col] = 1
            curr_neighbors = get_neighbors(curr_row, curr_col, H, W)
            for neighbor in curr_neighbors: 
                neighbor_row, neighbor_col = neighbor
                if neighbor not in visited_edges and weak_edges[neighbor_row, neighbor_col]:
                    queue.append(neighbor)
                visited_edges.add((neighbor_row, neighbor_col))

    return edges


def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
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
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    for x, y in zip(xs, ys):
        curr_rhos = np.round(x * (cos_t) + (y * sin_t))
        for theta_val_idx in range(num_thetas):
            rho_val_idxs = np.where(rhos==curr_rhos[theta_val_idx])
            accumulator[rho_val_idxs[0][0], theta_val_idx] += 1

    return accumulator, rhos, thetas