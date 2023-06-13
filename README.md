# Canny Lane Detection Project

This project involves building a Canny edge detector and Hough Transform function for straight lines. The two are combined to construct a lane line detection algorithm. 

## Overview

The project includes a Python script, `edge.py`, that contains functions for various stages of the edge detection process.

## Prerequisites

To run the script, you need Python 3.x and the following packages:

- Numpy
- Matplotlib

## Installation

Clone the repository with the following command: git clone https://github.com/your_username/your_repository.git

Navigate to the cloned directory: cd your_repository

## How to Run

You can run the script in a Python environment: python edge.py

## Functions

- `conv(image, kernel)`: Implements a convolution filter.  
- `zero_pad(image, pad_height, pad_width)`: Pads an input image with zeros for processing.
- `gaussian_kernel(size, sigma)`: Implements a Gaussian Kernel.
- `partial_x(img)`: Computes partial x-derivative of input img.
- `partial_y(img)`: Computes partial y-derivative of input img.
- `gradient(img)`: Returns gradient magnitude and direction of input img.
- `non_maximum_suppression(G, theta)`: Performs non-maximum suppression.
- `double_thresholding(img, high, low)`: Identifies strong edges and weak edges.
- `get_neighbors(y, x, H, W)`: Return indices of valid neighbors of (y, x).
- `link_edges(strong_edges, weak_edges)`: Finds weak edges connected to strong edges and links them.
- `canny(img, kernel_size=5, sigma=1.4, high=20, low=15)`: Implements the canny edge detector.
- `hough_transform(img)`: Transforms points in the input image into Hough space.

## Algorithm

1. An image is passed through a Gaussian filter (kernel size and sigma are configurable) for noise reduction.
2. Compute the gradient intensity representations of the image.
3. Non-maximum suppression is applied to thin out the edges.
4. Double threshold is applied to determine potential edges.
5. Finally, we track edges by Hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.

The output is a binary image with white pixels tracing out the detected edges and black elsewhere.

## Future Work

I am currently developing a lightweight Convolutional Neural Network (CNN) designed to identify sub-sections of an image where a lane line is present. This binary classifier will significantly enhance the efficiency of our edge detection algorithm by isolating its application to areas identified by the CNN, thereby reducing runtime.

The model architecture for this lightweight CNN is fairly straightforward to implement. The main challenge, however, lies in acquiring a robust dataset and ensuring proper formatting of the images. These steps are crucial for the performance of our CNN and will require some time to be effectively completed. This upgrade will be a substantial improvement to the lane line detection project, making it faster and more accurate.
