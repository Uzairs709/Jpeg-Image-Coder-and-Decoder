"""
encoder.py
This file contains functions for JPEG compression:
- Convert image to YCbCr (for color)
- Block splitting into 8x8
- Apply DCT
- Quantize DCT coefficients
- Zig-zag scanning
"""

import numpy as np
from scipy.fftpack import dct, idct

# Standard JPEG Quantization Matrix for Luminance (Y)
QUANTIZATION_MATRIX_LUMA = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


def rgb_to_ycbcr(img):
    """
    Converts an RGB image to YCbCr color space.
    """
    xform = np.array([[0.299, 0.587, 0.114],
                      [-0.1687, -0.3313, 0.5],
                      [0.5, -0.4187, -0.0813]])
    ycbcr = img @ xform.T
    ycbcr[:, :, [1, 2]] += 128
    return ycbcr


def block_split(img, block_size=8):
    """
    Splits image into non-overlapping blocks of block_size x block_size.
    """
    h, w = img.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i + block_size, j:j + block_size]
            # Handle padding if needed
            if block.shape[0] != block_size or block.shape[1] != block_size:
                block = np.pad(block, ((0, block_size - block.shape[0]),
                                       (0, block_size - block.shape[1])), mode='constant')
            blocks.append(block)
    return blocks


def dct_2d(block):
    """
    Applies 2D DCT on an 8x8 block.
    """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def quantize(block, quant_matrix=QUANTIZATION_MATRIX_LUMA):
    """
    Quantizes DCT coefficients by dividing by the quantization matrix and rounding.
    """
    return np.round(block / quant_matrix).astype(np.int32)


def zigzag_scan(block):
    """
    Converts an 8x8 block to a 1D array using zig-zag order.
    """
    zigzag_index = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
        (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
        (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
        (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]
    return np.array([block[i, j] for i, j in zigzag_index])


def encode_image(img):
    """
    Full encoding pipeline:
    - Convert to YCbCr
    - Split into blocks
    - Apply DCT
    - Quantize
    - Zig-zag scan
    - Return encoded list
    """
    # If color image, convert
    if len(img.shape) == 3:
        img = rgb_to_ycbcr(img)

    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1

    encoded_channels = []

    for ch in range(channels):
        channel = img[:, :, ch] if channels > 1 else img

        # Level shift by -128
        channel = channel - 128

        blocks = block_split(channel)
        encoded_blocks = []

        for block in blocks:
            dct_block = dct_2d(block)
            quant_block = quantize(dct_block)
            zigzag_block = zigzag_scan(quant_block)
            encoded_blocks.append(zigzag_block)

        encoded_channels.append((h, w, encoded_blocks))

    return encoded_channels
