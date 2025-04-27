"""
decoder.py
This file contains functions for JPEG decompression:
- Inverse zig-zag scan
- Dequantization
- Inverse DCT
- Block merging
- Convert YCbCr back to RGB
"""

import numpy as np
from scipy.fftpack import dct, idct

# Standard JPEG Quantization Matrix for Luminance (Y)
QUANTIZATION_MATRIX_LUMA = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99]
])


def inverse_zigzag(arr, block_size=8):
    """
    Converts a 1D zig-zag scanned array back to a 2D 8x8 block.
    """
    zigzag_index = [
        (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
        (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
        (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
        (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
        (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
        (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
        (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
        (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)
    ]
    block = np.zeros((block_size, block_size))
    for idx, (i, j) in enumerate(zigzag_index):
        block[i, j] = arr[idx]
    return block


def dequantize(block, quant_matrix=QUANTIZATION_MATRIX_LUMA):
    """
    Dequantizes a block by multiplying with the quantization matrix.
    """
    return block * quant_matrix


def idct_2d(block):
    """
    Applies 2D Inverse DCT on an 8x8 block.
    """
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def block_merge(blocks, image_shape, block_size=8):
    """
    Merges 8x8 blocks back into a full image.
    """
    h, w = image_shape
    img = np.zeros((h, w))

    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            img[i:i+block_size, j:j+block_size] = blocks[idx]
            idx += 1
    return img


def ycbcr_to_rgb(img):
    """
    Converts a YCbCr image back to RGB.
    """
    xform = np.array([[1, 0, 1.402],
                      [1, -0.34414, -0.71414],
                      [1, 1.772, 0]])
    img = img.astype(np.float32)
    img[:, :, [1, 2]] -= 128
    rgb = img @ xform.T
    rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8)


def decode_image(encoded_channels):
    """
    Full decoding pipeline:
    - Inverse zig-zag
    - Dequantize
    - Inverse DCT
    - Merge blocks
    - Convert back to RGB if needed
    """
    reconstructed = []

    for channel_data in encoded_channels:
        h, w, encoded_blocks = channel_data
        blocks = []

        for block_zigzag in encoded_blocks:
            quant_block = inverse_zigzag(block_zigzag)
            dequant_block = dequantize(quant_block)
            block = idct_2d(dequant_block)
            block = block + 128  # Level shift back
            blocks.append(block)

        channel_img = block_merge(blocks, (h, w))
        reconstructed.append(channel_img)

    # Stack channels
    if len(reconstructed) == 1:
        final_img = np.clip(reconstructed[0], 0, 255).astype(np.uint8)
    else:
        img = np.stack(reconstructed, axis=-1)
        final_img = ycbcr_to_rgb(img)

    return final_img
