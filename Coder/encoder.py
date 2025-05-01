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

# Standard JPEG Quantization Matrix for Chrominance (Cb, Cr)
QUANTIZATION_MATRIX_CHROMA = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

# Add these helper functions to both files

def get_size_category(value):
    """
    Returns the bit size category for a value according to JPEG standard (Table K.3).
    Size 0 for value 0.
    Size 1 for -1, 1.
    Size 2 for -3 to -2, 2 to 3.
    Size 3 for -7 to -4, 4 to 7.
    ...
    Size 10 for -1023 to -512, 512 to 1023.
    """
    if value == 0:
        return 0
    # Calculate size based on the number of bits needed for the absolute value
    size = 0
    abs_value = abs(value)
    # (1 << size) - 1 is equivalent to 2^size - 1
    # The size category is the smallest size such that abs_value fits in 'size' bits.
    while abs_value > (1 << size) -1:
        size += 1
    return size


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

def dc_encode_symbol(dc_diff):
    """Encodes DC difference into (size, amplitude)."""
    # Amplitude is the actual difference value
    # Size is the bit category of the difference value
    return (get_size_category(dc_diff), dc_diff)


def ac_encode_symbols(acs):
    """
    Encodes AC coefficients using RLE and returns a list of (symbol, amplitude) tuples.
    Symbols are (Run, Size), special cases EOB ('EOB', None), ZRL ((15, 0), None).
    """
    symbols = []
    zero_run = 0
    # Iterate through the 63 AC coefficients (indices 1 to 63 after zig-zag)
    for i in range(63):
        coeff = acs[i] # Get the current AC coefficient

        if coeff == 0:
            zero_run += 1
        else:
            # Handle runs of zeros longer than 15 by inserting ZRLs
            while zero_run > 15:
                symbols.append(((15, 0), None)) # ZRL symbol for 16 zeros
                zero_run -= 16

            # Encode the non-zero AC coefficient
            size = get_size_category(coeff)
            # The symbol is a tuple (Run of Zeros before this coeff, Size Category of coeff)
            symbol = (zero_run, size)
            amplitude = coeff # The amplitude is the actual value
            symbols.append((symbol, amplitude))

            zero_run = 0 # Reset zero run after a non-zero coefficient

    # After iterating through all 63 ACs, if there are trailing zeros, add EOB
    # EOB indicates the rest of the block (from the current position) are zeros.
    # We add EOB if the last AC wasn't processed (i.e., trailing zeros existed)
    # or if the block was entirely zeros (zero_run == 63).
    # A simple check is if the total number of encoded symbols + zero_run + 1 (for DC)
    # doesn't cover all 64 positions. A simpler approach: if zero_run > 0 after loop, add EOB.
    # Also if the list of symbols is empty (meaning all 63 ACs were zero), we need EOB.
    if zero_run > 0 or len(symbols) == 0:
         symbols.append(('EOB', None))

    return symbols


# Update the encode_image function to use these helpers
def encode_image(img):
    """
    Full encoding pipeline producing symbol/amplitude lists:
    - Convert to YCbCr
    - Split into blocks
    - Apply DCT
    - Quantize
    - Zig-zag scan
    - Perform DC differential and AC RLE encoding
    - Return list of channel data with symbol/amplitude lists
    """
    # If color image, convert
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = rgb_to_ycbcr(img)
        channels = 3
    elif len(img.shape) == 2: # Grayscale
        channels = 1
    else:
         raise ValueError("Input image must be grayscale or 3-channel color")


    h, w = img.shape[:2]


    encoded_channels = []

    # Store last_dc value for differential encoding, initialized to 0 for each channel
    last_dc = [0] * channels


    for ch_index in range(channels):
        # Extract channel data - ensure it's 2D for block_split
        if channels > 1:
            channel_data_2d = img[:, :, ch_index]
            # Determine channel type for decoder
            if ch_index == 0:
                channel_type = 'Y'
            elif ch_index == 1:
                channel_type = 'Cb'
            else: # ch_index == 2
                channel_type = 'Cr'
        else: # Grayscale
            channel_data_2d = img[:, :]
            channel_type = 'Y' # Grayscale channel is treated as Luminance

        # Level shift by -128
        channel_shifted = channel_data_2d.astype(np.float32) - 128 # Use float for intermediate steps

        blocks = block_split(channel_shifted)
        encoded_blocks_symbols_list = [] # List to hold symbol/amplitude lists for each block


        # Select quantization matrix based on channel type
        # In JPEG, Cb and Cr use the same chrominance matrix
        if channel_type in ('Cb', 'Cr'):
            quant_matrix = QUANTIZATION_MATRIX_CHROMA
        else: # 'Y' (Luminance)
            quant_matrix = QUANTIZATION_MATRIX_LUMA


        for block in blocks:
            dct_block = dct_2d(block)
            quant_block = quantize(dct_block, quant_matrix=quant_matrix)

            # Zig-zag scan to get the 1D array of 64 coefficients
            zigzag_block = zigzag_scan(quant_block)

            # --- Perform DC differential encoding ---
            current_dc = zigzag_block[0]
            dc_diff = current_dc - last_dc[ch_index]
            dc_symbol, dc_amplitude = dc_encode_symbol(dc_diff)
            last_dc[ch_index] = current_dc # Update last_dc for this channel

            # --- Perform AC RLE encoding ---
            # Get the 63 AC coefficients (elements 1 to 63 from the zig-zag scan)
            ac_coeffs = zigzag_block[1:]
            ac_symbols_and_amplitudes = ac_encode_symbols(ac_coeffs)

            # Combine DC and AC symbols/amplitudes for this block
            # The block symbols list starts with DC, followed by ACs including EOB/ZRL
            block_symbols = [(dc_symbol, dc_amplitude)] + ac_symbols_and_amplitudes

            encoded_blocks_symbols_list.append(block_symbols)


        # The encoded data for a channel is now a tuple:
        # (original_height, original_width, channel_type, list_of_block_symbols_lists)
        # We need original height and width for merging blocks correctly later
        encoded_channels.append((h, w, channel_type, encoded_blocks_symbols_list))

    # The return value is a list containing data for each channel
    # Example: [(h_y, w_y, 'Y', [block_symbols_y1, ...]), (h_cb, w_cb, 'Cb', [block_symbols_cb1, ...]), ...]
    return encoded_channels