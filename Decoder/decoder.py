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


def inverse_zigzag(arr, block_size=8):
    # ... (this function is correct)
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
        # Ensure the index is within bounds of the input array
        if idx < len(arr):
            block[i, j] = arr[idx]
        # Note: If 'arr' is shorter than 64 (due to truncation of zeros),
        # the remaining elements in 'block' will correctly stay 0.
    return block


def dequantize(block, quant_matrix=QUANTIZATION_MATRIX_LUMA):
    # ... (this function is correct)
    return block * quant_matrix


def idct_2d(block):
    # ... (this function is correct)
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def block_merge(blocks, image_shape, block_size=8):
    # ... (this function is correct)
    h, w = image_shape
    # Calculate padded dimensions based on blocks
    num_blocks_w = (w + block_size - 1) // block_size
    num_blocks_h = (h + block_size - 1) // block_size
    
    # Create a larger array to accommodate potential padding from block splitting
    padded_h = num_blocks_h * block_size
    padded_w = num_blocks_w * block_size
    
    img_padded = np.zeros((padded_h, padded_w))

    idx = 0
    for i in range(0, padded_h, block_size):
        for j in range(0, padded_w, block_size):
            if idx < len(blocks): # Safety check, though block count should match
                 img_padded[i:i+block_size, j:j+block_size] = blocks[idx]
            idx += 1
            
    # Return only the original image area, discarding padding
    img = img_padded[:h, :w]
            
    return img


def ycbcr_to_rgb(img):
    # ... (this function is correct)
    xform = np.array([[1, 0, 1.402],
                      [1, -0.34414, -0.71414],
                      [1, 1.772, 0]])
    img = img.astype(np.float32)
    img[:, :, [1, 2]] -= 128
    rgb = img @ xform.T
    rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8)

def dc_decode_value(last_dc, dc_amplitude):
    """Decodes DC amplitude back to the DC value."""
    # In our simplified symbol format, the amplitude is the actual difference value.
    # We just add it to the previous DC value.
    return last_dc + dc_amplitude


def ac_decode_symbols(ac_symbols_and_amplitudes):
    """
    Decodes a list of AC (symbol, amplitude) tuples back into 63 AC coefficients.
    Handles RLE, EOB, and ZRL.
    """
    acs = np.zeros(63, dtype=np.int32) # Initialize the 63 AC coefficients to zeros
    idx = 0 # Current index in the acs array (0 to 62)

    for symbol, amplitude in ac_symbols_and_amplitudes:
        if symbol == 'EOB':
            # End of Block symbol encountered. The remaining coefficients
            # from the current index 'idx' onwards are all zero.
            # Since the 'acs' array was initialized to zeros, we just break.
            break # Stop processing symbols for this block's ACs

        # Handle ZRL (16 zeros) symbol: symbol is (15, 0)
        if symbol == (15, 0):
            run = 15 # ZRL represents 16 zeros run, but run in (run, size) means number of zeros *before* the non-zero value. ZRL means 16 zeros *then* another symbol.
            # So if we see (15,0), it means skip 16 positions.
            idx += 16 # Move index past 16 zeros
            # There is no amplitude associated with ZRL

        else:
            # Regular AC symbol (Run, Size)
            run, size = symbol
            idx += run # Skip the number of zeros indicated by 'run'

            # Check for index overflow before placing the non-zero coefficient
            if idx >= 63:
                 # This should not happen with correctly encoded data, but handle defensively
                 print(f"Warning: AC decoding index overflow at index {idx}. Data might be corrupted.")
                 break # Stop processing if we go past the 63 AC positions

            # Place the decoded non-zero amplitude value at the current index
            acs[idx] = amplitude # In our simplified format, amplitude is the actual value

            idx += 1 # Move to the next position for the next symbol/amplitude


    return acs # Return the reconstructed 63 AC coefficients


def decode_image(encoded_channels):
    """
    Full decoding pipeline from symbol/amplitude lists:
    - Decode DC/AC symbols/amplitudes back to coefficients
    - Inverse zig-zag
    - Dequantization
    - Inverse DCT
    - Block merging
    - Convert back to RGB if needed
    """
    reconstructed = []

    # Store last_dc value for differential decoding, initialized to 0 for each channel
    # The size of last_dc list depends on the number of channels in the encoded data
    last_dc = [0] * len(encoded_channels)


    # Use enumerate to get both the index (ch_index) and the data (channel_data)
    # ch_index will be 0 for the first channel, 1 for the second, etc.
    for ch_index, channel_data in enumerate(encoded_channels):
        # Unpack the new structure: (original_height, original_width, channel_type, list_of_block_symbols_lists)
        h, w, channel_type, encoded_blocks_symbols_list = channel_data
        blocks = [] # List to hold reconstructed 8x8 coefficient blocks (before dequant)

        # Select dequantization matrix based on channel type
        if channel_type in ('Cb', 'Cr'): # Cb or Cr (they share the same matrix)
            quant_matrix = QUANTIZATION_MATRIX_CHROMA
        else: # 'Y' (Luminance)
            quant_matrix = QUANTIZATION_MATRIX_LUMA


        # Iterate through the list of symbols/amplitudes for each block
        for block_symbols in encoded_blocks_symbols_list:

            # --- Decode DC ---
            # The first item in the block_symbols list is the DC (symbol, amplitude)
            dc_symbol, dc_amplitude = block_symbols[0]
            # Use the dc_amplitude (the difference value) to get the current DC value
            current_dc = dc_decode_value(last_dc[ch_index], dc_amplitude)
            last_dc[ch_index] = current_dc # Update last_dc for this channel


            # --- Decode ACs ---
            # The remaining items in the block_symbols list are the AC symbols and amplitudes
            ac_symbols_and_amplitudes = block_symbols[1:]
            # Decode the list of AC symbols/amplitudes into a flat array of 63 AC coefficients
            ac_coeffs = ac_decode_symbols(ac_symbols_and_amplitudes)


            # --- Reconstruct the 64 coefficients in zig-zag order ---
            # Create a zero-filled 64-element array
            zigzag_coeffs = np.zeros(64, dtype=np.int32)
            # Place the decoded DC value at the first position
            zigzag_coeffs[0] = current_dc
            # Place the decoded AC coefficients at the remaining 63 positions
            zigzag_coeffs[1:] = ac_coeffs


            # --- Inverse Zig-zag ---
            # Convert the 1D array back to an 8x8 block
            quant_block = inverse_zigzag(zigzag_coeffs)


            # --- Dequantization ---
            # Multiply by the quantization matrix
            dequant_block = dequantize(quant_block, quant_matrix=quant_matrix)


            # --- Inverse DCT ---
            block = idct_2d(dequant_block)

            # Level shift back (+128)
            block = block + 128 # Add back the 128 we subtracted during encoding

            blocks.append(block)

        # Pass original h, w to block_merge so it can crop back correctly
        # Use the original height and width stored in the encoded data tuple
        channel_img = block_merge(blocks, (h, w))

        # Add the reconstructed channel image (still in YCbCr range if color) to the list
        reconstructed.append(channel_img)


    # Stack channels back together
    if len(reconstructed) == 1:
        # Grayscale image: only one channel (Y)
        # Ensure the final image is clipped to 0-255 and converted to uint8
        final_img = np.clip(reconstructed[0], 0, 255).astype(np.uint8)
    else:
        # Color image: stack Y, Cb, Cr channels (assuming the order Y, Cb, Cr from encoding)
        img = np.stack(reconstructed, axis=-1)
        # Convert from YCbCr to RGB. ycbcr_to_rgb already handles clipping and uint8 conversion.
        final_img = ycbcr_to_rgb(img)

    return final_img