"""
decoder.py
This file contains functions for JPEG decompression, including Huffman decoding:
- Reads Huffman-encoded bitstream
- Decodes Huffman codes and amplitude bits to get symbols and coefficients
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


# --- Helper functions for Size Category and Amplitude encoding/decoding ---

def get_size_category(value):
    """
    Returns the bit size category for a value according to JPEG standard (Table K.3).
    Size 0 for value 0. Size 1 for -1, 1, etc.
    """
    if value == 0:
        return 0
    size = 0
    abs_value = abs(value)
    while abs_value > (1 << size) -1:
        size += 1
        if size > 11: # Max DC size is 11, max AC size is 10. Should not exceed this.
             print(f"Warning: Calculated size category {size} for value {value}. Capping at 11.")
             size = 11
             break
    return size


def decode_amplitude(size, amplitude_bits):
    """
    Decodes the amplitude bits back to the signed integer value
    based on its size category (Ref. JPEG standard, Table K.3, K.5).
    """
    if size == 0:
        return 0 # Should not happen for non-zero size
    # If the most significant bit of the amplitude_bits is 1, it's a negative number
    # The check is (amplitude_bits >> (size - 1)) & 1
    if (amplitude_bits >> (size - 1)) & 1:
        # Convert from amplitude bits representation to signed value
        return amplitude_bits - (1 << size)
    else:
        # Positive number: amplitude bits are the direct value
        return amplitude_bits


# --- Function to build Huffman tables from standard lengths and values ---

def build_huffman_table(bits, values, table_type):
    """
    Builds a Huffman code table mapping symbol -> (code_length, code_value).
    :param bits: List with counts of codes of length 1 to 16 (index 1-16).
    :param values: List of symbols' byte representations (AC) or integer size categories (DC), ordered by code length.
    :param table_type: String, either 'dc' or 'ac', to correctly interpret values.
    :return: Dictionary mapping symbol (integer for DC, tuple (Run, Size) for AC) -> (code_length, code_value).
    """
    huffman_table = {}
    code = 0
    value_pos = 0

    for bit_length in range(1, 17):
        if bit_length < len(bits):
             num_codes_of_length = bits[bit_length]
        else:
             num_codes_of_length = 0

        for i in range(num_codes_of_length):
            if value_pos >= len(values):
                 print(f"DEBUG: Error condition met in build_huffman_table.")
                 print(f"DEBUG: Current bit_length: {bit_length}, code_index: {i}")
                 print(f"DEBUG: Current value_pos: {value_pos}")
                 print(f"DEBUG: Length of values list: {len(values)}")
                 sum_nrcodes_so_far = sum(bits[l] for l in range(1, bit_length)) + i
                 print(f"DEBUG: Sum of nrcodes processed up to this point: {sum_nrcodes_so_far}")
                 raise IndexError(f"list index out of range in build_huffman_table: value_pos {value_pos} >= len(values) {len(values)} at bit_length {bit_length}, code_index {i}")


            raw_symbol_value = values[value_pos]

            if table_type == 'dc':
                symbol = raw_symbol_value
                if not (0 <= symbol <= 11):
                     print(f"Warning: Unexpected raw DC symbol value {raw_symbol_value} encountered.")

            elif table_type == 'ac':
                 raw_val_int = int(raw_symbol_value)
                 if raw_val_int == 0x00:
                     symbol = (0, 0) # EOB symbol
                 elif raw_val_int == 0xF0:
                      symbol = (15, 0) # ZRL symbol
                 elif 0x01 <= raw_val_int <= 0xEF:
                      run = (raw_val_int >> 4) & 0x0F
                      size = raw_val_int & 0x0F
                      symbol = (run, size)
                 else:
                      print(f"Warning: Unexpected raw AC symbol value {raw_symbol_value} ({raw_val_int}) encountered.")
                      value_pos += 1 # Increment value_pos even if skipping
                      continue # Skip this value

            else:
                 raise ValueError(f"Unknown table_type: {table_type}")

            if isinstance(symbol, (int, tuple)):
                 huffman_table[symbol] = (bit_length, code)
            else:
                 print(f"Warning: Skipping unhashable symbol type {type(symbol)}: {symbol}")

            code += 1
            value_pos += 1

        code <<= 1

    if table_type == 'ac':
        print(f"DEBUG: Finished building {table_type} table. Total symbols added: {len(huffman_table)}")
        if (15, 1) in huffman_table:
             print(f"DEBUG: Symbol (15, 1) FOUND in {table_type} table. Code: {huffman_table[(15, 1)]}")
        else:
             print(f"DEBUG: Symbol (15, 1) NOT FOUND in {table_type} table.")
             # print(f"DEBUG: All keys in {table_type} table: {list(huffman_table.keys())}")


    return huffman_table

# --- Standard JPEG Huffman Table Data (as defined in the spec) ---

# DC Luminance: counts of codes per length, followed by symbols (0-11)
std_dc_luminance_nrcodes = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0] # Index 0 unused, lengths 1-16
std_dc_luminance_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # Total 12

# AC Luminance: counts of codes per length, followed by symbols (byte representation)
std_ac_luminance_nrcodes = [0,0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,0x7d] # Last element 0x7d (125) is count for length 16
std_ac_luminance_values = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, # 0xF0 is ZRL
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
    0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
    0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
    0x10, 0x20 # Total 162 elements
]

# DC Chrominance: counts of codes per length, followed by symbols (0-11)
std_dc_chrominance_nrcodes = [0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0] # Index 0 unused, lengths 1-16
std_dc_chrominance_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # Total 12

# AC Chrominance: counts of codes per length, followed by symbols (byte representation)
std_ac_chrominance_nrcodes = [0,0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,0x77] # Last element 0x77 (119) is count for length 16
std_ac_chrominance_values = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, # ZRL
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
    0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
    0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
    0x10, 0x20
] # Total 162 elements


# Add temporary print statements to verify list lengths *after* definition
print(f"Length of std_dc_luminance_values: {len(std_dc_luminance_values)}")
print(f"Length of std_ac_luminance_values: {len(std_ac_luminance_values)}")
print(f"Length of std_dc_chrominance_values: {len(std_dc_chrominance_values)}")
print(f"Length of std_ac_chrominance_values: {len(std_ac_chrominance_values)}")


# === BUILD THE FINAL HUFFMAN TABLES (map symbol -> (length, code)) ===

# Pass the table type ('dc' or 'ac') to the build function
HUFF_TABLE_LUMA_DC = build_huffman_table(std_dc_luminance_nrcodes, std_dc_luminance_values, 'dc')
HUFF_TABLE_LUMA_AC = build_huffman_table(std_ac_luminance_nrcodes, std_ac_luminance_values, 'ac')
HUFF_TABLE_CHROMA_DC = build_huffman_table(std_dc_chrominance_nrcodes, std_dc_chrominance_values, 'dc')
HUFF_TABLE_CHROMA_AC = build_huffman_table(std_ac_chrominance_nrcodes, std_ac_chrominance_values, 'ac')


# Helper to get the correct Huffman tables based on channel type
def get_huffman_tables(channel_type):
    if channel_type == 'Y':
        return HUFF_TABLE_LUMA_DC, HUFF_TABLE_LUMA_AC
    elif channel_type in ('Cb', 'Cr'): # Cb and Cr use Chroma tables
        return HUFF_TABLE_CHROMA_DC, HUFF_TABLE_CHROMA_AC
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")


# --- Bitstream Handling Classes ---

# Note: BitStreamWriter is only needed in encoder.py, but BitStreamReader is needed here.
class BitStreamReader:
    """Reads bits from a byte stream, handling byte unstuffing (0xFF -> 0xFF 0x00)."""
    def __init__(self, byte_data):
        self._byte_data = byte_data # The input byte stream (bytes or bytearray)
        self._byte_index = 0      # Current index in the byte stream
        self._bits_in_byte = 0    # Number of bits already read from the current byte (0-7)
        self._current_byte = 0    # The byte currently being read from (integer 0-255)

        # Load the first byte to start if data exists
        if len(self._byte_data) > 0:
            self._current_byte = self._byte_data[self._byte_index]

    def read_bits(self, num_bits):
        """Reads the specified number of bits from the stream."""
        if num_bits <= 0:
            return 0

        result = 0
        for _ in range(num_bits):
            # If all bits in the current byte are read, load the next byte
            if self._bits_in_byte == 8:
                self._byte_index += 1
                # Handle JPEG Byte Unstuffing: If the previous byte was 0xFF,
                # and the current byte is 0x00, skip the 0x00.
                # Check against length - 1 to avoid index error on the last byte
                if self._byte_index < len(self._byte_data) and self._byte_data[self._byte_index - 1] == 0xFF and self._byte_data[self._byte_index] == 0x00:
                     self._byte_index += 1

                # Check if we ran out of data
                if self._byte_index >= len(self._byte_data):
                    # print(f"Warning: Tried to read {num_bits} bits past end of bitstream. Returning -1.")
                    return -1 # Indicate end of stream/error

                # Load the next byte
                if self._byte_index < len(self._byte_data): # Added safety check
                    self._current_byte = self._byte_data[self._byte_index]
                    self._bits_in_byte = 0 # Reset bit counter for the new byte
                else:
                     # Really out of data
                     # print("Warning: Ran out of bytes in BitStreamReader")
                     return -1


            # Get the next bit (from MSB to LSB of the current byte)
            bit = (self._current_byte >> (7 - self._bits_in_byte)) & 1
            result = (result << 1) | bit # Add the bit to the result
            self._bits_in_byte += 1 # Increment bit counter within the byte

        return result # Return the integer value of the bits read

    def has_more_bits(self):
        """Checks if there are more bits available in the stream."""
        # More bits available if there are more bytes to read OR if the current byte
        # is not fully read.
        return self._byte_index < len(self._byte_data) or (self._byte_index == len(self._byte_data) - 1 and self._bits_in_byte < 8)


    # Helper for Huffman decoding: read bits and check against codes
    # A simple way is to read bit by bit and check against all codes of increasing length
    def decode_symbol(self, huff_table):
        """
        Reads bits from the stream and decodes the next Huffman symbol.
        Returns the decoded symbol (integer or tuple) or None if decoding fails.
        """
        current_value = 0
        current_length = 0

        # Create a reverse lookup: (code_value, code_length) -> symbol
        # This is inefficient if done every time. Better to build it once per channel.
        # For now, build it here for clarity.
        reverse_lookup = {}
        for symbol, (length, code) in huff_table.items():
             # Ensure uniqueness - canonical codes should be unique
             if (code, length) in reverse_lookup:
                  # This indicates an issue with the Huffman table itself
                  print(f"Error: Duplicate Huffman code entry for {(code, length)}")
                  return None # Indicate failure
             reverse_lookup[(code, length)] = symbol


        # Read up to the maximum possible Huffman code length (16 for AC, 9 for DC symbols 0-11)
        # Determine max length dynamically from the table passed in
        max_code_length = 0
        if huff_table:
            # Max length for DC symbols 0-11 is 9. Max length for AC symbols is 16.
            # We need to know if this is a DC or AC table to set the max length correctly
            # A simple check: if any symbol is a tuple, it's an AC table.
            is_ac_table = any(isinstance(s, tuple) for s in huff_table.keys())
            if is_ac_table:
                 max_code_length = max(length for symbol, (length, code) in huff_table.items()) # Max AC code length is 16
            else:
                 # For DC tables, the max symbol is 11, max code length is 9
                 max_code_length = max(length for symbol, (length, code) in huff_table.items()) # Max DC code length is 9 (for symbol 11)
                 # Add a safety cap, standard max is 16
                 if max_code_length > 16: max_code_length = 16 # Should not happen with standard tables


        else:
             # Should not happen if tables are built correctly
             print("Error: decode_symbol called with empty Huffman table.")
             return None


        for _ in range(max_code_length): # Read up to max_code_length bits
            bit = self.read_bits(1)
            if bit == -1: # Ran out of data
                 # print("Error: Ran out of bits while decoding Huffman symbol")
                 return None # Indicate failure

            current_value = (current_value << 1) | bit
            current_length += 1

            # Check if the current bit sequence matches any code in the reverse lookup
            if (current_value, current_length) in reverse_lookup:
                symbol = reverse_lookup[(current_value, current_length)]
                return symbol # Return the decoded symbol

        # If we read max_code_length bits and didn't find a symbol, something is wrong
        print(f"Error: Could not decode Huffman symbol after {max_code_length} bits. Current value: {current_value}, length: {current_length}")
        return None # Indicate failure


# --- Inverse Transformation and Block Merging (from previous code) ---

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
    block = np.zeros((block_size, block_size), dtype=np.int32) # Ensure integer type
    for idx, (i, j) in enumerate(zigzag_index):
        if idx < len(arr):
            block[i, j] = arr[idx]
    return block


def dequantize(block, quant_matrix=QUANTIZATION_MATRIX_LUMA):
    # ... (this function is correct)
    return block * quant_matrix


def idct_2d(block):
    """
    Applies 2D Inverse DCT on an 8x8 block (expects float input).
    """
    # IDCT output should be float before level shift
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def block_merge(blocks, image_shape, block_size=8):
    # ... (this function is correct)
    h, w = image_shape
    num_blocks_w = (w + block_size - 1) // block_size
    num_blocks_h = (h + block_size - 1) // block_size

    padded_h = num_blocks_h * block_size
    padded_w = num_blocks_w * block_size

    # Use float32 type for image reconstruction before final clipping/conversion
    img_padded = np.zeros((padded_h, padded_w), dtype=np.float32)

    idx = 0
    for i in range(0, padded_h, block_size):
        for j in range(0, padded_w, block_size):
            if idx < len(blocks):
                 # Ensure block is float32 before assignment if not already
                 img_padded[i:i+block_size, j:j+block_size] = blocks[idx].astype(np.float32)
            idx += 1

    img = img_padded[:h, :w]

    return img


def ycbcr_to_rgb(img):
    """
    Converts a YCbCr image back to RGB (Assumes float32 input).
    Handles clipping and returns uint8.
    """
    # Ensure input is float32 as expected
    img = img.astype(np.float32)

    xform = np.array([[1, 0, 1.402],
                      [1, -0.34414, -0.71414],
                      [1, 1.772, 0]])
    img[:, :, [1, 2]] -= 128 # Subtract 128 from Cb and Cr
    # Matrix multiplication
    rgb = img @ xform.T
    # Clip to 0-255 range and convert to uint8
    rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8)


def dc_decode_value(last_dc, dc_amplitude):
    """Decodes DC amplitude back to the DC value (last_dc + amplitude)."""
    return last_dc + dc_amplitude


# Remove the old ac_decode_symbols function definition here.
# The logic to decode AC symbols will be integrated into the decode_image loop.


def decode_image(encoded_channels):
    """
    Full decoding pipeline from Huffman-encoded bitstream bytes per channel.
    Reads bitstream, decodes symbols/amplitudes, reconstructs coefficients,
    then performs inverse zig-zag, dequantization, IDCT, merge, and color conversion.
    """
    reconstructed = []

    # Store last_dc value for differential decoding, initialized to 0 for each channel
    # The size of last_dc list depends on the number of channels in the encoded data
    last_dc = [0] * len(encoded_channels)


    # Use enumerate to get both the index (ch_index) and the data (channel_data)
    for ch_index, channel_data in enumerate(encoded_channels):
        # Unpack the new structure: (original_height, original_width, channel_type, bitstream_bytes)
        h, w, channel_type, bitstream_bytes = channel_data

        # Get the correct Huffman tables for this channel
        huff_dc_table, huff_ac_table = get_huffman_tables(channel_type)

        # Initialize a bitstream reader for this channel's byte data
        bitstream_reader = BitStreamReader(bitstream_bytes)

        blocks = [] # List to hold reconstructed 8x8 blocks (after IDCT, before merge)

        # Select dequantization matrix based on channel type
        if channel_type in ('Cb', 'Cr'):
            quant_matrix = QUANTIZATION_MATRIX_CHROMA
        else: # 'Y' or grayscale
            quant_matrix = QUANTIZATION_MATRIX_LUMA


        # --- Decode Blocks from Bitstream ---
        # Loop and decode blocks as long as there's data in the bitstream.
        # A real decoder checks block count or uses restart markers.
        # Here, we just read until EOB doesn't leave enough bits for the next DC or we finish all blocks.
        block_size = 8
        num_blocks_h = (h + block_size - 1) // block_size
        num_blocks_w = (w + block_size - 1) // block_size
        expected_num_blocks = num_blocks_h * num_blocks_w
        blocks_decoded_count = 0


        while blocks_decoded_count < expected_num_blocks and bitstream_reader.has_more_bits():
            # Start of a new block
            # print(f"DEBUG: Decoding block {blocks_decoded_count} for {channel_type} channel.")
            # print(f"DEBUG: Bitstream position before DC: Byte {bitstream_reader._byte_index}, Bit {bitstream_reader._bits_in_byte}")


            # --- Decode DC ---
            # Read bits and decode the DC symbol (size category)
            dc_symbol = bitstream_reader.decode_symbol(huff_dc_table) # This table now has integer keys
            if dc_symbol is None: # Error decoding symbol or ran out of bits unexpectedly
                 print(f"Error decoding DC symbol for block {blocks_decoded_count} in {channel_type} channel. Likely corrupted data or unexpected end of stream.")
                 break # Stop decoding this channel

            # print(f"DEBUG: Decoded DC symbol: {dc_symbol}")

            # If size > 0, read the DC amplitude bits
            dc_amplitude_bits = 0
            if dc_symbol > 0:
                dc_amplitude_bits = bitstream_reader.read_bits(dc_symbol) # Size is the number of bits for amplitude
                if dc_amplitude_bits == -1: # Error reading bits or ran out
                     print(f"Error reading DC amplitude bits (size {dc_symbol}) for block {blocks_decoded_count} in {channel_type} channel.")
                     break # Stop decoding this channel
                # print(f"DEBUG: Decoded DC amplitude bits: {dc_amplitude_bits} (size {dc_symbol})")


            # Decode the amplitude bits back to the signed difference value
            dc_amplitude = decode_amplitude(dc_symbol, dc_amplitude_bits)

            # Calculate the current DC value using the differential value
            current_dc = last_dc[ch_index] + dc_amplitude
            last_dc[ch_index] = current_dc # Update last_dc for this channel

            # print(f"DEBUG: Decoded DC value: {current_dc}")


            # --- Decode ACs ---
            # Initialize a 64-element array for zig-zag coefficients for this block
            # DC is at index 0, ACs are at indices 1-63
            zigzag_coeffs = np.zeros(64, dtype=np.int32)
            zigzag_coeffs[0] = current_dc # Place the decoded DC value

            ac_idx_in_block = 1 # Start filling AC coefficients from index 1 in zig-zag order (after DC)

            # Loop to decode AC symbols until EOB is decoded or all 63 AC positions are filled
            while ac_idx_in_block < 64: # Loop until the 64th position is potentially filled
                # Decode the next AC symbol (Run, Size), EOB, or ZRL
                ac_symbol = bitstream_reader.decode_symbol(huff_ac_table) # This table has tuple keys
                if ac_symbol is None: # Error decoding symbol or ran out unexpectedly
                     print(f"Error decoding AC symbol for block {blocks_decoded_count}, AC index {ac_idx_in_block -1} in {channel_type} channel. Likely corrupted data or unexpected end of stream.")
                     ac_idx_in_block = 64 # Exit inner loop gracefully
                     break # Exit outer block decoding loop

                # print(f"DEBUG: Decoded AC symbol: {ac_symbol}")

                # Handle EOB symbol (0, 0)
                if ac_symbol == (0, 0):
                    # Remaining coefficients from ac_idx_in_block onwards are zeros
                    # (our zigzag_coeffs array is initialized to zeros).
                    # We just need to advance ac_idx_in_block to 64 to exit the loop.
                    ac_idx_in_block = 64 # Exit AC decoding loop for this block
                    break # Exit inner AC decoding loop for this block

                # Handle ZRL (16 zeros) symbol (15, 0)
                if ac_symbol == (15, 0):
                    run = 16 # ZRL means 16 zeros
                    ac_idx_in_block += run # Skip 16 positions
                    # Check for overflow (should not go past 64, including DC)
                    if ac_idx_in_block > 64:
                         print(f"Warning: ZRL decoding overflow at block {blocks_decoded_count}, AC index {ac_idx_in_block - run} in {channel_type} channel. Index reached {ac_idx_in_block}.")
                         ac_idx_in_block = 64 # Cap index
                         break # Exit inner AC decoding loop for this block
                    continue # Go to decode the next symbol (which comes after the 16 zeros)

                # Handle regular AC symbol (Run, Size)
                run, size = ac_symbol
                ac_idx_in_block += run # Skip 'run' number of zeros

                # Check for index overflow before placing the non-zero coefficient
                if ac_idx_in_block >= 64:
                     print(f"Warning: AC run overflow at block {blocks_decoded_count}, AC index {ac_idx_in_block - run} in {channel_type} channel. Index reached {ac_idx_in_block}.")
                     ac_idx_in_block = 64 # Cap index
                     break # Exit inner AC decoding loop for this block


                # Read the AC amplitude bits (only if size > 0)
                ac_amplitude_bits = 0
                if size > 0:
                     ac_amplitude_bits = bitstream_reader.read_bits(size)
                     if ac_amplitude_bits == -1: # Error reading bits or ran out
                          print(f"Error reading AC amplitude bits (size {size}) for block {blocks_decoded_count}, AC index {ac_idx_in_block-1} in {channel_type} channel.")
                          ac_idx_in_block = 64 # Exit inner loop
                          break # Exit outer block decoding loop
                     # print(f"DEBUG: Decoded AC amplitude bits: {ac_amplitude_bits} (size {size})")


                # Decode the amplitude bits back to the signed value
                ac_amplitude = decode_amplitude(size, ac_amplitude_bits)

                # Place the decoded non-zero amplitude value at the current index
                zigzag_coeffs[ac_idx_in_block] = ac_amplitude

                # print(f"DEBUG: Decoded AC value at index {ac_idx_in_block}: {ac_amplitude}")

                ac_idx_in_block += 1 # Move to the next position for the next symbol/amplitude


            # After decoding all symbols for the block (either hit EOB or filled 63 ACs),
            # proceed with inverse transformations for this block.

            # --- Inverse Zig-zag ---
            # Convert the 1D zig-zag array back to an 8x8 block
            quant_block = inverse_zigzag(zigzag_coeffs)

            # --- Dequantization ---
            dequant_block = dequantize(quant_block, quant_matrix=quant_matrix)

            # --- Inverse DCT ---
            block = idct_2d(dequant_block)

            # Level shift back (+128)
            block = block + 128

            blocks.append(block)
            blocks_decoded_count += 1 # Increment block count


        # --- Merge Blocks back into Channel Image ---
        # Pass original h, w to block_merge so it can crop back correctly
        channel_img = block_merge(blocks, (h, w))

        # Add the reconstructed channel image (still in YCbCr range if color) to the list
        reconstructed.append(channel_img)

        # After processing a channel, if not all expected blocks were decoded, issue warning
        if blocks_decoded_count < expected_num_blocks:
             print(f"Warning: Decoded only {blocks_decoded_count}/{expected_num_blocks} blocks for {channel_type} channel. Bitstream might be incomplete or corrupted.")


    # --- Stack Channels back together ---
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
