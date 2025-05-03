import numpy as np
from scipy.fftpack import dct
import logging

# Standard JPEG Quantization Matrix for Luminance (Y)
QUANTIZATION_MATRIX_LUMA = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 78, 103, 121, 120, 101],  # corrected
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

logging.basicConfig(level=logging.WARNING)

# Helper functions for Size Category and Amplitude encoding/decoding

def get_size_category(value):
    if value == 0:
        return 0
    size = 0
    abs_value = abs(value)
    while abs_value > (1 << size) - 1:
        size += 1
        if size > 11:
            logging.warning(f"Calculated unusually large size category {size} for value {value}, capping at 11.")
            return 11
    return size


def get_amplitude_bits(value, size):
    if size == 0:
        return 0
    if value > 0:
        return value
    else:
        return (1 << size) - 1 + value

# Build Huffman table

def build_huffman_table(bits, values, table_type):
    # early check: sum of code counts must equal number of values
    assert len(values) == sum(bits[1:]), \
        f"{table_type} values length {len(values)} != sum(nrcodes) {sum(bits[1:])}"

    table = {}
    code = 0
    pos = 0
    for length in range(1, 17):
        count = bits[length] if length < len(bits) else 0
        for _ in range(count):
            raw = values[pos]
            if table_type == 'dc':
                symbol = int(raw)
            else:
                raw_int = int(raw)
                if raw_int == 0x00:
                    symbol = (0, 0)
                elif raw_int == 0xF0:
                    symbol = (15, 0)
                else:
                    r = (raw_int >> 4) & 0x0F
                    s = raw_int & 0x0F
                    symbol = (r, s)
            table[symbol] = (length, code)
            code += 1
            pos += 1
        code <<= 1
    logging.debug(f"Built {table_type} table with {len(table)} symbols")
    return table

# Standard Huffman definitions
std_dc_luminance_nrcodes = [0,0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0]
std_dc_luminance_values = list(range(12))
std_ac_luminance_nrcodes = [0,0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,0x7d]
std_ac_luminance_values = [
    0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,
    0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,
    0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,
    0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,
    0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,
    0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
    0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,
    0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,
    0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,
    0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
    0xf9,0xfa
]
std_dc_chrominance_nrcodes = [0,0,3,2,2,1,1,1,1,1,0,0,0,0,0,0,0]
std_dc_chrominance_values = list(range(12))
std_ac_chrominance_nrcodes = [0,0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,0x77]
std_ac_chrominance_values = [
    0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,
    0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,0xa1,0xb1,0xc1,0x09,0x23,0x33,0x52,0xf0,
    0x15,0x62,0x72,0xd1,0x0a,0x16,0x24,0x34,0xe1,0x25,0xf1,0x17,0x18,0x19,0x1a,0x26,
    0x27,0x28,0x29,0x2a,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,
    0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,
    0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x82,0x83,0x84,0x85,0x86,0x87,
    0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,
    0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,
    0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,
    0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
    0xf9,0xfa
]

# Build tables
HUFF_TABLE_LUMA_DC = build_huffman_table(std_dc_luminance_nrcodes, std_dc_luminance_values, 'dc')
HUFF_TABLE_LUMA_AC = build_huffman_table(std_ac_luminance_nrcodes, std_ac_luminance_values, 'ac')
HUFF_TABLE_CHROMA_DC = build_huffman_table(std_dc_chrominance_nrcodes, std_dc_chrominance_values, 'dc')
HUFF_TABLE_CHROMA_AC = build_huffman_table(std_ac_chrominance_nrcodes, std_ac_chrominance_values, 'ac')

print(f"Lengths: DC_Y={len(std_dc_luminance_values)}, AC_Y={len(std_ac_luminance_values)}, DC_C={len(std_dc_chrominance_values)}, AC_C={len(std_ac_chrominance_values)}")

# … rest of encoder unchanged …


# Helper to get the correct Huffman tables based on channel type
def get_huffman_tables(channel_type):
    if channel_type == 'Y':
        return HUFF_TABLE_LUMA_DC, HUFF_TABLE_LUMA_AC
    elif channel_type in ('Cb', 'Cr'): # Cb and Cr use Chroma tables
        return HUFF_TABLE_CHROMA_DC, HUFF_TABLE_CHROMA_AC
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")


# --- Bitstream Handling Class ---

class BitStreamWriter:
    """Writes bits to a byte stream, handling byte stuffing (0xFF -> 0xFF 0x00)."""
    def __init__(self):
        self._buffer = bytearray() # Stores the output bytes
        self._current_byte = 0    # The byte currently being built (integer 0-255)
        self._bits_in_byte = 0    # Number of bits added to the current byte (0-7)

    def write_bits(self, value, num_bits):
        """Writes the specified number of least significant bits from value."""
        if num_bits < 0:
             raise ValueError("Number of bits cannot be negative")
        if num_bits == 0:
             return # Nothing to write for 0 bits

        # Ensure we only consider the relevant least significant bits
        # Use a mask: (1 << num_bits) - 1 creates a bitmask like 0b111 for num_bits=3
        value &= (1 << num_bits) - 1

        # Write bits from MSB to LSB of the 'value'
        # Iterate num_bits times
        for i in range(num_bits - 1, -1, -1): # Loop from num_bits-1 down to 0
            # Get the i-th bit of the value
            bit = (value >> i) & 1

            # Add the bit to the current byte being built
            self._current_byte = (self._current_byte << 1) | bit
            self._bits_in_byte += 1

            # If the current byte is full (8 bits)
            if self._bits_in_byte == 8:
                # Append the complete byte to the buffer
                self._buffer.append(self._current_byte)

                # JPEG Byte Stuffing: If the byte is 0xFF, append an extra 0x00
                if self._current_byte == 0xFF:
                    self._buffer.append(0x00)

                # Reset for the next byte
                self._current_byte = 0
                self._bits_in_byte = 0

    def get_bytes(self):
        """Returns the final byte stream, padding the last byte if necessary."""
        # If there are remaining bits in the last byte (byte is not full)
        if self._bits_in_byte > 0:
            # JPEG standard padding: Pad with 1s to fill the last byte (bits 765...10)
            # The padding is (8 - self._bits_in_byte) number of 1s.
            remaining_bits = 8 - self._bits_in_byte
            # Create padding bits: (1 << remaining_bits) - 1 gives a mask of 1s
            padding_bits = (1 << remaining_bits) - 1
            # Shift current byte left and OR with padding bits
            self._current_byte = (self._current_byte << remaining_bits) | padding_bits
            self._buffer.append(self._current_byte)

        # Return the contents of the buffer as an immutable bytes object
        return bytes(self._buffer)


# --- Image Processing Steps (from previous code) ---

def rgb_to_ycbcr(img):
    # ... (this function is correct)
    xform = np.array([[0.299, 0.587, 0.114],
                      [-0.1687, -0.3313, 0.5],
                      [0.5, -0.4187, -0.0813]])
    ycbcr = img @ xform.T
    ycbcr[:, :, [1, 2]] += 128
    return ycbcr

def block_split(img, block_size=8):
    # ... (this function is correct)
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
    Applies 2D DCT on an 8x8 block (expects float input).
    """
    # DCT input should typically be float
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def quantize(block, quant_matrix=QUANTIZATION_MATRIX_LUMA):
    """
    Quantizes DCT coefficients by dividing by the quantization matrix and rounding.
    """
    # Quantization input is float (from DCT), output is int32
    return np.round(block / quant_matrix).astype(np.int32)


def zigzag_scan(block):
    """
    Converts an 8x8 block (expects int32) to a 1D array using zig-zag order.
    """
    # Zig-zag input is int32 from quantization
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
    return np.array([block[i, j] for i, j in zigzag_index], dtype=np.int32) # Ensure int32 output


def dc_encode_symbol(dc_diff):
    """Encodes DC difference into (size, amplitude value)."""
    # The symbol for DC is just the size category
    return (get_size_category(dc_diff), dc_diff)


def ac_encode_symbols(acs):
    """
    Encodes AC coefficients using RLE and returns a list of (symbol, amplitude value) tuples.
    Symbols are (Run, Size), special cases EOB ((0, 0)), ZRL ((15, 0)).
    """
    symbols_and_amplitudes = []
    zero_run = 0
    # Iterate through the 63 AC coefficients (indices 1 to 63 after zig-zag)
    for i in range(63):
        coeff = acs[i]

        if coeff == 0:
            zero_run += 1
        else:
            # Handle runs of zeros longer than 15 by inserting ZRLs
            while zero_run > 15:
                symbols_and_amplitudes.append(((15, 0), None)) # ZRL symbol, no amplitude
                zero_run -= 16

            # Encode the non-zero AC coefficient
            size = get_size_category(coeff)
            # The symbol is a tuple (Run of Zeros before this coeff, Size Category of coeff)
            symbol = (zero_run, size)
            amplitude = coeff # The amplitude is the actual value
            symbols_and_amplitudes.append((symbol, amplitude))

            zero_run = 0 # Reset zero run after a non-zero coefficient

    # After iterating through all 63 ACs, if there are trailing zeros, add EOB
    # EOB symbol is (0, 0), no amplitude.
    if zero_run > 0 or len(symbols_and_amplitudes) == 0: # If list is empty, all 63 were zero
         symbols_and_amplitudes.append(((0, 0), None)) # EOB symbol

    return symbols_and_amplitudes


# --- Main Encoding Function (Modified for Huffman Encoding) ---

def encode_image(img):
    """
    Full encoding pipeline producing Huffman-encoded bitstream bytes per channel.
    - Convert to YCbCr (if color)
    - Split into blocks
    - Apply DCT
    - Quantize
    - Zig-zag scan
    - Perform DC differential and AC RLE encoding (get symbols/amplitudes)
    - Perform Huffman encoding (write codes and amplitude bits to bitstream)
    - Return list of channel data with bitstream bytes
    """
    # If color image, convert
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = rgb_to_ycbcr(img)
        channels = 3
    elif len(img.shape) == 2: # Grayscale
        channels = 1
        # Ensure grayscale image is treated as 2D for block_split etc.
        img = img[:, :]
    else:
         raise ValueError("Input image must be grayscale (2D) or 3-channel color (3D)")


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
            channel_data_2d = img[:, :] # Already sliced above, but keep for clarity
            channel_type = 'Y' # Grayscale channel is treated as Luminance

        # Level shift by -128
        # Use float32 for processing before quantization
        channel_shifted = channel_data_2d.astype(np.float32) - 128

        blocks = block_split(channel_shifted)

        # Get the correct Huffman tables for this channel type
        huff_dc_table, huff_ac_table = get_huffman_tables(channel_type)

        # Initialize a bitstream writer for this channel
        bitstream_writer = BitStreamWriter()

        # --- Encode Blocks ---
        for block in blocks:
            # DCT
            dct_block = dct_2d(block)

            # Select quantization matrix
            if channel_type in ('Cb', 'Cr'):
                quant_matrix = QUANTIZATION_MATRIX_CHROMA
            else: # 'Y' (Luminance)
                quant_matrix = QUANTIZATION_MATRIX_LUMA

            # Quantize
            quant_block = quantize(dct_block, quant_matrix=quant_matrix)

            # Zig-zag scan
            zigzag_block = zigzag_scan(quant_block)

            # --- DC Encoding (Differential + Huffman) ---
            current_dc = zigzag_block[0]
            dc_diff = current_dc - last_dc[ch_index]
            last_dc[ch_index] = current_dc # Update last_dc for this channel

            # Get the DC symbol (size category) and amplitude (the difference value)
            # The symbol for DC is just the size category (an integer)
            dc_symbol = get_size_category(dc_diff)
            dc_amplitude = dc_diff # The amplitude is the actual difference value


            # Look up the Huffman code for the DC symbol (size category - integer 0-11)
            if dc_symbol not in huff_dc_table:
                 # This indicates an unexpected DC size category (>11) or a table build error
                 raise ValueError(f"DC symbol size {dc_symbol} (from dc_diff {dc_diff}) not found in {channel_type} DC Huffman table! Max supported size is 11.")
            dc_huff_len, dc_huff_code = huff_dc_table[dc_symbol] # Table maps symbol (int) -> (length, code)

            # Write the DC Huffman code to the bitstream
            bitstream_writer.write_bits(dc_huff_code, dc_huff_len)

            # Write the DC amplitude bits (only if size > 0)
            if dc_symbol > 0:
                dc_amplitude_bits = get_amplitude_bits(dc_amplitude, dc_symbol)
                bitstream_writer.write_bits(dc_amplitude_bits, dc_symbol) # Size is the number of bits for amplitude


            # --- AC Encoding (RLE + Huffman) ---
            ac_coeffs = zigzag_block[1:] # Get the 63 AC coefficients
            # Get the list of (symbol, amplitude value) pairs from RLE
            # AC symbols are tuples (Run, Size) or (0,0) for EOB
            ac_symbols_and_amplitudes = ac_encode_symbols(ac_coeffs)

            # Iterate through the AC symbols and encode them using Huffman
            for symbol, amplitude in ac_symbols_and_amplitudes:
                 # Look up the Huffman code for the AC symbol ((Run, Size) or (0,0))
                 if symbol not in huff_ac_table:
                      # This indicates an unexpected AC symbol (e.g., size > 10 or run > 15 not handled by ZRL)
                      raise ValueError(f"AC symbol {symbol} not found in {channel_type} AC Huffman table!")

                 ac_huff_len, ac_huff_code = huff_ac_table[symbol] # Table maps symbol (tuple) -> (length, code)

                 # Write the AC Huffman code to the bitstream
                 bitstream_writer.write_bits(ac_huff_code, ac_huff_len)

                 # Write the AC amplitude bits if it's a non-zero symbol (not EOB or ZRL)
                 # EOB symbol is (0,0), ZRL is (15,0). Regular symbols are (Run, Size) where Size > 0.
                 if symbol != (0,0) and symbol != (15,0):
                     # Size is the second element of the (Run, Size) tuple symbol
                     size = symbol[1]
                     if size > 0: # Should always be > 0 for non-zero symbols
                        ac_amplitude_bits = get_amplitude_bits(amplitude, size)
                        bitstream_writer.write_bits(ac_amplitude_bits, size) # Size is the number of bits for amplitude


        # Get the final byte data for this channel's bitstream
        channel_bitstream_bytes = bitstream_writer.get_bytes()

        # Store the original dimensions, channel type, and the byte data
        encoded_channels.append((h, w, channel_type, channel_bitstream_bytes))

    # The return value is a list containing data for each channel:
    # [(h_y, w_y, 'Y', bitstream_bytes_y), (h_cb, w_cb, 'Cb', bitstream_bytes_cb), ...]
    return encoded_channels
