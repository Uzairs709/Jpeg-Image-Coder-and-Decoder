import numpy as np
from scipy.fftpack import dct, idct
import logging

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


logging.basicConfig(level=logging.INFO)


# --- Helper functions for Size Category and Amplitude encoding/decoding ---
def get_size_category(value):
    if value == 0:
        return 0
    size = 0
    abs_value = abs(value)
    while abs_value > (1 << size) -1:
        size += 1
        if size > 11: # Max DC size is 11, max AC size is 10. Should not exceed this.
             logging.warning(f"Calculated size category {size} for value {value}. Capping at 11.")
             size = 11
             break
    return size


def decode_amplitude(size, amplitude_bits):
    if size == 0:
        return 0 

    threshold = 1 << (size - 1)

    if amplitude_bits >= threshold:
        return amplitude_bits
    else:
        return amplitude_bits - (1 << size) + 1


# --- Function to build Huffman tables from standard lengths and values ---
def build_huffman_table(bits, values, table_type):
    huffman_table = {}
    code = 0
    value_pos = 0
    symbols_added_count = 0 

    expected_symbols_count = sum(bits[1:17])

    assert len(values) == expected_symbols_count, \
        f"Error building {table_type} Huffman table: 'values' list has unexpected length. Expected {expected_symbols_count}, but got {len(values)}. Check list content."


    for bit_length in range(1, 17):
        if bit_length < len(bits):
             num_codes_of_length = bits[bit_length]
        else:
             num_codes_of_length = 0

        for i in range(num_codes_of_length):
            if value_pos >= len(values):
                 logging.error(f"Index out of range in build_huffman_table for {table_type} table.")
                 logging.error(f"bit_length: {bit_length}, code_index: {i}, value_pos: {value_pos}, len(values): {len(values)}")
                 raise IndexError(f"list index out of range in build_huffman_table: value_pos {value_pos} >= len(values) {len(values)} at bit_length {bit_length}, code_index {i}")


            raw_symbol_value = values[value_pos]

            if table_type == 'dc':
                symbol = int(raw_symbol_value)
                if not (0 <= symbol <= 11):
                     logging.warning(f"Unexpected raw DC symbol value {raw_symbol_value} encountered.")

            elif table_type == 'ac':
                 raw_val_int = int(raw_symbol_value)
                 if raw_val_int == 0x00:
                     symbol = (0, 0)
                 elif raw_val_int == 0xF0:
                      symbol = (15, 0)
                 elif raw_val_int != 0x00 and raw_val_int != 0xF0:
                      run = (raw_val_int >> 4) & 0x0F 
                      size = raw_val_int & 0x0F 
                      symbol = (run, size)
                 else:
                      logging.warning(f"Truly unexpected raw AC symbol value {raw_symbol_value} ({raw_val_int}) encountered in {table_type} values list. Skipping.")
                      value_pos += 1 
                      continue


            else:
                 raise ValueError(f"Unknown table_type: {table_type}")

            if isinstance(symbol, (int, tuple)):
                 huffman_table[symbol] = (bit_length, code)
                 symbols_added_count += 1 
            else:
                 logging.warning(f"Skipping unhashable symbol type {type(symbol)}: {symbol}")

            code += 1
            value_pos += 1

        code <<= 1

    assert symbols_added_count == expected_symbols_count, \
        f"Error building {table_type} Huffman table: Expected {expected_symbols_count} symbols, but added {symbols_added_count}. This indicates an issue processing the 'values' list content."


    logging.info(f"Finished building {table_type} table. Total symbols added: {len(huffman_table)}")


    return huffman_table

# --- Standard JPEG Huffman Table Data (as defined in the spec) 
std_dc_luminance_nrcodes = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
std_dc_luminance_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

std_ac_luminance_nrcodes = [0,0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,0x7d] 
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
    0xf9, 0xfa
]

std_dc_chrominance_nrcodes = [0, 0, 3, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
std_dc_chrominance_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

std_ac_chrominance_nrcodes = [0,0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,0x77]
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
    0xf9, 0xfa
]

# Build tables
HUFF_TABLE_LUMA_DC = build_huffman_table(std_dc_luminance_nrcodes, std_dc_luminance_values, 'dc')
HUFF_TABLE_LUMA_AC = build_huffman_table(std_ac_luminance_nrcodes, std_ac_luminance_values, 'ac')
HUFF_TABLE_CHROMA_DC = build_huffman_table(std_dc_chrominance_nrcodes, std_dc_chrominance_values, 'dc')
HUFF_TABLE_CHROMA_AC = build_huffman_table(std_ac_chrominance_nrcodes, std_ac_chrominance_values, 'ac')

print(f"Lengths: DC_Y={len(std_dc_luminance_values)}, AC_Y={len(std_ac_luminance_values)}, DC_C={len(std_dc_chrominance_values)}, AC_C={len(std_ac_chrominance_values)}")


# Helper to get the correct Huffman tables based on channel type
def get_huffman_tables(channel_type):
    if channel_type == 'Y':
        return HUFF_TABLE_LUMA_DC, HUFF_TABLE_LUMA_AC
    elif channel_type in ('Cb', 'Cr'): # Cb and Cr use Chroma tables
        return HUFF_TABLE_CHROMA_DC, HUFF_TABLE_CHROMA_AC
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")


# --- Bitstream Handling Classes
class BitStreamReader:
    def __init__(self, byte_data):
        self._byte_data = byte_data
        self._byte_index = 0
        self._bits_in_byte = 0
        self._current_byte = 0

        # Load the first byte to start if data exists
        if len(self._byte_data) > 0:
            self._current_byte = self._byte_data[self._byte_index]
            logging.info(f"Initialized BitStreamReader. First byte: {self._current_byte:02x} at index {self._byte_index}")


    def read_bits(self, num_bits):
        if num_bits <= 0:
            return 0

        result = 0
        # logging.info(f"Attempting to read {num_bits} bits. Current byte index: {self._byte_index}, bits in byte: {self._bits_in_byte}")

        for i in range(num_bits):
            # If all bits in the current byte are read, load the next byte
            if self._bits_in_byte == 8:
                self._byte_index += 1
                if self._byte_index < len(self._byte_data) and self._byte_data[self._byte_index - 1] == 0xFF and self._byte_data[self._byte_index] == 0x00:
                     logging.info(f"Byte unstuffing: Skipping byte {self._byte_index} (0x00) after 0xFF at index {self._byte_index - 1}")
                     self._byte_index += 1

                # Check if we ran out of data after potential unstuffing
                if self._byte_index >= len(self._byte_data):
                    logging.warning(f"Ran out of bytes while reading bit {i+1}/{num_bits}. Returning None.")
                    return None # Indicate end of stream/error

                # Load the next byte
                self._current_byte = self._byte_data[self._byte_index]
                self._bits_in_byte = 0 # Reset bit counter for the new byte
                logging.info(f"Loaded new byte: {self._current_byte:02x} at index {self._byte_index}")


            bit = (self._current_byte >> (7 - self._bits_in_byte)) & 1
            result = (result << 1) | bit
            self._bits_in_byte += 1


        return result

    def has_more_bits(self):
        bits_remaining = len(self._byte_data) * 8 - (self._byte_index * 8 + self._bits_in_byte)
        return self._byte_index < len(self._byte_data)


    def decode_symbol(self, huff_table):
        current_value = 0
        current_length = 0

        reverse_lookup = {}
        for symbol, (length, code) in huff_table.items():
             if (code, length) in reverse_lookup:
                  logging.error(f"Duplicate Huffman code entry for {(code, length)}")
                  return None
             reverse_lookup[(code, length)] = symbol


        max_code_length = 0
        if huff_table:
            max_code_length = max(length for symbol, (length, code) in huff_table.items())
            if max_code_length > 16: max_code_length = 16
        else:
             logging.error("decode_symbol called with empty Huffman table.")
             return None


        for _ in range(max_code_length):
            bit = self.read_bits(1)
            if bit is None: 
                 return None

            current_value = (current_value << 1) | bit
            current_length += 1

            if (current_value, current_length) in reverse_lookup:
                symbol = reverse_lookup[(current_value, current_length)]
                return symbol

        logging.error(f"Could not decode Huffman symbol after {max_code_length} bits. Current value: {current_value}, length: {current_length}")
        return None


# --- Inverse Transformation and Block Merging
def inverse_zigzag(arr, block_size=8):
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
    return block * quant_matrix


def idct_2d(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def block_merge(blocks, image_shape, block_size=8):
    h, w = image_shape
    num_blocks_w = (w + block_size - 1) // block_size
    num_blocks_h = (h + block_size - 1) // block_size

    padded_h = num_blocks_h * block_size
    padded_w = num_blocks_w * block_size

    img_padded = np.zeros((padded_h, padded_w), dtype=np.float32)

    idx = 0
    for i in range(0, padded_h, block_size):
        for j in range(0, padded_w, block_size):
            if idx < len(blocks):
                 img_padded[i:i+block_size, j:j+block_size] = blocks[idx].astype(np.float32)
            idx += 1

    img = img_padded[:h, :w]

    return img


def ycbcr_to_rgb(img):
    img = img.astype(np.float32)

    xform = np.array([[1, 0, 1.402],
                      [1, -0.34414, -0.71414],
                      [1, 1.772, 0]])
    img[:, :, [1, 2]] -= 128
    rgb = img @ xform.T
    rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8)


def dc_decode_value(last_dc, dc_amplitude):
    return last_dc + dc_amplitude


def decode_image(encoded_channels):
    reconstructed = []

    last_dc = [0] * len(encoded_channels)


    for ch_index, channel_data in enumerate(encoded_channels):
        h, w, channel_type, bitstream_bytes = channel_data

        huff_dc_table, huff_ac_table = get_huffman_tables(channel_type)

        bitstream_reader = BitStreamReader(bitstream_bytes)

        blocks = []

        if channel_type in ('Cb', 'Cr'):
            quant_matrix = QUANTIZATION_MATRIX_CHROMA
        else:
            quant_matrix = QUANTIZATION_MATRIX_LUMA

        block_size = 8
        num_blocks_h = (h + block_size - 1) // block_size
        num_blocks_w = (w + block_size - 1) // block_size
        expected_num_blocks = num_blocks_h * num_blocks_w
        blocks_decoded_count = 0

        logging.info(f"Starting decoding for {channel_type} channel. Expected blocks: {expected_num_blocks}")

        while blocks_decoded_count < expected_num_blocks: 

            # --- Decode DC ---
            dc_symbol = bitstream_reader.decode_symbol(huff_dc_table)
            if dc_symbol is None:
                 logging.error(f"Failed to decode DC symbol for block {blocks_decoded_count} in {channel_type} channel. Likely corrupted data or unexpected end of stream.")
                 break # Stop decoding this channel


            dc_amplitude_bits = 0
            if dc_symbol > 0:
                dc_amplitude_bits = bitstream_reader.read_bits(dc_symbol)
                if dc_amplitude_bits is None: # Check for None
                     logging.error(f"Failed to read DC amplitude bits (size {dc_symbol}) for block {blocks_decoded_count} in {channel_type} channel. Stopping decoding for this channel.")
                     break

            dc_amplitude = decode_amplitude(dc_symbol, dc_amplitude_bits)

            current_dc = last_dc[ch_index] + dc_amplitude
            last_dc[ch_index] = current_dc
            zigzag_coeffs = np.zeros(64, dtype=np.int32)
            zigzag_coeffs[0] = current_dc # Place the decoded DC value

            ac_idx_in_block = 1

            while ac_idx_in_block < 64:
                ac_symbol = bitstream_reader.decode_symbol(huff_ac_table)
                if ac_symbol is None:
                     logging.error(f"Failed to decode AC symbol for block {blocks_decoded_count}, AC index {ac_idx_in_block -1} in {channel_type} channel. Likely corrupted data or unexpected end of stream.")
                     ac_idx_in_block = 64
                     break 


                if ac_symbol == (0, 0):
                    ac_idx_in_block = 64
                    break

                if ac_symbol == (15, 0):
                    run = 16
                    ac_idx_in_block += run
                    if ac_idx_in_block > 64:
                         logging.warning(f"ZRL decoding overflow at block {blocks_decoded_count}, AC index {ac_idx_in_block - run} in {channel_type} channel. Index reached {ac_idx_in_block}.")
                         ac_idx_in_block = 64
                         break
                    continue

                run, size = ac_symbol
                ac_idx_in_block += run

                if ac_idx_in_block >= 64:
                     logging.warning(f"AC run overflow at block {blocks_decoded_count}, AC index {ac_idx_in_block - run} in {channel_type} channel. Index reached {ac_idx_in_block}.")
                     ac_idx_in_block = 64
                     break


                ac_amplitude_bits = 0
                if size > 0:
                     ac_amplitude_bits = bitstream_reader.read_bits(size)
                     if ac_amplitude_bits is None: # Check for None
                          logging.error(f"Failed to read AC amplitude bits (size {size}) for block {blocks_decoded_count}, AC index {ac_idx_in_block-1} in {channel_type} channel. Stopping decoding for this channel.")
                          ac_idx_in_block = 64
                          break 

                ac_amplitude = decode_amplitude(size, ac_amplitude_bits)

                zigzag_coeffs[ac_idx_in_block] = ac_amplitude

                ac_idx_in_block += 1

            if ac_idx_in_block < 64 and bitstream_reader.read_bits(1) is None and ac_symbol is None:
                 break


            # --- Inverse Zig-zag
            quant_block = inverse_zigzag(zigzag_coeffs)

            # --- Dequantization ---
            dequant_block = dequantize(quant_block, quant_matrix=quant_matrix)

            # --- Inverse DCT ---
            block = idct_2d(dequant_block)

            # Level shift back (+128)
            block = block + 128

            blocks.append(block)
            blocks_decoded_count += 1 # Increment block count


        # --- Merge Blocks back into Channel Image
        channel_img = block_merge(blocks, (h, w))

        # Add the reconstructed channel image (still in YCbCr range if color) to the list
        reconstructed.append(channel_img)

        # After processing a channel, if not all expected blocks were decoded, issue warning
        if blocks_decoded_count < expected_num_blocks:
             logging.warning(f"Decoded only {blocks_decoded_count}/{expected_num_blocks} blocks for {channel_type} channel. Bitstream might be incomplete or corrupted.")


    # --- Stack Channels back together ---
    if len(reconstructed) == 1:
        final_img = np.clip(reconstructed[0], 0, 255).astype(np.uint8)
    else:
        img = np.stack(reconstructed, axis=-1)
        final_img = ycbcr_to_rgb(img)

    return final_img
