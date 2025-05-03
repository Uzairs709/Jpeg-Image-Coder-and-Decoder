"""
main_window.py
PyQt5 GUI for JPEG Compression/Decompression
"""

import sys
import os
# Add parent directory to sys.path to import Coder and Decoder
# Adjust the path based on your exact folder structure if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog,
    QMessageBox, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
from PIL import Image
import pickle # Import the pickle module for saving/loading data structures

# Import your coder and decoder
from Coder.encoder import encode_image
from Decoder.decoder import decode_image


class JPEGCompressorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simplified JPEG Compressor/Decompressor")
        self.setGeometry(100, 100, 800, 600) # Make window a bit larger

        # Main Layout
        main_layout = QVBoxLayout()

        # Image Display Label
        self.label = QLabel("No image loaded")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("border: 1px solid gray;") # Add border to visualize label size
        self.label.setScaledContents(True) # Automatically scale pixmap to fit label size

        # Stats Label (Optional - Alternative to Message Box)
        # You could add a dedicated label here to always show stats
        # self.stats_label = QLabel("")
        # self.stats_label.setAlignment(Qt.AlignCenter)
        # main_layout.addWidget(self.stats_label)


        # Buttons Layout
        button_layout = QHBoxLayout() # Use horizontal layout for buttons

        # Buttons
        self.btn_load = QPushButton("Upload Image")
        self.btn_compress = QPushButton("Compress Image")
        self.btn_decompress = QPushButton("Decompress Data")
        self.btn_save_compressed = QPushButton("Save Compressed Data")
        self.btn_load_compressed = QPushButton("Load Compressed Data")


        self.btn_compress.setEnabled(False)
        self.btn_decompress.setEnabled(False)
        self.btn_save_compressed.setEnabled(False) # Enabled after compression

        # Add buttons to horizontal layout
        button_layout.addWidget(self.btn_load)
        button_layout.addWidget(self.btn_compress)
        button_layout.addWidget(self.btn_decompress)
        button_layout.addWidget(self.btn_save_compressed)
        button_layout.addWidget(self.btn_load_compressed)


        # Add widgets/layouts to main layout
        main_layout.addWidget(self.label, 1) # Image label takes more space
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # Connect buttons to actions
        self.btn_load.clicked.connect(self.load_image)
        self.btn_compress.clicked.connect(self.compress_image)
        self.btn_decompress.clicked.connect(self.decompress_image)
        # Connect new save/load buttons
        self.btn_save_compressed.clicked.connect(self.save_compressed_data)
        self.btn_load_compressed.clicked.connect(self.load_compressed_data)


        # Internal variables
        self.original_image = None # Stores the original loaded image (NumPy array)
        # Stores the compressed data: list of (h, w, channel_type, bitstream_bytes) tuples
        self.encoded_data = None

    # Helper function to format bytes into human-readable string
    def format_bytes(self, size_in_bytes):
        """Formats a size in bytes to a human-readable string (B, KB, MB)."""
        if size_in_bytes is None:
             return "N/A"
        # Ensure size is an integer for comparison
        size_in_bytes = int(size_in_bytes)
        if size_in_bytes < 1024:
            return f"{size_in_bytes} B"
        elif size_in_bytes < 1024**2:
            return f"{size_in_bytes / 1024:.2f} KB"
        elif size_in_bytes < 1024**3:
            return f"{size_in_bytes / (1024**2):.2f} MB"
        else:
            return f"{size_in_bytes / (1024**3):.2f} GB"


    def load_image(self):
        """Loads an image file and displays it."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif)",
            options=options
        )
        if file_name:
            try:
                pil_img = Image.open(file_name)

                # Convert to RGB or L (grayscale) if needed
                if pil_img.mode != 'RGB' and pil_img.mode != 'L':
                     # For consistency with encoder, convert other modes (like RGBA, P)
                     pil_img = pil_img.convert('RGB') if pil_img.mode in ('RGBA', 'P') else pil_img.convert('L')


                self.original_image = np.array(pil_img)

                self.show_image(self.original_image)
                QMessageBox.information(self, "Image Loaded", f"Successfully loaded: {os.path.basename(file_name)}")

                # Reset state for compression/decompression
                self.encoded_data = None # Clear any previous compressed data
                self.label.setText("") # Clear potential text message
                # if hasattr(self, 'stats_label'): self.stats_label.setText("") # Clear stats if using a label

                # Update button states
                self.btn_compress.setEnabled(True)
                self.btn_decompress.setEnabled(False) # Cannot decompress until compressed/loaded
                self.btn_save_compressed.setEnabled(False) # Cannot save until compressed


            except Exception as e:
                QMessageBox.critical(self, "Error Loading Image", f"Failed to load image: {e}")
                self.original_image = None
                self.encoded_data = None
                self.label.setText("Failed to load image")
                # if hasattr(self, 'stats_label'): self.stats_label.setText("")
                self.btn_compress.setEnabled(False)
                self.btn_decompress.setEnabled(False)
                self.btn_save_compressed.setEnabled(False)


    def compress_image(self):
        """Compresses the loaded image using the encoder and shows stats."""
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Please upload an image first.")
            return

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor) # Show busy cursor

            # --- Calculate Original Size ---
            # Size of the NumPy array in bytes (Height * Width * Channels * Bytes_per_pixel)
            original_size = self.original_image.nbytes
            original_size_formatted = self.format_bytes(original_size)

            # Perform Compression - This now returns a list of (h, w, channel_type, bitstream_bytes)
            encoded_channels_data = encode_image(self.original_image)

            # --- Calculate Compressed Size ---
            # The 'compressed size' is the sum of the lengths of the bitstream byte data for all channels
            compressed_size = sum(len(channel_data[3]) for channel_data in encoded_channels_data)
            compressed_size_formatted = self.format_bytes(compressed_size)

            # Store the encoded data (the list of tuples including byte data)
            self.encoded_data = encoded_channels_data


            QApplication.restoreOverrideCursor() # Restore cursor

            # --- Calculate and Display Stats ---
            stats_message = "No size data available." # Default message
            # Calculate stats only if original size is meaningful and compressed data was produced
            if original_size > 0 and self.encoded_data is not None and compressed_size is not None:
                 # Avoid division by zero if compressed size is 0 (e.g., for an empty image)
                 compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
                 size_reduction_percent = (1 - compressed_size / original_size) * 100

                 stats_message = (
                     f"Original size (raw pixels): {original_size_formatted}\n" # Clarified size
                     f"Compressed size (bitstream bytes): {compressed_size_formatted}\n" # Clarified size
                     f"Compression Ratio: {compression_ratio:.2f}:1\n"
                     f"Size Reduction: {size_reduction_percent:.2f}%"
                 )
            elif original_size == 0:
                 stats_message = "Original image size is 0 bytes."
            elif self.encoded_data is None:
                 stats_message = "Compression failed, no data produced."


            QMessageBox.information(self, "Compression Complete",
                                    f"Image compressed successfully!\n\n{stats_message}")

            # if hasattr(self, 'stats_label'): self.stats_label.setText(stats_message) # Update stats label if using one


            # Update button states
            # After compression, you can now decompress or save the data
            self.btn_decompress.setEnabled(True)
            self.btn_save_compressed.setEnabled(True)
            # Keep compress enabled if user wants to compress another loaded image

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Compression Error", f"Failed to compress image: {e}")
            # In case of error, clear encoded data and disable subsequent steps
            self.encoded_data = None
            self.btn_decompress.setEnabled(False)
            self.btn_save_compressed.setEnabled(False)
            # if hasattr(self, 'stats_label'): self.stats_label.setText("Compression failed.")


    def decompress_image(self):
        """Decompresses the encoded data (list of tuples with bitstream bytes) and displays the result."""
        if self.encoded_data is None:
            QMessageBox.warning(self, "Warning", "No compressed data to decompress. Load compressed data or compress an image first.")
            return

        # --- Validate Stored Data Format ---
        # Check if the stored data is in the expected format (list of tuples with bytes)
        if not isinstance(self.encoded_data, list) or not all(
            isinstance(item, tuple) and
            len(item) == 4 and
            isinstance(item[0], int) and # Check height
            isinstance(item[1], int) and # Check width
            isinstance(item[2], str) and # Check channel_type string
            isinstance(item[3], bytes)   # Check bitstream bytes
            for item in self.encoded_data
        ):
             QMessageBox.warning(self, "Warning", "Stored data is not in the correct bitstream format. Please load a valid compressed file or compress an image first.")
             self.encoded_data = None # Clear invalid data
             self.btn_decompress.setEnabled(False)
             self.btn_save_compressed.setEnabled(False)
             # if hasattr(self, 'stats_label'): self.stats_label.setText("Decompression failed: Invalid data format.")
             return


        try:
            QApplication.setOverrideCursor(Qt.WaitCursor) # Show busy cursor

            # Pass the list of channel data containing bitstream bytes to the decoder
            decoded_image = decode_image(self.encoded_data)

            QApplication.restoreOverrideCursor() # Restore cursor

            self.show_image(decoded_image)
            QMessageBox.information(self, "Success", "Data decompressed successfully!")
            self.label.setText("") # Clear text message
            # if hasattr(self, 'stats_label'): self.stats_label.setText("") # Clear stats if using a label


            # Button states: Can now compress the decompressed image (lossy process)
            # Update self.original_image for potential re-compression
            self.original_image = decoded_image
            self.encoded_data = None # Clear compressed data after decompression

            self.btn_compress.setEnabled(True)
            self.btn_decompress.setEnabled(False) # Already decompressed this data
            self.btn_save_compressed.setEnabled(False) # Cannot save pixel data with 'Save Compressed Data' button


        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Decompression Error", f"Failed to decompress data: {e}")
            # If decompression fails, clear the encoded data state
            self.encoded_data = None
            self.label.setText("Failed to decompress data")
            # if hasattr(self, 'stats_label'): self.stats_label.setText("Decompression failed.")
            self.btn_decompress.setEnabled(False)
            self.btn_save_compressed.setEnabled(False)


    def save_compressed_data(self):
        """Saves the in-memory encoded data structure (list of tuples with bytes) to a file using pickle."""
        if self.encoded_data is None:
            QMessageBox.warning(self, "Warning", "No compressed data to save.")
            return

        # --- Validate Stored Data Format before saving ---
        if not isinstance(self.encoded_data, list) or not all(
            isinstance(item, tuple) and
            len(item) == 4 and
            isinstance(item[0], int) and
            isinstance(item[1], int) and
            isinstance(item[2], str) and
            isinstance(item[3], bytes)
            for item in self.encoded_data
        ):
             QMessageBox.warning(self, "Warning", "Stored data is not in the correct bitstream format for saving. Compression might have failed or data was corrupted.")
             return


        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Compressed Data",
            "",
            "My JPEG Data (*.myjpeg);;All Files (*)",
            options=options
        )
        if file_name:
            if not file_name.lower().endswith(".myjpeg"):
                file_name += ".myjpeg"
            try:
                with open(file_name, 'wb') as f:
                    # Pickle saves the list structure, including the embedded byte data
                    pickle.dump(self.encoded_data, f)
                QMessageBox.information(self, "Success", f"Compressed data saved successfully to:\n{os.path.basename(file_name)}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save compressed data: {e}")


    def load_compressed_data(self):
        """Loads compressed data structure (list of tuples with bytes) from a file using pickle."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Load Compressed Data",
            "",
            "My JPEG Data (*.myjpeg);;All Files (*)",
            options=options
        )
        # Check if the user selected a file (file_name is not empty string)
        if file_name:
            # Start the try block here. It handles potential errors like file not found,
            # permission issues, or errors during pickle loading.
            try:
                with open(file_name, 'rb') as f:
                    loaded_data = pickle.load(f) # Load the structure

                # --- Perform Validation ---
                # Check if the loaded data is in the expected bitstream format.
                # This validation happens *after* successfully loading with pickle.
                is_valid_format = False
                # Check if it's a list and not empty
                if isinstance(loaded_data, list) and len(loaded_data) > 0:
                     # Check if every item in the list is a tuple of length 4
                     # and if the elements inside the tuple are of the expected types.
                     if all(isinstance(item, tuple) and len(item) == 4 and
                            isinstance(item[0], int) and isinstance(item[1], int) and
                            isinstance(item[2], str) and item[2] in ('Y', 'Cb', 'Cr') and # Check channel type string
                            isinstance(item[3], bytes)
                            for item in loaded_data):
                          # Additional check: number of channels should be 1 or 3
                          if len(loaded_data) in (1, 3):
                               is_valid_format = True


                # --- Handle Validation Result ---
                # This if/else block is *inside* the try block.
                if is_valid_format:
                     # Success branch: Data is valid
                     self.encoded_data = loaded_data
                     QMessageBox.information(self, "Success", f"Compressed data loaded successfully from:\n{os.path.basename(file_name)}")

                     self.label.setPixmap(QPixmap())
                     self.label.setText("Compressed data loaded. Click 'Decompress Data'.")
                     # if hasattr(self, 'stats_label'): self.stats_label.setText("Compressed data loaded.")
                     self.original_image = None # Clear original image state

                     # Update button states for successful load
                     self.btn_load.setEnabled(True) # Can load a new original image
                     self.btn_compress.setEnabled(False) # Cannot compress bitstream data directly
                     self.btn_decompress.setEnabled(True) # Can now decompress
                     self.btn_save_compressed.setEnabled(True) # Can potentially re-save

                else:
                     # Validation failed branch: Data loaded but format is incorrect
                     QMessageBox.warning(self, "Warning", "Invalid compressed data format loaded.")
                     self.encoded_data = None # Ensure invalid data is not kept
                     self.label.setText("Invalid compressed data.")
                     # if hasattr(self, 'stats_label'): self.stats_label.setText("Load failed: Invalid format.")
                     # Update button states for load failure
                     self.btn_load.setEnabled(True)
                     self.btn_compress.setEnabled(False)
                     self.btn_decompress.setEnabled(False)
                     self.btn_save_compressed.setEnabled(False)

            # --- Handle Exceptions ---
            # This except block is associated with the try block above.
            # It must be at the same indentation level as the 'try' line.
            except Exception as e:
                # Handle any error that occurred during the try block (file not found, pickle error, etc.)
                QMessageBox.critical(self, "Load Error", f"Failed to load compressed data: {e}")
                self.encoded_data = None # Ensure potentially corrupted data is not kept
                self.label.setText("Error loading compressed data.")
                # if hasattr(self, 'stats_label'): self.stats_label.setText("Load failed.")
                # Update button states for load failure
                self.btn_load.setEnabled(True)
                self.btn_compress.setEnabled(False)
                self.btn_decompress.setEnabled(False)
                self.btn_save_compressed.setEnabled(False)

        # Optional: An 'else' block here would be for the 'if file_name:' statement
        # It would execute if file_name was an empty string (user cancelled dialog)
        # else:
        #     print("File selection cancelled.")
        #     pass # Or reset state, etc.


    def show_image(self, img_array):
        """Displays a NumPy image array in the QLabel."""
        if img_array is None:
            self.label.setPixmap(QPixmap()) # Clear display
            self.label.setText("No image loaded")
            return

        height, width = img_array.shape[:2]
        # Ensure image data is in a suitable format for QImage before creating QImage
        if img_array.ndim == 2:  # Grayscale (L)
            bytes_per_line = width
            # Must be contiguous (C-order) and uint8
            q_image = QImage(img_array.astype(np.uint8).copy().data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif img_array.ndim == 3 and img_array.shape[2] == 3: # Color (RGB)
            bytes_per_line = 3 * width
             # Must be contiguous (C-order) and uint8
            rgb_array = img_array.astype(np.uint8).copy()
            q_image = QImage(rgb_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
             # Handle unexpected dimensions gracefully
             print(f"Warning: show_image received unexpected array shape {img_array.shape}")
             self.label.setPixmap(QPixmap())
             self.label.setText(f"Cannot display image of shape {img_array.shape}")
             return


        if q_image.isNull():
            print("Failed to create QImage.")
            self.label.setPixmap(QPixmap())
            self.label.setText("Error displaying image")
            return

        pixmap = QPixmap.fromImage(q_image)

        # Scale the pixmap to fit the label while maintaining aspect ratio
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


# Standard boilerplate to run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = JPEGCompressorApp()
    window.show()
    sys.exit(app.exec_())