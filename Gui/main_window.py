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
        self.encoded_data = None # Stores the compressed data structure (list of tuples)

    # Helper function to format bytes into human-readable string
    def format_bytes(self, size_in_bytes):
        """Formats a size in bytes to a human-readable string (B, KB, MB)."""
        if size_in_bytes is None:
             return "N/A"
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
        # Allow selecting standard image formats
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif)", # Added more formats via PIL
            options=options
        )
        if file_name:
            try:
                # Use PIL to open image, which handles various formats
                pil_img = Image.open(file_name)

                # Convert to RGB if not already (PIL can load different modes)
                # This is important as our encoder expects RGB or Grayscale
                if pil_img.mode != 'RGB' and pil_img.mode != 'L': # L is grayscale
                     pil_img = pil_img.convert('RGB')

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
            # Size of the NumPy array in bytes
            original_size = self.original_image.nbytes
            original_size_formatted = self.format_bytes(original_size)

            # Perform Compression
            self.encoded_data = encode_image(self.original_image)

            # --- Calculate Compressed Size ---
            # Pickle the data structure to estimate its size if saved to a file.
            # This is a reasonable proxy for the 'compressed size' in this implementation.
            compressed_bytes = pickle.dumps(self.encoded_data)
            compressed_size = len(compressed_bytes)
            compressed_size_formatted = self.format_bytes(compressed_size)


            QApplication.restoreOverrideCursor() # Restore cursor

            # --- Calculate and Display Stats ---
            stats_message = "No size data available." # Default message
            if original_size > 0 and compressed_size is not None:
                 compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
                 # Calculate percentage reduction, handle case where original_size is 0 or same as compressed
                 if original_size > 0:
                      size_reduction_percent = (1 - compressed_size / original_size) * 100
                 else:
                      size_reduction_percent = 0

                 stats_message = (
                     f"Original size: {original_size_formatted}\n"
                     f"Compressed size (pickled data): {compressed_size_formatted}\n"
                     f"Compression Ratio: {compression_ratio:.2f}:1\n"
                     f"Size Reduction: {size_reduction_percent:.2f}%"
                 )
            elif original_size == 0:
                 stats_message = "Original image size is 0 bytes."


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
            self.encoded_data = None # Clear potentially invalid data
            self.btn_decompress.setEnabled(False) # Cannot decompress if compression failed
            self.btn_save_compressed.setEnabled(False)
            # if hasattr(self, 'stats_label'): self.stats_label.setText("Compression failed.")


    def decompress_image(self):
        """Decompresses the encoded data and displays the result."""
        if self.encoded_data is None:
            QMessageBox.warning(self, "Warning", "No compressed data to decompress. Load compressed data or compress an image first.")
            return

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor) # Show busy cursor
            decoded_image = decode_image(self.encoded_data)
            QApplication.restoreOverrideCursor() # Restore cursor

            self.show_image(decoded_image)
            QMessageBox.information(self, "Success", "Data decompressed successfully!")
            self.label.setText("") # Clear text message
            # if hasattr(self, 'stats_label'): self.stats_label.setText("") # Clear stats if using a label


            # Button states: Can now compress the decompressed image (lossy process)
            self.original_image = decoded_image # The decompressed image becomes the 'new original'
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
        """Saves the in-memory encoded data structure to a file using pickle."""
        if self.encoded_data is None:
            QMessageBox.warning(self, "Warning", "No compressed data to save.")
            return

        options = QFileDialog.Options()
        # Suggest a custom file extension
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Compressed Data",
            "",
            "My JPEG Data (*.myjpeg);;All Files (*)", # Filter for your custom extension
            options=options
        )
        if file_name:
            # Ensure the file has the correct extension if the user didn't add it
            if not file_name.lower().endswith(".myjpeg"):
                file_name += ".myjpeg"
            try:
                with open(file_name, 'wb') as f: # Open in binary write mode
                    pickle.dump(self.encoded_data, f) # Use pickle to save the data structure
                QMessageBox.information(self, "Success", f"Compressed data saved successfully to:\n{os.path.basename(file_name)}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save compressed data: {e}")


    def load_compressed_data(self):
        """Loads compressed data structure from a file using pickle."""
        options = QFileDialog.Options()
        # Filter to only show your custom file extension
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Load Compressed Data",
            "",
            "My JPEG Data (*.myjpeg);;All Files (*)", # Filter for your custom extension
            options=options
        )
        if file_name:
            try:
                with open(file_name, 'rb') as f: # Open in binary read mode
                    loaded_data = pickle.load(f) # Use pickle to load the data structure

                # Perform basic validation to check if the loaded data structure
                # resembles the expected format (list of tuples)
                is_valid_format = False
                if isinstance(loaded_data, list) and len(loaded_data) > 0:
                    # Check if first element is a tuple with 3 items
                    if isinstance(loaded_data[0], tuple) and len(loaded_data[0]) == 3:
                        # Check if the third item is a list of numpy arrays
                        if isinstance(loaded_data[0][2], list) and len(loaded_data[0][2]) > 0 and isinstance(loaded_data[0][2][0], np.ndarray):
                             is_valid_format = True # Seems valid enough for this purpose

                if is_valid_format:
                     self.encoded_data = loaded_data
                     QMessageBox.information(self, "Success", f"Compressed data loaded successfully from:\n{os.path.basename(file_name)}")

                     # Clear the image display and show a message
                     self.label.setPixmap(QPixmap()) # Clear display
                     self.label.setText("Compressed data loaded. Click 'Decompress Data'.")
                     # if hasattr(self, 'stats_label'): self.stats_label.setText("Compressed data loaded.")

                     self.original_image = None # Clear original image state, as we loaded compressed data

                     # Update button states
                     self.btn_load.setEnabled(True) # Can load a new original image
                     self.btn_compress.setEnabled(False) # Cannot compress data structure directly
                     self.btn_decompress.setEnabled(True) # Can now decompress the loaded data
                     self.btn_save_compressed.setEnabled(True) # Can potentially re-save the loaded data


                else:
                    QMessageBox.warning(self, "Warning", "Invalid compressed data format loaded.")
                    self.encoded_data = None # Ensure invalid data is not kept
                    self.label.setText("Invalid compressed data.")
                    # if hasattr(self, 'stats_label'): self.stats_label.setText("Load failed: Invalid format.")
                    # Update button states
                    self.btn_load.setEnabled(True)
                    self.btn_compress.setEnabled(False)
                    self.btn_decompress.setEnabled(False)
                    self.btn_save_compressed.setEnabled(False)


            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load compressed data: {e}")
                self.encoded_data = None # Ensure invalid data is not kept
                self.label.setText("Error loading compressed data.")
                # if hasattr(self, 'stats_label'): self.stats_label.setText("Load failed.")
                # Update button states
                self.btn_load.setEnabled(True)
                self.btn_compress.setEnabled(False)
                self.btn_decompress.setEnabled(False)
                self.btn_save_compressed.setEnabled(False)


    def show_image(self, img_array):
        """Displays a NumPy image array in the QLabel."""
        if img_array is None:
            self.label.setPixmap(QPixmap()) # Clear display
            self.label.setText("No image loaded")
            return

        height, width = img_array.shape[:2]
        if img_array.ndim == 2:  # Grayscale (L)
            # QImage.Format_Grayscale8 expects data to be contiguous (C-order)
            # bytesPerLine is width for grayscale
            bytes_per_line = width
            # Make sure data is contiguous and correctly typed
            # Use .copy() to ensure contiguous if original array wasn't
            q_image = QImage(img_array.astype(np.uint8).copy().data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:  # Color (RGB)
            # QImage.Format_RGB888 expects data to be contiguous (C-order)
            # bytesPerLine is 3 * width for RGB
            bytes_per_line = 3 * width
             # Make sure data is contiguous, correctly typed, and in the right order
            # Use .copy() to ensure contiguous if original array wasn't
            rgb_array = img_array.astype(np.uint8).copy()
            q_image = QImage(rgb_array.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Check if the QImage was created successfully
        if q_image.isNull():
            print("Failed to create QImage.") # For debugging
            self.label.setPixmap(QPixmap())
            self.label.setText("Error displaying image")
            return

        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(q_image)

        # Scale the pixmap to fit the label while maintaining aspect ratio
        # self.label.size() gives the current size of the label widget
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


# Standard boilerplate to run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = JPEGCompressorApp()
    window.show()
    sys.exit(app.exec_())