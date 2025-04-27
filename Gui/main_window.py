"""
main_window.py
PyQt5 GUI for JPEG Compression/Decompression
"""

import sys
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
from PIL import Image

# Import your coder and decoder
from Coder.encoder import encode_image
from Decoder.decoder import decode_image


class JPEGCompressorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JPEG Compressor/Decompressor")
        self.setGeometry(100, 100, 600, 400)

        # Layout
        layout = QVBoxLayout()

        # Labels
        self.label = QLabel("No image loaded")
        self.label.setAlignment(Qt.AlignCenter)

        # Buttons
        self.btn_load = QPushButton("Upload Image")
        self.btn_compress = QPushButton("Compress")
        self.btn_decompress = QPushButton("Decompress")

        self.btn_compress.setEnabled(False)
        self.btn_decompress.setEnabled(False)

        # Add widgets to layout
        layout.addWidget(self.label)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_compress)
        layout.addWidget(self.btn_decompress)

        self.setLayout(layout)

        # Connect buttons to actions
        self.btn_load.clicked.connect(self.load_image)
        self.btn_compress.clicked.connect(self.compress_image)
        self.btn_decompress.clicked.connect(self.decompress_image)

        # Internal variables
        self.original_image = None
        self.encoded_data = None

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
            options=options
        )
        if file_name:
            self.original_image = np.array(Image.open(file_name))
            self.show_image(self.original_image)
            self.btn_compress.setEnabled(True)
            self.btn_decompress.setEnabled(False)
            self.encoded_data = None

    def compress_image(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Please upload an image first.")
            return

        self.encoded_data = encode_image(self.original_image)
        QMessageBox.information(self, "Success", "Image compressed successfully!")
        self.btn_decompress.setEnabled(True)

    def decompress_image(self):
        if self.encoded_data is None:
            QMessageBox.warning(self, "Warning", "Please compress an image first.")
            return

        decoded_image = decode_image(self.encoded_data)
        self.show_image(decoded_image)

    def show_image(self, img_array):
        if img_array.ndim == 2:  # Grayscale
            height, width = img_array.shape
            bytes_per_line = width
            q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:  # Color
            height, width, channel = img_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap.scaled(self.label.width(), self.label.height(), aspectRatioMode=True))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = JPEGCompressorApp()
    window.show()
    sys.exit(app.exec_())
