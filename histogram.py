import numpy as np
from PyQt5.QtCore import Qt
# from PyQt5 import Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from matplotlib import pyplot as plt


def calculate_histogram(image):
    # Calculate histogram
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    return hist

def histogram_to_qimage(hist):
    # Calculate maximum histogram value
    max_hist_value = max(hist)

    # Determine image dimensions
    num_bins = len(hist)
    image_width = num_bins
    image_height = max_hist_value + 50  # Add space for axes labels

    # Create a blank QImage with a white background
    q_image = QImage(image_width, image_height, QImage.Format_RGB32)
    q_image.fill(Qt.white)

    # Create a QPainter to draw on the QImage
    painter = QPainter()
    painter.begin(q_image)

    # Set pen color to black for drawing histogram bars and axes
    painter.setPen(Qt.black)

    # Draw histogram bars
    for i, value in enumerate(hist):
        bar_height = value  # Use histogram value directly for height

        # Draw a vertical line for each bin in the histogram
        painter.drawLine(i, image_height - 50, i, image_height - 50 - bar_height)

    # Draw x-axis (intensity level)
    painter.drawLine(0, image_height - 50, image_width - 1, image_height - 50)

    # Draw y-axis (frequency)
    painter.drawLine(0, 0, 0, image_height - 50)

    # Add labels to axes
    font = QFont("Arial", 10)
    painter.setFont(font)
    painter.drawText(image_width - 20, image_height - 20, "Intensity Level")
    painter.drawText(10, 10, "Frequency")

    # End painting
    painter.end()

    return q_image

def display_qimage(q_image, widget):
    # Convert QImage to QPixmap
    pixmap = QPixmap.fromImage(q_image)

    # Create a QPainter to draw on the widget
    painter = QPainter(widget)
    painter.drawPixmap(widget.rect(), pixmap)
    painter.end()