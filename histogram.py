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
    # Create a blank QImage with a white background
    q_image = QImage(4096, 250, QImage.Format_Grayscale8)
    q_image.fill(255)

    image_width = 4096
    # image_width = 512
    image_height = 200  # Adjust this value to leave space below the histogram
    max_hist_value = max(hist)

    num_bins = len(hist)
    total_space = image_width - num_bins  # Total available space for bars and gaps
    gap_size = 1  # Size of gap between bars
    bar_width = total_space // num_bins  # Width of each bar

    # Draw histogram bars
    for i, value in enumerate(hist):
        bar_height = int(value * (image_height - 1) / max_hist_value)  # Adjust height calculation

        # Calculate the start position of each bar
        start_x = i * (bar_width + gap_size)

        for x in range(start_x, start_x + bar_width):
            for y in range(image_height - bar_height, image_height):
                q_image.setPixel(x, y, 0)

        # Draw label on the x-axis
        label_x = start_x + bar_width // 2 - 10  # Adjust for alignment
        label_y = image_height + 15  # Position below the x-axis
        label_value = str(i)  # Display the index of the bin
        font = QFont("Arial", 10)  # Increase font size
        painter = QPainter(q_image)
        painter.setFont(font)
        painter.drawText(label_x, label_y, label_value)

    return q_image


def display_qimage(q_image, widget, scene):
    # Get the size of the widget
    widget_width = widget.width()
    widget_height = widget.height()

    # Calculate the aspect ratio of the image
    aspect_ratio = q_image.width() / q_image.height()

    # Calculate the new width and height to maintain the aspect ratio
    if aspect_ratio > 1:
        new_width = widget_width
        new_height = int(widget_width / aspect_ratio)
    else:
        new_height = widget_height
        new_width = int(widget_height * aspect_ratio)

    # Scale the image to the new dimensions
    # scaled_image = q_image.scaled(new_width, new_height, aspectMode=Qt.KeepAspectRatio)

    # Create a QGraphicsScene
    # scene = QGraphicsScene()
    widget.setScene(scene)

    # Create a QPixmap from the scaled QImage
    pixmap = QPixmap.fromImage(q_image)

    # Create a QGraphicsPixmapItem and add it to the scene
    pixmap_item = QGraphicsPixmapItem(pixmap)
    scene.addItem(pixmap_item)

