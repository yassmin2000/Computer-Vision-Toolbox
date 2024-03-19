from PyQt5.QtWidgets import QDialog, QVBoxLayout
from custom import HistogramWidget
from PyQt5.QtGui import QColor
from histogram import calculate_histogram


class RGBHistogramWindow(QDialog):
    def __init__(self, original_img):
        super().__init__()
        self.setWindowTitle("RGB Histogram")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Resize the window to a bigger size
        self.resize(800, 600)

        # Calculate RGB histogram of the original image
        r_hist, g_hist, b_hist = self.calculate_rgb_histogram(original_img)

        # Create HistogramWidget instances for each channel with colors
        r_histogram_widget = HistogramWidget(r_hist, color=QColor("red"))
        g_histogram_widget = HistogramWidget(g_hist, color=QColor("green"))
        b_histogram_widget = HistogramWidget(b_hist, color=QColor("blue"))
        
        # Add HistogramWidget instances to the layout
        self.layout.addWidget(r_histogram_widget)
        self.layout.addWidget(g_histogram_widget)
        self.layout.addWidget(b_histogram_widget)


    def calculate_rgb_histogram(self, original_img):
        # Calculate histogram for each channel (R, G, B)
        r_hist = calculate_histogram(original_img[:, :, 0])
        g_hist = calculate_histogram(original_img[:, :, 1])
        b_hist = calculate_histogram(original_img[:, :, 2])

        return r_hist, g_hist, b_hist


    def show_histogram_window(self, original_img):
        self.exec_()
