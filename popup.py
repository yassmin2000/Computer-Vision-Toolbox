from PyQt5.QtWidgets import QDialog, QVBoxLayout,QMessageBox
from custom import HistogramWidget
from PyQt5.QtGui import QColor
import numpy as np
from histogram import calculate_histogram


class RGBHistogramWindow(QDialog):
    def __init__(self, original_img):
        super().__init__()
        self.setWindowTitle("RGB Histogram")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Resize the window to a bigger size
        self.resize(800, 600)
    
        # Check if the number of channels is not 2
        if original_img.shape[-1] == 3:

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
        else:
            self.plot_distribution_curves
            # QMessageBox.warning(self, "Warning", "Image has only 2 channels. RGB histogram cannot be calculated.")

    def calculate_rgb_histogram(self, original_img):
        # Calculate histogram for each channel (R, G, B)
        r_hist = calculate_histogram(original_img[:, :, 0])
        g_hist = calculate_histogram(original_img[:, :, 1])
        b_hist = calculate_histogram(original_img[:, :, 2])

        return r_hist, g_hist, b_hist
    
    def plot_distribution_curves(self):
        ax = self.figure.add_subplot(111)

        # Calculate histograms for gray image and processed image
        hist_gray, bins_gray = np.histogram(self.gray_img.flatten(), bins=256, range=[0,256])
        hist_processed, bins_processed = np.histogram(self.processed_img.flatten(), bins=256, range=[0,256])

        # Calculate cumulative histograms
        cdf_gray = hist_gray.cumsum()
        cdf_processed = hist_processed.cumsum()

        # Normalize cumulative histograms
        cdf_normalized_gray = cdf_gray / cdf_gray.max()
        cdf_normalized_processed = cdf_processed / cdf_processed.max()

        # Plot distribution curves and cumulative curves
        ax.plot(bins_gray[:-1], hist_gray, color='r', label='Gray Image Distribution Curve')
        ax.plot(bins_processed[:-1], hist_processed, color='g', label='Processed Image Distribution Curve')
        ax.plot(bins_gray[:-1], cdf_normalized_gray, color='b', label='Gray Image Cumulative Curve')
        ax.plot(bins_processed[:-1], cdf_normalized_processed, color='y', label='Processed Image Cumulative Curve')

        ax.legend()
        self.canvas.draw()


    def show_histogram_window(self, original_img):
        self.exec_()
