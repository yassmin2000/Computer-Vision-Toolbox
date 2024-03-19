import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class CurvesWindow(QDialog):
    def __init__(self, gray_image, processed_image, parent=None):
        super().__init__(parent)
        self.gray_image = gray_image
        self.processed_image = processed_image
        self.setWindowTitle("Distribution Curves")

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.figure = plt.figure(figsize=(10, 6))

        self.plot_curves()

        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def plot_curves(self):
        ax1 = self.figure.add_subplot(221)
        self.plot_distribution_curve(self.gray_image.flatten(), ax1, 'b', 'Gray Image Distribution')

        ax2 = self.figure.add_subplot(222)
        self.plot_cumulative_curve(self.gray_image.flatten(), ax2, 'b', 'Gray Image Cumulative Histogram')

        ax3 = self.figure.add_subplot(223)
        self.plot_distribution_curve(self.processed_image.flatten(), ax3, 'r', 'Processed Image Distribution')

        ax4 = self.figure.add_subplot(224)
        self.plot_cumulative_curve(self.processed_image.flatten(), ax4, 'r', 'Processed Image Cumulative Histogram')

        self.figure.tight_layout()

    def plot_distribution_curve(self, image_data, ax, color, title):
        intensity_counts, _ = np.histogram(image_data, bins=256, density=True)
        intensities = np.arange(256)
        ax.plot(intensities, intensity_counts, color=color, alpha=0.5)
        ax.set_title(title)

    def plot_cumulative_curve(self, image_data, ax, color, title):
        intensity_counts, _ = np.histogram(image_data, bins=256, density=True)
        cumulative_counts = np.cumsum(intensity_counts)
        intensities = np.arange(256)
        ax.plot(intensities, cumulative_counts, color=color, alpha=0.5)
        ax.set_title(title)


