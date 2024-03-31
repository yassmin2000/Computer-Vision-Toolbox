# histogram_widget.py
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPainter,QColor
from PyQt5.QtWidgets import QWidget


class HistogramWidget(QWidget):
    def __init__(self, histogram_data, color=QColor("black")):
        super().__init__()
        self.histogram_data = histogram_data
        self.color = color
        self.hist=histogram_data

    def paintEvent(self, event):
        """
    Handles the painting of the widget's content.
    
    Parameters:
    - event: the paint event that triggered the painting
    """
        painter = QPainter(self)
        painter.setPen(self.color)  # Set the color for drawing

        margin_left = 35
        # bar_width = self.width() / len(self.histogram_data)
        bar_width =(self.width() - margin_left) / len(self.histogram_data)
        margin_height = 25

        max_hist_value = max(self.histogram_data)
        max_widget_height = self.height() - margin_height  # Adjust for margin
        bar_height_factor = max_widget_height / max_hist_value

        for i, value in enumerate(self.histogram_data):
            x = int(i * bar_width + margin_left)
            bar_height = int(value * bar_height_factor)

            # Ensure that the bars stay within the visible boundaries of the widget
            bar_height = min(bar_height, max_widget_height)

            painter.drawLine(x, max_widget_height, x, max_widget_height - bar_height)

        self.draw_histogram(painter, self.hist, self.width(), self.height())

    def draw_histogram(self, painter, hist, widget_width, widget_height):
        """
    Draws a histogram on the widget using the provided QPainter and histogram data.

    Parameters:
    - painter: the QPainter object used for drawing
    - hist: the histogram data to be visualized
    - widget_width: the width of the widget
    - widget_height: the height of the widget
    """

        max_hist_value = max(hist)
        num_bins = len(hist)
        total_space = widget_width - num_bins  # Total available space for bars and gaps
        gap_size = 1  # Size of gap between bars
        bar_width = total_space // num_bins  # Width of each bar

        # Set pen color and width
        painter.setPen(QColor(0, 0, 0))
        painter.setBrush(QColor(0, 0, 0, 100))

        margin_height = 25
        margin_left = 35

        # Draw x-axis label
        x_label = "Intensity Level"
        x_label_font = QFont("Arial", 8)
        painter.setFont(x_label_font)
        x_label_width = painter.fontMetrics().width(x_label)
        x_label_x = max(0, min((widget_width - x_label_width) / 2, widget_width - x_label_width))  # Adjust x_label_x
        x_label_y = min(widget_height + 30, widget_height)  # Adjust x_label_y

        painter.drawText(int(x_label_x), int(x_label_y), x_label)

        # Draw y-axis label
        y_label = "Frequency"
        y_label_font = QFont("Arial", 8)
        painter.setFont(y_label_font)
        y_label_width = painter.fontMetrics().width(y_label)
        y_label_height = painter.fontMetrics().height()

        y_label_x = min(10, widget_width - y_label_width)  # Adjust y_label_x
        y_label_y = min((widget_height + y_label_height) / 2, widget_height)  # Adjust y_label_y

        painter.rotate(-90)  # Rotate for vertical text
        painter.drawText(-int(y_label_y), int(y_label_x), y_label)
        painter.rotate(90)  # Rotate back to normal

        # draw x-axis
        painter.drawLine(margin_left, widget_height-margin_height, self.width(), widget_height-margin_height)

        # draw_y_axis
        painter.drawLine(margin_left, widget_height - margin_height, margin_left, 0)

        # Draw x-axis ticks
        tick_label_font = QFont("Arial", 6)  # Adjust font size as needed
        painter.setFont(tick_label_font)
        num_ticks = 10  # Number of ticks on x-axis
        tick_spacing = num_bins // num_ticks
        for i in range(0, num_bins, tick_spacing):
            tick_label = str(i)
            tick_label_width = painter.fontMetrics().width(tick_label)
            tick_label_x = margin_left + i * (bar_width + gap_size) + (bar_width - tick_label_width) / 2
            tick_label_y = widget_height - 10

            painter.drawText(int(tick_label_x), int(tick_label_y), tick_label)

        # Draw y-axis ticks
        tick_label_font = QFont("Arial", 6)  # Adjust font size as needed
        painter.setFont(tick_label_font)

        max_tick_value = int(max_hist_value)
        tick_spacing_y = max_tick_value // num_ticks
        for i in range(0, max_tick_value + 1, tick_spacing_y):
            tick_label = str(i)
            tick_label_width = painter.fontMetrics().width(tick_label)
            tick_label_x = margin_left - tick_label_width - 5
            tick_label_y = widget_height - margin_height - i * ((widget_height - margin_height) / max_tick_value)
            painter.drawText(int(tick_label_x), int(tick_label_y), tick_label)

