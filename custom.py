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
        painter = QPainter(self)
        painter.setPen(self.color)  # Set the color for drawing

        # Draw histogram bars based on histogram data
        bar_width = self.width() / len(self.histogram_data)
        bar_height_factor = self.height() / max(self.histogram_data)

        for i, value in enumerate(self.histogram_data):
            x = int(i * bar_width)
            bar_height = int(value * bar_height_factor)
            painter.drawLine(x, self.height(), x, self.height() - bar_height)


    def draw_histogram(self, painter, hist, widget_width, widget_height):
        max_hist_value = max(hist)
        num_bins = len(hist)
        total_space = widget_width - num_bins  # Total available space for bars and gaps
        gap_size = 1  # Size of gap between bars
        bar_width = total_space // num_bins  # Width of each bar

        # Set pen color and width
        painter.setPen(QColor(0, 0, 0))
        painter.setBrush(QColor(0, 0, 0, 100))

        # Draw histogram bars
        for i, value in enumerate(hist):
            bar_height = int(value * (widget_height - 1) / max_hist_value)  # Adjust height calculation

            # Calculate the start position of each bar
            start_x = i * (bar_width + gap_size)

            # Draw rectangle for each bar
            painter.drawRect(int(start_x), int(widget_height - bar_height), int(bar_width), int(bar_height))

        # Draw x-axis label
        x_label = "Intensity Level"
        x_label_font = QFont("Arial", 10)
        painter.setFont(x_label_font)
        x_label_width = painter.fontMetrics().width(x_label)
        x_label_x = (widget_width - x_label_width) / 2
        x_label_y = widget_height + 30  # Position below the x-axis
        painter.drawText(int(x_label_x), int(x_label_y), x_label)

        # Draw y-axis label
        y_label = "Frequency"
        y_label_font = QFont("Arial", 10)
        painter.setFont(y_label_font)
        y_label_width = painter.fontMetrics().width(y_label)
        y_label_height = painter.fontMetrics().height()
        y_label_x = 10
        y_label_y = (widget_height + y_label_height) / 2
        painter.rotate(-90)  # Rotate for vertical text
        painter.drawText(-int(y_label_y), int(y_label_x), y_label)
        painter.rotate(90)  # Rotate back to normal

        # Draw x-axis ticks
        num_ticks = 10  # Number of ticks on x-axis
        tick_spacing = num_bins // num_ticks
        for i in range(0, num_bins, tick_spacing):
            tick_label = str(i)
            tick_label_width = painter.fontMetrics().width(tick_label)
            tick_label_x = i * (bar_width + gap_size) + (bar_width - tick_label_width) / 2
            tick_label_y = widget_height + 15
            painter.drawText(int(tick_label_x), int(tick_label_y), tick_label)

        # Draw y-axis ticks
        max_tick_value = int(max_hist_value)
        tick_spacing_y = max_tick_value // num_ticks
        for i in range(0, max_tick_value + 1, tick_spacing_y):
            tick_label = str(i)
            tick_label_width = painter.fontMetrics().width(tick_label)
            tick_label_x = -tick_label_width - 5
            tick_label_y = widget_height - i * (widget_height / max_tick_value)
            painter.drawText(int(tick_label_x), int(tick_label_y), tick_label)

