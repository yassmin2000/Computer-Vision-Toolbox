from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem, QFileDialog, QGraphicsScene,QVBoxLayout
from ui_handler import handlecomboBoxChange, handlecomboBoxChange2, handelcomboxchanges3
from noise import apply_uniform_noise, apply_gaussian_noise, apply_salt_and_pepper_noise
from filters import apply_average_filter, apply_gaussian_filter, apply_median_filter
from edge_detection import (
    sobel_operator,
    roberts_operator,
    prewitt_operator,
    canny_operator,
)
from thresholding import global_thresholding, local_thresholding
from histogram import calculate_histogram, histogram_to_qimage, display_qimage
from custom import HistogramWidget  # Import HistogramWidget class
from popup import RGBHistogramWindow
import lh_filters_hybrid
import equalization_and_normalization
import numpy as np
import sys
import cv2


class mainwindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        uic.loadUi("mainwindow.ui", self)
        self.tabWidget.setCurrentIndex(0)
        self.original_img = []
        self.gray_img = np.array([], dtype=np.uint8)
        self.processed_img = []
        self.hybrid_input = []
        self.hybrid_input_1 = []
        self.hybrid_input_2 = []
        self.transformed_img = []

        self.filters_combo.currentIndexChanged.connect(self.handlecomboBoxChange)
        self.comboBox_2.currentIndexChanged.connect(self.handlecomboBoxChange2)
        self.types.currentIndexChanged.connect(self.handelcomboxchanges3)
        self.pushButton.clicked.connect(self.apply_button_clicked)
        self.pushButton_3.clicked.connect(self.clear)
        self.pushButton_4.clicked.connect(self.clear)
        self.pushButton_5.clicked.connect(self.show_histogram)
        self.graphicsView.mouseDoubleClickEvent = lambda event: self.browse_image(
            self.graphicsView
        )
        self.graphicsView_3.mouseDoubleClickEvent = lambda event: self.browse_image(
            self.graphicsView_3
        )
        self.graphicsView_5.mouseDoubleClickEvent = lambda event: self.browse_image(
            self.graphicsView_5
        )
        self.graphicsView_10.mouseDoubleClickEvent = lambda event: self.browse_image(
            self.graphicsView_10
        )
        self.radioButton.toggled.connect(self.handle_radio_buttons)
        self.radioButton_2.toggled.connect(self.handle_radio_buttons)
        self.horizontalSlider.valueChanged.connect(self.thresholding)
        self.horizontalSlider_2.valueChanged.connect(self.thresholding)
        self.horizontalSlider_3.valueChanged.connect(self.thresholding)
        self.horizontalSlider_4.valueChanged.connect(self.thresholding)
        self.label_3.setVisible(True)
        self.comboBox_2.setVisible(True)
        self.label_14.setVisible(False)
        self.horizontalSlider_5.setVisible(False)
        self.label_24.setVisible(False)
        self.horizontalSlider_6.setVisible(False)
        self.label_25.setVisible(False)
        self.horizontalSlider_7.setVisible(False)
        self.label_26.setVisible(False)
        self.horizontalSlider_8.setVisible(False)
        self.label_27.setVisible(False)
        self.horizontalSlider_9.setVisible(False)
        self.label_28.setVisible(False)
        self.label_29.setVisible(False)
        self.label_30.setVisible(False)
        self.label_31.setVisible(False)
        self.label_32.setVisible(False)
        self.label_33.setVisible(False)
        self.label_34.setVisible(False)
        self.label_35.setVisible(False)
        self.label_36.setVisible(False)
        self.label_37.setVisible(False)
        self.label_38.setVisible(False)
        self.label_39.setVisible(False)
        self.label_40.setVisible(False)

        self.comboBox_2.clear()
        new_options = ["Uniform", "Gaussian", "Salt & pepper"]
        self.comboBox_2.addItems(new_options)
        self.radioButton.setChecked(True)
        self.pushButton_2.clicked.connect(self.hybrid_image)

    def handlecomboBoxChange(self):
        handlecomboBoxChange(self)

    def handlecomboBoxChange2(self):
        handlecomboBoxChange2(self)

    def handelcomboxchanges3(self):
        handelcomboxchanges3(self)

    def handle_radio_buttons(self):
        # Check if graphicsView has contents
        if self.graphicsView.scene() and self.graphicsView.scene().items():
            if self.radioButton.isChecked():
                # Display original image
                self.display_image(self.original_img, self.graphicsView)
            elif self.radioButton_2.isChecked():
                # Display gray image
                self.display_image(self.gray_img, self.graphicsView)

    def check_tap(self):
        if self.tabWidget.currentIndex() == 2:
            return 1
        else:
            return 0

    def hybrid_image(self):
        img_out = lh_filters_hybrid.hybrid_image(
            self.hybrid_input_1,
            self.hybrid_input_2,
            self.lineEdit_5.text(),
            self.lineEdit_9.text(),
            self.comboBox_3.currentIndex(),
            self.comboBox_4.currentIndex(),
        )
        self.display_image(img_out, self.graphicsView_4)

    def browse_image(self, widget):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.gif *.jpeg)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_():
            selected_file = file_dialog.selectedFiles()[0]
            # Create an instance of the Image class with the selected image path
            tap_index = self.check_tap()
            if tap_index:
                if widget == self.graphicsView_3:
                    self.hybrid_input_1 = cv2.imread(selected_file)
                    # self.hybrid_input_1 = cv2.cvtColor(self.hybrid_input_1, cv2.COLOR_BGR2RGB)
                    self.display_image(self.hybrid_input_1, widget)
                elif widget == self.graphicsView_5:
                    self.hybrid_input_2 = cv2.imread(selected_file)
                    # self.hybrid_input_2 = cv2.cvtColor(self.hybrid_input_2, cv2.COLOR_BGR2RGB)
                    self.display_image(self.hybrid_input_2, widget)
            else:
                self.original_img = cv2.imread(selected_file)
                self.gray_img = self.convert_gry(self.original_img)
                self.processed_img = self.gray_img
                if widget == self.graphicsView:
                    if self.radioButton_2.isChecked():
                        self.radioButton.setChecked(False)

                        self.display_image(self.gray_img, widget)
                        self.display_image(self.gray_img, self.graphicsView_6)
                        self.histogram_plot(self.gray_img, self.widget)
                    else:
                        self.radioButton_2.setChecked(False)
                        self.display_image(self.original_img, widget)
                        self.display_image(self.gray_img, self.graphicsView_6)
                        self.histogram_plot(self.gray_img, self.widget)
                elif widget == self.graphicsView_10:
                    self.transformed_img = cv2.imread(selected_file)
                    self.transformed_img = self.convert_gry(self.transformed_img)
                    self.display_image(self.transformed_img, widget)

    def display_image(self, img_data, widget):
        # Get the size of the widget
        widget_width = widget.width()
        widget_height = widget.height()

        # Check if the image is grayscale or color
        if len(img_data.shape) == 2:  # Grayscale image
            img_height, img_width = img_data.shape
            bytes_per_line = img_width
            q_image = QImage(
                bytes(img_data.data),
                img_width,
                img_height,
                bytes_per_line,
                QImage.Format_Grayscale8,
            )
        elif len(img_data.shape) == 3:  # Color image
            img_height, img_width, channels = img_data.shape
            bytes_per_line = 3 * img_width
            img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

            q_image = QImage(
                img_rgb.data,
                img_width,
                img_height,
                bytes_per_line,
                QImage.Format_RGB888,
            )
        else:
            raise ValueError("Unsupported image format.")

        # Calculate the new width and height to maintain the aspect ratio
        aspect_ratio = img_width / img_height
        if aspect_ratio > 1:
            new_width = widget_width
            new_height = int(widget_width / aspect_ratio)
        else:
            new_height = widget_height
            new_width = int(widget_height * aspect_ratio)
        # Create a QGraphicsScene
        scene = QGraphicsScene(self)
        widget.setScene(scene)
        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(q_image)
        # Create a QGraphicsPixmapItem and add it to the scene
        pixmap_item = QGraphicsPixmapItem(pixmap)
        scene.addItem(pixmap_item)

    def convert_gry(self, img_data):
        if len(img_data.shape) == 3:  # Check if it's a color image
            gray_image = np.zeros(
                (img_data.shape[0], img_data.shape[1]), dtype=np.uint8
            )
            for i in range(img_data.shape[0]):
                for j in range(img_data.shape[1]):
                    # Get individual pixel values
                    R, G, B = img_data[i, j, 2], img_data[i, j, 1], img_data[i, j, 0]
                    # Calculate grayscale using the luminosity method
                    gray_image[i, j] = int(R * 0.299 + G * 0.587 + B * 0.114)
            # Ensure that the values are normalized to the range [0, 255]
            gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)
            self.gray_img = gray_image
            return gray_image

    def resize_image(self, target_width, target_height):
        # Resize the image to the specified width and height
        resized_image = cv2.resize(self.image_data, (target_width, target_height))

        # Convert the resized image from BGR to RGB
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        height, width, channel = resized_image_rgb.shape  # Get image shape
        bytes_per_line = 3 * width  # RGB image has 3 channels

        # Update the QImage with the resized RGB image data
        self.dimage = QtGui.QImage(
            resized_image_rgb.data,
            width,
            height,
            bytes_per_line,
            QtGui.QImage.Format_RGB888,
        )

    def convert_array_to_image(self, img_array, original_img_dtype):
        # Convert the array to an image format compatible with PyQt
        img_rgb = cv2.cvtColor(
            img_array, cv2.COLOR_BGR2RGB
        )  # Assuming img_array is in BGR format

        # Get the original image dimensions
        img_height, img_width, img_channels = img_rgb.shape

        # Convert the data type of the array to match the original image
        img_rgb = img_rgb.astype(original_img_dtype)

        # Convert the image to QImage
        q_image = QImage(
            img_rgb.data,
            img_width,
            img_height,
            img_rgb.strides[0],
            QImage.Format_RGB888,
        )

        return q_image

    def edge_detection(
        self, selected_method, selected_dir, low_threshold, high_threshold
    ):
        if selected_method == "Sobel":
            edges = sobel_operator(self.original_img, selected_dir)
        elif selected_method == "Roberts":
            edges = roberts_operator(self.original_img, selected_dir)
        elif selected_method == "Prewitt":
            edges = prewitt_operator(self.original_img, selected_dir)
        elif selected_method == "Canny":
            edges = canny_operator(self.original_img, low_threshold, high_threshold)

        self.display_image(edges, self.graphicsView_2)

    def thresholding(self):
        if self.comboBox_2.currentIndex() == 0:
            thresholded_image_path = global_thresholding(
                self.original_img, self.horizontalSlider.value()
            )
        else:
            t1 = self.horizontalSlider.value()
            t2 = self.horizontalSlider_2.value()
            t3 = self.horizontalSlider_3.value()
            t4 = self.horizontalSlider_4.value()
            thresholded_image_path = local_thresholding(
                self.original_img, t1, t2, t3, t4
            )

        self.display_image(thresholded_image_path, self.graphicsView_2)

    def apply_button_clicked(self):
        if self.filters_combo.currentIndex() == 0:
            if self.comboBox_2.currentIndex() == 0:
                noise_value = int(self.lineEdit.text())
                result = apply_uniform_noise(self.gray_img, noise_value)
                self.processed_img = apply_uniform_noise(
                    self.processed_img, noise_value
                )
            elif self.comboBox_2.currentIndex() == 1:
                mean = int(self.lineEdit.text())
                sigma = int(self.lineEdit_2.text())
                result = apply_gaussian_noise(self.gray_img, mean, sigma)
                self.processed_img = apply_gaussian_noise(
                    self.processed_img, mean, sigma
                )
            else:
                w = float(self.lineEdit_2.text())
                b = float(self.lineEdit.text())
                result = apply_salt_and_pepper_noise(self.gray_img, w, b)
                self.processed_img = apply_salt_and_pepper_noise(
                    self.processed_img, w, b
                )
            self.display_image(result, self.graphicsView_2)
            self.display_image(result, self.graphicsView_7)
            self.histogram_plot(result, self.widget_2)

        elif self.filters_combo.currentIndex() == 1:
            if self.comboBox_2.currentIndex() == 0:
                kernel_size = int(self.lineEdit.text())
                result = apply_average_filter(self.gray_img, kernel_size)
                self.processed_img = apply_average_filter(
                    self.processed_img, kernel_size
                )
            elif self.comboBox_2.currentIndex() == 1:
                sigma = int(self.lineEdit.text())
                result = apply_gaussian_filter(self.gray_img, sigma)
                self.processed_img = apply_gaussian_filter(self.processed_img, sigma)
            elif self.comboBox_2.currentIndex() == 2:
                result = apply_median_filter(self.gray_img)
                self.processed_img = apply_median_filter(self.processed_img)
            self.display_image(self.processed_img, self.graphicsView_2)
            self.display_image(self.processed_img, self.graphicsView_7)
            self.histogram_plot(self.processed_img, self.widget_2)

        elif self.filters_combo.currentIndex() == 2:
            selected_index = self.comboBox_2.currentIndex()
            selected_method = self.comboBox_2.itemText(selected_index)
            selected_index_2 = self.comboBox.currentIndex()
            selected_dir = self.comboBox.itemText(selected_index_2)
            if selected_method == "Canny":
                low_threshold = int(self.lineEdit.text())
                high_threshold = int(self.lineEdit_2.text())
                self.edge_detection(
                    selected_method, selected_dir, low_threshold, high_threshold
                )
            else:
                self.edge_detection(selected_method, selected_dir, 0, 0)
        elif self.filters_combo.currentIndex() == 4:
            equalized_image_array = equalization_and_normalization.equalization(
                self.gray_img
            )
            # q_image = self.convert_array_to_image(equalized_image_array, self.gray_img.dtype)
            self.display_image(equalized_image_array, self.graphicsView_2)
            self.display_image(equalized_image_array, self.graphicsView_7)
            self.histogram_plot(equalized_image_array, self.widget_2)
        elif self.filters_combo.currentIndex() == 5:
            normalized_image_array = equalization_and_normalization.normalization(
                self.gray_img
            )
            # q_image = self.convert_array_to_image(normalized_image_array, self.gray_img.dtype)
            self.display_image(normalized_image_array, self.graphicsView_2)
            self.display_image(normalized_image_array, self.graphicsView_7)
            self.histogram_plot(normalized_image_array, self.widget_2)
        elif (
            self.filters_combo.currentIndex() == 6
            or self.filters_combo.currentIndex() == 7
        ):
            img = lh_filters_hybrid.fourier_transform(self.original_img)
            lowpass_filter, highpass_filter = lh_filters_hybrid.low_high_filters(
                float(self.lineEdit.text()), img
            )
            if self.filters_combo.currentIndex() == 6:
                filtered_img = img * lowpass_filter
            elif self.filters_combo.currentIndex() == 7:
                filtered_img = img * highpass_filter
            output = np.fft.ifftshift(filtered_img)
            output_img = np.abs(np.fft.ifft2(output))
            output_img = output_img.astype(np.uint8)
            self.display_image(output_img, self.graphicsView_2)
            self.display_image(output_img, self.graphicsView_7)
            self.histogram_plot(output_img, self.widget_2)

    def histogram_plot(self, image, widget):
        histogram_array = calculate_histogram(image)
        histogram_widget = HistogramWidget(histogram_array)  # Create instance of HistogramWidget
        if widget.layout() is None:  # Check if widget has a layout
            layout = QVBoxLayout()  # Create a vertical layout
            widget.setLayout(layout)  # Set the layout to the widget
        else:
            layout = widget.layout()  # Get the existing layout
        layout.addWidget(histogram_widget)  # Add HistogramWidget to the layout
    def show_histogram(self):
        rgb_histogram_window = RGBHistogramWindow(self.original_img)
        rgb_histogram_window.show_histogram_window(self.original_img)
    def clear(self):
        tap_index = self.check_tap()
        if tap_index:
            self.hybrid_input = []
            self.hybrid_input_1 = []
            self.hybrid_input_2 = []
            self.graphicsView_3.scene().clear()
            self.graphicsView_4.scene().clear()
            self.graphicsView_5.scene().clear()
        else:
            self.original_img = []
            self.gray_img = np.array([], dtype=np.uint8)
            self.processed_img = []
            self.graphicsView.scene().clear()
            self.graphicsView_2.scene().clear()
            self.graphicsView_6.scene().clear()
            self.graphicsView_7.scene().clear()
            self.graphicsView_8.scene().clear()
            self.graphicsView_9.scene().clear()


if __name__ == "__main__":
    Main_App = QtWidgets.QApplication(sys.argv)
    App = mainwindow()
    App.show()
    sys.exit(Main_App.exec_())
