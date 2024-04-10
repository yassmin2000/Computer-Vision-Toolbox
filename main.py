from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QGraphicsPixmapItem,
    QFileDialog,
    QGraphicsScene,
    QVBoxLayout,
    QMessageBox,
)
from ui_handler import handlecomboBoxChange, handlecomboBoxChange2, handelcomboxchanges3 , handlematchingcombo
from noise import apply_uniform_noise, apply_gaussian_noise, apply_salt_and_pepper_noise
from filters import apply_average_filter, apply_gaussian_filter, apply_median_filter
from skimage.filters import gaussian
from edge_detection import (
    sobel_operator,
    roberts_operator,
    prewitt_operator,
    canny_operator,
)
from thresholding import global_thresholding, local_thresholding
from histogram import calculate_histogram
from custom import HistogramWidget  # Import HistogramWidget class
from popup import RGBHistogramWindow
from popup2 import CurvesWindow
import lh_filters_hybrid
import equalization_and_normalization
from active_contour import active_contour
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
from circle import hough_circles
from lines import hough_line_detection
from elipse import edge_ellipse_detector


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
        self.matching_img = []
        self.template_img = []
        self.segmentation_img = []
        self.seg_img_height = 0
        self.seg_img_width = 0
        self.center_x = 0
        self.center_y = 0
        self.radius = 0
        self.s = np.linspace(0, 2 * np.pi, 400)
        self.r = np.zeros_like(self.s)
        self.c = np.zeros_like(self.s)
        self.init_coords = np.array([self.r, self.c]).T
        self.canvas = FigureCanvas(plt.figure())

        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.filters_combo.currentIndexChanged.connect(self.handlecomboBoxChange)
        self.comboBox_2.currentIndexChanged.connect(self.handlecomboBoxChange2)
        self.types.currentIndexChanged.connect(self.handelcomboxchanges3)
        self.pushButton.clicked.connect(self.apply_button_clicked)
        self.pushButton_3.clicked.connect(self.clear)
        self.pushButton_4.clicked.connect(self.clear)
        self.pushButton_7.clicked.connect(self.clear)
        self.clear_matching.clicked.connect(self.clear)
        self.pushButton_5.clicked.connect(self.show_histogram)
        self.horizontalSlider.valueChanged.connect(self.update_label_text)
        self.horizontalSlider_2.valueChanged.connect(self.update_label_text)
        self.horizontalSlider_3.valueChanged.connect(self.update_label_text)
        self.horizontalSlider_4.valueChanged.connect(self.update_label_text)
        self.matching_combo.currentIndexChanged.connect(self.handlematchingcombo)
        self.graphicsView.mouseDoubleClickEvent = lambda event: self.browse_image(
            self.graphicsView
        )
        self.graphicsView_3.mouseDoubleClickEvent = lambda event: self.browse_image( self.graphicsView_3)

        self.graphicsView_8.mouseDoubleClickEvent = lambda event: self.browse_image( self.graphicsView_8)
        self.graphicsView_9.mouseDoubleClickEvent = lambda event: self.browse_image( self.graphicsView_9)
        self.graphicsView_12.mouseDoubleClickEvent = lambda event: self.browse_image( self.graphicsView_12)

        self.graphicsView_5.mouseDoubleClickEvent = lambda event: self.browse_image(
            self.graphicsView_5
        )
        self.graphicsView_10.mouseDoubleClickEvent = lambda event: self.browse_image(
            self.graphicsView_10
        )
        self.widget_3.mouseDoubleClickEvent = lambda event: self.browse_image(
            self.widget_3
        )
        self.radioButton.toggled.connect(self.handle_radio_buttons)
        self.radioButton_2.toggled.connect(self.handle_radio_buttons)
        self.horizontalSlider.valueChanged.connect(self.thresholding)
        self.horizontalSlider_2.valueChanged.connect(self.thresholding)
        self.horizontalSlider_3.valueChanged.connect(self.thresholding)
        self.horizontalSlider_4.valueChanged.connect(self.thresholding)


        
        self.label_52.setVisible(False)
        self.graphicsView_12.setVisible(False)
        
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
        self.label_32.setVisible(False)
        self.label_33.setVisible(False)
        self.label_34.setVisible(False)
        self.label_35.setVisible(False)
        self.label_37.setVisible(False)
        self.label_38.setVisible(False)
        self.label_39.setVisible(False)
        self.label_40.setVisible(False)
       
        self.slider_r.setMinimum(0)
        self.slider_r.valueChanged.connect(self.plot_contour)
        self.slider_c.setMinimum(0)
        self.slider_c.valueChanged.connect(self.plot_contour)
        self.pushButton_8.clicked.connect(self.plot_contour)
        self.pushButton_9.clicked.connect(self.clear)
        self.horizontalSlider_9.valueChanged.connect(self.update_label_text)
        self.horizontalSlider_10.valueChanged.connect(self.update_label_text)
        self.horizontalSlider_11.valueChanged.connect(self.update_label_text)
        self.horizontalSlider_12.valueChanged.connect(self.update_label_text)
        self.horizontalSlider_13.valueChanged.connect(self.update_label_text)
        self.horizontalSlider_5.valueChanged.connect(self.update_label_text)
        self.horizontalSlider_6.valueChanged.connect(self.update_label_text)
        self.horizontalSlider_7.valueChanged.connect(self.update_label_text)
        self.horizontalSlider_8.valueChanged.connect(self.update_label_text)
        self.horizontalSlider_14.valueChanged.connect(self.update_label_text)
        self.horizontalSlider_9.setMinimum(0)

        self.horizontalSlider_9.setMaximum(
            100
        )  # Maximum value set to 1 scaled to integer
        self.horizontalSlider_9.setSingleStep(
            1
        )  # Step size set to 0.01 scaled to integer

        # Adjustments for beta (self.horizontalSlider_10)
        self.horizontalSlider_10.setMinimum(0)
        # Starting value set to 0.1 scaled to integer
        self.horizontalSlider_10.setMaximum(
            100
        )  # Maximum value set to 1 scaled to integer
        self.horizontalSlider_10.setSingleStep(
            1
        )  # Step size set to 0.1 scaled to integer

        # Adjustments for gamma (self.horizontalSlider_11)
        self.horizontalSlider_11.setMinimum(0)
        # Starting value set to 0.01 scaled to integer
        self.horizontalSlider_11.setMaximum(
            100
        )  # Maximum value set to 1 scaled to integer
        self.horizontalSlider_11.setSingleStep(
            1
        )  # Step size set to 0.01 scaled to integer

        # Adjustments for max_num_iter (self.horizontalSlider_12)
        self.horizontalSlider_12.setMinimum(500)  # Minimum value set to 500
        # Starting value set to 2500
        self.horizontalSlider_12.setMaximum(10000)  # Maximum value set to 10000
        self.horizontalSlider_12.setSingleStep(500)  # Step size set to 500

        # Adjustments for convergence (self.horizontalSlider_13)
        self.horizontalSlider_13.setMinimum(0)
        self.horizontalSlider_13.setValue(
            10
        )  # Starting value set to 0.1 scaled to integer
        # Maximum value set to 1 scaled to integer
        self.horizontalSlider_13.setSingleStep(
            1
        )  # Step size set to 0.01 scaled to integer

        self.comboBox_2.clear()
        new_options = ["Uniform", "Gaussian", "Salt & pepper"]
        self.comboBox_2.addItems(new_options)
        self.radioButton.setChecked(True)
        self.pushButton_2.clicked.connect(self.hybrid_image)

        self.pushButton_6.clicked.connect(self.show_distribution_curves)
        self.apply_objdetect_btn.clicked.connect(self.handle_apply_objdetect_btn)
       

    def handle_apply_objdetect_btn(self):
        """
         Apply the selected object detection method to the loaded image.
         If Canny edge detection method is selected, the user can choose
         the kernel size using the combo box.

         If Hough line detection method is selected, the user can choose
         the number of lines to detect, lower threshold, upper threshold,
         neighborhood size and step size using the sliders and export the
        detected lines as a new image.

         If edge ellipse detection method is selected, the user can choose
         the thickness and edge color using the sliders and export the
         detected ellipses as a new image.

         If Hough circle detection method is selected, the user can choose
         the minimum and maximum radius, bin threshold, pixel threshold
         and edge color using the sliders and export the detected circles
        as a new image.

         Parameters
         ----------
        None

         Returns
         -------
        None
        """

        if self.types.currentIndex() == 0:
            img = self.original_img
            if self.kernalsize.currentIndex() == 0:
                kernal = 3
            elif self.kernalsize.currentIndex() == 1:
                kernal = 5
            output_img = canny_operator(img, 50, 100, kernal)
            self.display_image(output_img, self.graphicsView_11)

        elif self.types.currentIndex() == 1:
            number_of_lines = self.horizontalSlider_5.value()
            low_threshold = self.horizontalSlider_6.value()
            high_threshold = self.horizontalSlider_7.value()
            # low_threshold = 50
            # high_threshold = 100
            # neighborhood_size = self.horizontalSlider_8.value()
            img = self.original_img
            detected_lines, hough_space = hough_line_detection(
                img, number_of_lines, low_threshold, high_threshold
            )
            output_image = np.copy(img)

            for rho, theta in detected_lines:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite("output/hough_line_detected.jpg", output_image)

            self.display_image(output_image, self.graphicsView_11)

        elif self.types.currentIndex() == 2:
            thickness = int(self.horizontalSlider_5.value() / 10)
            img = self.original_img
            color = "Red"
            # edge_image = canny_operator(img , 50 ,100)
            # out = houghEllipses(edge_image, a_min=20, a_max=100, b_min=20, b_max=100,
            #       delta_a=1, delta_b=1, num_thetas=100, bin_threshold=0.4,
            #       min_edge_threshold=100, max_edge_threshold=200,
            #       pixel_threshold=20, post_process=True)
            output_img = edge_ellipse_detector(img, thickness, color)
            self.display_image(output_img, self.graphicsView_11)

        elif self.types.currentIndex() == 3:
            min_raduis = self.horizontalSlider_5.value()
            max_raduis = self.horizontalSlider_6.value()
            
            circle_color = "Red"
            edges = canny_operator(
                self.original_img, low_threshold=50, high_threshold=100
            )
            output = hough_circles(
                circle_color, edges, min_raduis, max_raduis, 1, 100, 0.4, 20, True
            )
            self.display_image(output, self.graphicsView_11)

   

    def show_distribution_curves(self):
        """
        This function is used to display the distribution curves.

        If the gray image and processed image are loaded, it opens a new window (CurvesWindow) to show the distribution curves.
        If either the gray image or the processed image is not loaded, it shows a warning message box with the message "Gray image or processed image is not loaded yet."

        Returns:
            None
        """
        if self.gray_img.size > 0 and self.processed_img.size > 0:
            distribution_window = CurvesWindow(self.gray_img, self.processed_img)
            distribution_window.exec_()
        else:
            QMessageBox.warning(
                self, "Warning", "Gray image or processed image is not loaded yet."
            )

    def handlecomboBoxChange(self):
        handlecomboBoxChange(self)

    def handlecomboBoxChange2(self):
        handlecomboBoxChange2(self)

    def handelcomboxchanges3(self):
        handelcomboxchanges3(self)

    def handlematchingcombo(self):
        handlematchingcombo(self)    

    def handle_radio_buttons(self):
        """
        This function is used to handle the logic when radio buttons are clicked.

        It checks if the `graphicsView` has contents. If it does, it checks the state of the radio buttons.
        If the first radio button is checked, it calls the `display_image` function with the `original_img` and `graphicsView` as arguments.
        If the second radio button is checked, it calls the `display_image` function with the `gray_img` and `graphicsView` as arguments.

        Returns:
            None
        """
        # Check if graphicsView has contents
        if self.graphicsView.scene() and self.graphicsView.scene().items():
            if self.radioButton.isChecked():
                # Display original image
                self.display_image(self.original_img, self.graphicsView)
            elif self.radioButton_2.isChecked():
                # Display gray image
                self.display_image(self.gray_img, self.graphicsView)


    def update_label_text(self, value):
        """
        This function is used to update the text of a label based on the value of a slider.

        It first retrieves the sender object that emitted the signal. Then, it uses a dictionary to map slider objects to label objects.
        It updates the text of the corresponding label with the slider value, with special handling for specific sliders (9, 11, and 13).

        Args:
            value: The value of the slider.

        Returns:
            None
        """
        # Get the sender object that emitted the signal
        sender = self.sender()

        # Define a dictionary to map slider objects to label objects
        slider_label_map = {
            self.horizontalSlider: self.label_37,
            self.horizontalSlider_2: self.label_38,
            self.horizontalSlider_3: self.label_39,
            self.horizontalSlider_4: self.label_40,
            self.horizontalSlider_5: self.label_32,
            self.horizontalSlider_6: self.label_33,
            self.horizontalSlider_7: self.label_34,
            self.horizontalSlider_8: self.label_35,
            self.horizontalSlider_9: self.label_28,
            self.horizontalSlider_10: self.label_30,
            self.horizontalSlider_11: self.label_36,
            self.horizontalSlider_12: self.label_42,
            self.horizontalSlider_13: self.label_44,
            self.horizontalSlider_14: self.label_53,
        }

        # Update the text of the corresponding label with the slider value
        label = slider_label_map.get(sender)
        if label:
            # Special handling for sliders 9, 11, and 13
            if sender == self.horizontalSlider_9:
                label.setText(str(value / 10000))
            elif sender == self.horizontalSlider_11:
                label.setText(str(value / 1000))
            elif sender == self.horizontalSlider_13:
                label.setText(str(value / 100))
            else:
                label.setText(str(value))

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
        #     This function is used to browse and load an image file from the user's system.

        # Parameters:
        # - widget: The widget on which the image is to be loaded.

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
                elif widget == self.widget_3:
                    self.segmentation_img = self.convert_gry(cv2.imread(selected_file))
                    # Get the dimensions of the image
                    self.seg_img_height, self.seg_img_width = (
                        self.segmentation_img.shape[:2]
                    )
                    self.slider_c.setMaximum(self.seg_img_height * 2)

                    self.slider_r.setMaximum(self.seg_img_width * 2)

                    # Calculate the center coordinates of the image
                    self.center_x = self.seg_img_width // 2
                    self.center_y = self.seg_img_height // 2
                    # Define the radius of the circle
                    self.radius = (
                        min(self.seg_img_width, self.seg_img_height) // 4
                    )  # Choose a reasonable radius
                    # Generate the coordinates of the circle
                    self.s = np.linspace(0, 2 * np.pi, 400)
                    self.r = self.center_x + self.radius * np.sin(self.s)
                    self.c = self.center_y + self.radius * np.cos(self.s)
                    self.init_coords = np.array([self.r, self.c]).T
                    self.plot_image()
                    # print(self.r)
                elif widget == self.graphicsView_8:
                    self.matching_img = cv2.imread(selected_file)
                    self.matching_img = self.convert_gry(self.matching_img)
                    self.display_image(self.matching_img, widget)   
                elif widget == self.graphicsView_12:
                    self.template_img = cv2.imread(selected_file)
                    self.template_img = self.convert_gry(self.template_img)
                    self.display_image(self.template_img, widget)

    def display_image(self, img_data, widget):
        """
        This function is used to display an image on a specified widget while maintaining the aspect ratio of the image.

        Parameters:
        - img_data: The image data to be displayed.
        - widget: The widget on which the image is to be displayed.

        The function first determines whether the input image is grayscale or color by checking the shape of the `img_data` array.
        - If the image is grayscale (2 dimensions), a QImage is created with `QImage.Format_Grayscale8`.
        - If the image is color (3 dimensions), the image is converted to RGB format using `cv2.cvtColor` and a QImage is created with `QImage.Format_RGB888`.
        - If the image format is not supported, a ValueError is raised.

        The function then calculates the new width and height of the image to maintain the aspect ratio when displayed on the widget. The image is resized accordingly.

        A QGraphicsScene is created on the specified widget, and a QGraphicsPixmapItem containing the image data is added to the scene for display.

        Note: This function assumes the presence of classes and functions such as QImage, QPixmap, QGraphicsScene, and QGraphicsPixmapItem from the PyQt5 library.
        """
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
        """
        This function converts a color image to grayscale using the luminosity method.

        Parameters:
        - img_data: The image data to be converted.

        The function first checks if the input image is a color image by checking the shape of the `img_data` array.
        If it is a color image, the function creates a grayscale image with the same dimensions as the input image.
        Then, it iterates over each pixel of the image and calculates the grayscale value using the luminosity method:
        `gray_image[i, j] = int(R * 0.299 + G * 0.587 + B * 0.114)`, where `R`, `G`, and `B` are the individual pixel values.
        Finally, the function clips the values of the grayscale image to the range [0, 255] and returns the grayscale image.

        Note: This function assumes that the input image is a numpy array.
        """
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
        """
        This function converts a NumPy array representing an image to a QImage format compatible with PyQt.

        Parameters:
        - img_array: The NumPy array representing the image.
        - original_img_dtype: The original data type of the image.

        The function performs the following steps:
        1. Converts the image array from BGR format to RGB format using `cv2.cvtColor`.
        2. Gets the dimensions of the image array.
        3. Converts the data type of the image array to match the original image data type.
        4. Creates a QImage object using the image array data, width, height, strides, and format.

        Note: This function assumes that the input image array is in BGR format and that the original image data type is known.
        """
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
        self.display_image(edges, self.graphicsView_7)

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
                noise_value = float(self.lineEdit.text())
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
            # self.display_image(self.processed_img, self.graphicsView_7)

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
        histogram_widget = HistogramWidget(
            histogram_array
        )  # Create instance of HistogramWidget
        if widget.layout() is None:  # Check if widget has a layout
            layout = QVBoxLayout()  # Create a vertical layout
            widget.setLayout(layout)  # Set the layout to the widget
        else:
            layout = widget.layout()  # Get the existing layout
        layout.addWidget(histogram_widget)  # Add HistogramWidget to the layout

    def show_histogram(self):
        rgb_histogram_window = RGBHistogramWindow(self.original_img)
        rgb_histogram_window.show_histogram_window(self.original_img)

    def on_mouse_press(self, event):
        if event.button == 1:  # Left mouse button
            self.center_x = int(event.xdata)  # Store x and y as attributes of the class
            self.center_y = int(event.ydata)
            print("Mouse coordinates (x, y):", self.center_x, self.center_y)
            radius = (
                min(self.seg_img_width, self.seg_img_height) // 4
            )  # Choose a reasonable radius
            s = np.linspace(0, 2 * np.pi, 400)
            self.r = self.center_x + radius * np.sin(
                s
            )  # Use self.center_x and self.center_y here
            self.c = self.center_y + radius * np.cos(s)
            # self.init_coords = np.column_stack((self.r, self.c))
            self.init_coords = np.array([self.r, self.c]).T
            print("Initial contour coordinates:", self.init_coords)
            self.horizontalSlider_9.setValue(1)
            self.horizontalSlider_10.setValue(10)
            self.horizontalSlider_11.setValue(1)
            self.horizontalSlider_12.setValue(500)
            self.horizontalSlider_13.setValue(10)
            self.slider_c.setValue(int(self.seg_img_height / 2))
            self.slider_r.setValue(int(self.seg_img_width / 2))
            self.plot_contour()

    def plot_image(self):
        layout = QVBoxLayout()
        self.widget_3.setLayout(layout)
        self.widget_3.layout().addWidget(self.canvas)
        ax = self.canvas.figure.add_subplot(111)
        ax.clear()
        ax.imshow(self.segmentation_img, cmap=plt.cm.gray)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, self.segmentation_img.shape[1], self.segmentation_img.shape[0], 0])
        self.canvas.draw()

    def plot_contour(self):
        r_value = self.slider_r.value()
        c_value = self.slider_c.value()
        alpha = self.horizontalSlider_9.value() / 10000
        beta = self.horizontalSlider_10.value()
        gamma = self.horizontalSlider_11.value() / 1000
        num_of_iterations = self.horizontalSlider_12.value()
        sigma = self.horizontalSlider_13.value() / 100
        self.init_coords[:, 0] = self.center_x + r_value * np.sin(
            np.linspace(0, 2 * np.pi, 400)
        )
        self.init_coords[:, 1] = self.center_y + c_value * np.cos(
            np.linspace(0, 2 * np.pi, 400)
        )

        snake = active_contour(
            gaussian(self.segmentation_img, 3, preserve_range=False),
            self.init_coords,
            alpha=alpha,
            beta=beta,
            w_line=0,
            w_edge=1,
            gamma=gamma,
            max_px_move=1.0,
            max_num_iter=num_of_iterations,
            convergence=sigma,
        )

        ax = self.canvas.figure.add_subplot(111)
        ax.clear()
        ax.imshow(self.segmentation_img, cmap=plt.cm.gray)
        ax.plot(self.init_coords[:, 1], self.init_coords[:, 0], "--r", lw=3)
        ax.plot(snake[:, 1], snake[:, 0], "-b", lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, self.segmentation_img.shape[1], self.segmentation_img.shape[0], 0])
        self.canvas.draw()

    def clear(self):
        tap_index = self.check_tap()
        if tap_index:
            self.hybrid_input = []
            self.hybrid_input_1 = []
            self.hybrid_input_2 = []
            self.graphicsView_3.scene().clear()
            self.graphicsView_4.scene().clear()
            self.graphicsView_5.scene().clear()
        elif self.tabWidget.currentIndex() == 4:
            self.segmentation_img = None
            # Clear layout of widget_3 (if it contains any widgets)
            self.clear_layout(self.widget_3.layout())
            # Reinitialize variables to their initial values
            self.seg_img_height = 0
            self.seg_img_width = 0
            self.center_x = 0
            self.center_y = 0
            self.radius = 0
            self.s = np.linspace(0, 2 * np.pi, 400)
            self.r = np.zeros_like(self.s)
            self.c = np.zeros_like(self.s)
            self.init_coords = np.array([self.r, self.c]).T
            self.canvas = FigureCanvas(plt.figure())
        elif self.tabWidget.currentIndex() == 3:
            self.transformed_img = []
            self.graphicsView_10.scene().clear()
            self.graphicsView_11.scene().clear()

        elif self.tabWidget.currentIndex() == 5:
            self.graphicsView_8.scene().clear()
            self.graphicsView_9.scene().clear()
            self.graphicsView_12.scene().clear()  
            self.template_img = []
            self.matching_img = []
            self.horizontalSlider_14.setValue(0)  
    
        else:
            self.original_img = []
            self.gray_img = np.array([], dtype=np.uint8)
            self.processed_img = []
            self.graphicsView.scene().clear()
            self.graphicsView_2.scene().clear()
            self.graphicsView_6.scene().clear()
            self.graphicsView_7.scene().clear()
            self.clear_layout(self.widget.layout())
            if self.filters_combo.currentIndex() != 2:
                self.clear_layout(self.widget_2.layout())

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()


if __name__ == "__main__":
    Main_App = QtWidgets.QApplication(sys.argv)
    App = mainwindow()
    App.show()
    sys.exit(Main_App.exec_())
