from PyQt5 import QtWidgets,uic,QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem, QFileDialog, QGraphicsScene
import numpy as np
import sys
import cv2
class mainwindow (QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        uic.loadUi("mainwindow.ui",self)
        self.original_img =[]
        self.gray_img=[]
        self.processed_img=[]
        self.hybrid_input=[]
        self.filters_combo.currentIndexChanged.connect(self.handlecomboBoxChange)
        self.comboBox_2.currentIndexChanged.connect(self.handlecomboBoxChange2)
        self.graphicsView.mouseDoubleClickEvent = lambda event: self.browse_image(self.graphicsView)
        self.graphicsView_3.mouseDoubleClickEvent = lambda event: self.browse_image(self.graphicsView_3)
        self.graphicsView_5.mouseDoubleClickEvent = lambda event: self.browse_image(self.graphicsView_5)
        self.label_3.setVisible(True)
        self.comboBox_2.setVisible(True)
        self.comboBox_2.clear()
        new_options = ["Uniform", "Gaussian", "Salt & pepper"]
        self.comboBox_2.addItems(new_options)
        self.radioButton.setChecked(True)

    def handlecomboBoxChange(self):
        if self.filters_combo.currentIndex() ==0:
            self.label_3.setVisible(True)
            self.comboBox_2.setVisible(True)
            self.label_3.setText("Noise Type")
            self.comboBox_2.clear()
            new_options = ["Uniform", "Gaussian", "Salt & pepper"]
            self.comboBox_2.addItems(new_options)
            
        elif self.filters_combo.currentIndex() ==1:
            self.label_3.setVisible(True)
            self.comboBox_2.setVisible(True)
            self.label_3.setText("Filter Type")
            self.comboBox_2.clear()
            new_options = ["Average", "Gaussian", "Median"]
            self.comboBox_2.addItems(new_options)
            
        elif self.filters_combo.currentIndex() ==2:
            self.label_3.setVisible(True)
            self.comboBox_2.setVisible(True)
            self.label_3.setText("Detection Method")
            self.comboBox_2.clear()
            new_options = ["Sobel" , "Roberts" , "Prewitt","Canny"]
            self.comboBox_2.addItems(new_options)
            
        elif self.filters_combo.currentIndex() ==3:
            self.label_3.setVisible(True)
            self.comboBox_2.setVisible(True)
            self.label_3.setText("Thresholding Method")
            self.comboBox_2.clear()
            new_options = ["Global", "Local"]
            self.comboBox_2.addItems(new_options)
        
        elif self.filters_combo.currentIndex()==6 or self.filters_combo.currentIndex()==7:
            self.label_3.setVisible(False)
            self.comboBox_2.setVisible(False)
            self.label_13.setVisible(False)
            self.label_21.setVisible(False)
            self.label_22.setVisible(False)
            self.label_23.setVisible(False)
            self.horizontalSlider.setVisible(False)
            self.horizontalSlider_2.setVisible(False)
            self.horizontalSlider_3.setVisible(False)
            self.horizontalSlider_4.setVisible(False)
            self.label_4.setVisible(True)
            self.label_4.setText("Cutoff frequency")
            self.lineEdit.setVisible(True)
        else:
            self.label_3.setVisible(False)
            self.comboBox_2.setVisible(False)
            self.label_13.setVisible(False)
            self.label_21.setVisible(False)
            self.label_22.setVisible(False)
            self.label_23.setVisible(False)
            self.horizontalSlider.setVisible(False)
            self.horizontalSlider_2.setVisible(False)
            self.horizontalSlider_3.setVisible(False)
            self.horizontalSlider_4.setVisible(False)
            
    def handlecomboBoxChange2(self):
        if self.filters_combo.currentIndex() ==0:
            self.label_13.setVisible(False)
            self.label_21.setVisible(False)
            self.label_22.setVisible(False)
            self.label_23.setVisible(False)
            self.horizontalSlider.setVisible(False)
            self.horizontalSlider_2.setVisible(False)
            self.horizontalSlider_3.setVisible(False)
            self.horizontalSlider_4.setVisible(False)
            self.comboBox.setVisible(False)
            self.label_12.setVisible(False)
            if self.comboBox_2.currentIndex()==0: 
                    self.label_4.setText("Noise Value")
                    self.label_5.setVisible(False)
                    self.label_6.setVisible(False)
                    self.lineEdit_2.setVisible(False)
                    self.lineEdit_3.setVisible(False)
            elif self.comboBox_2.currentIndex()==1:
                    self.label_5.setVisible(True)
                    self.lineEdit_2.setVisible(True)
                    self.label_4.setText("Mean")
                    self.label_5.setText("Sigma")
                    self.label_6.setVisible(False)
                    self.lineEdit_3.setVisible(False)
            else:
                    self.label_5.setVisible(True)
                    self.lineEdit_2.setVisible(True)
                    self.label_4.setText("W pixels")
                    self.label_5.setText("B pixels")
                    self.label_6.setVisible(False)
                    self.lineEdit_3.setVisible(False)
        elif self.filters_combo.currentIndex() ==1:
            self.label_13.setVisible(False)
            self.label_21.setVisible(False)
            self.label_22.setVisible(False)
            self.label_23.setVisible(False)
            self.horizontalSlider.setVisible(False)
            self.horizontalSlider_2.setVisible(False)
            self.horizontalSlider_3.setVisible(False)
            self.horizontalSlider_4.setVisible(False)
            self.comboBox.setVisible(False)
            self.label_12.setVisible(False)
            if self.comboBox_2.currentIndex()==0:
                self.label_4.setVisible(True)
                self.lineEdit.setVisible(True) 
                self.label_4.setText("Kernel size")
                self.label_5.setVisible(False)
                self.label_6.setVisible(False)
                self.lineEdit_2.setVisible(False)
                self.lineEdit_3.setVisible(False)
            elif self.comboBox_2.currentIndex()==1:
                self.label_4.setVisible(True)
                self.lineEdit.setVisible(True)
                self.label_4.setText("Sigma")
                self.label_5.setVisible(False)
                self.label_6.setVisible(False)
                self.lineEdit_2.setVisible(False)
                self.lineEdit_3.setVisible(False)
            else:
                self.label_4.setVisible(False)
                self.label_5.setVisible(False)
                self.label_6.setVisible(False)
                self.lineEdit.setVisible(False)
                self.lineEdit_2.setVisible(False)
                self.lineEdit_3.setVisible(False)
        elif self.filters_combo.currentIndex() ==2:
            self.label_13.setVisible(False)
            self.label_21.setVisible(False)
            self.label_22.setVisible(False)
            self.label_23.setVisible(False)
            self.horizontalSlider.setVisible(False)
            self.horizontalSlider_2.setVisible(False)
            self.horizontalSlider_3.setVisible(False)
            self.horizontalSlider_4.setVisible(False)
            self.label_12.setText("Detection type") 
            if self.comboBox_2.currentIndex()!=3:
                self.comboBox.clear()
                new_options = ["Horizontal", "Vertical","Both"]
                self.comboBox.addItems(new_options)
                self.label_4.setVisible(False)
                self.label_5.setVisible(False)
                self.label_6.setVisible(False)
                self.lineEdit.setVisible(False)
                self.lineEdit_2.setVisible(False)
                self.lineEdit_3.setVisible(False)
                self.comboBox.setVisible(True)
                self.label_12.setVisible(True)
            elif self.comboBox_2.currentIndex()==3:
                    self.label_4.setVisible(True)
                    self.lineEdit.setVisible(True)
                    self.comboBox.setVisible(True)
                    self.label_12.setVisible(True)
                    self.label_12.setText("Mask size")
                    self.label_4.setText("Sigma")
                    self.comboBox.clear()
                    new_options = ["3x3", "5x5"]
                    self.comboBox.addItems(new_options)
        

        elif self.filters_combo.currentIndex() ==3:
            if self.comboBox_2.currentIndex()==0: 
                self.label_13.setVisible(True)
                self.horizontalSlider.setVisible(True)
                self.label_13.setVisible(True)
                self.label_21.setVisible(False)
                self.label_22.setVisible(False)
                self.label_23.setVisible(False)
                self.horizontalSlider.setVisible(True)
                self.horizontalSlider_2.setVisible(False)
                self.horizontalSlider_3.setVisible(False)
                self.horizontalSlider_4.setVisible(False)
            else:
                self.label_4.setVisible(False)
                self.label_5.setVisible(False)
                self.label_6.setVisible(False)
                self.lineEdit.setVisible(False)
                self.lineEdit_2.setVisible(False)
                self.lineEdit_3.setVisible(False)
                self.label_13.setVisible(True)
                self.label_21.setVisible(True)
                self.label_22.setVisible(True)
                self.label_23.setVisible(True)
                self.horizontalSlider.setVisible(True)
                self.horizontalSlider_2.setVisible(True)
                self.horizontalSlider_3.setVisible(True)
                self.horizontalSlider_4.setVisible(True)
                self.comboBox.setVisible(False)
                self.label_12.setVisible(False)
        
    def browse_image(self,widget):
            file_dialog = QFileDialog()
            file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.gif)")
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            
            if file_dialog.exec_():
                selected_file = file_dialog.selectedFiles()[0]
                # Create an instance of the Image class with the selected image path
                self.original_img=  cv2.imread(selected_file)
                self.hybrid_input=  cv2.imread(selected_file)
                if widget == self.graphicsView: 
                    if self.radioButton_2.isChecked():
                        self.radioButton.setChecked(False)
                        self.gray_img =self.convert_gry(self.original_img)
                        # self.display_image(self.gray_img,widget)
                        # self.display_image(self.gray_img,self.graphicsView_6)
                    else:
                        self.radioButton_2.setChecked(False)
                        self.display_image(self.original_img,widget)
                        # self.display_image(self.gray_img,self.graphicsView_6)
                else:
                     self.display_image(self.hybrid_input,widget)


    def display_image(self, img_data, widget):
        # Get the size of the widget
        widget_width = widget.width()
        widget_height = widget.height()
        if self.radioButton.isChecked()==True:
             img_height, img_width, _ = img_data.shape
        # else:
        #     img_height, img_width, _ = img_data.shape + (1,)
        aspect_ratio = img_width / img_height

        # Calculate the new width and height to maintain the aspect ratio
        if aspect_ratio > 1:
            new_width = widget_width
            new_height = int(widget_width / aspect_ratio)
        else:
            new_height = widget_height
            new_width = int(widget_height * aspect_ratio)

        # Resize the image using OpenCV
        resized_image = cv2.resize(img_data, (new_width, new_height))

        # Convert the OpenCV image to QImage
        height, width, _ = resized_image.shape
        # + (1,)
        bytes_per_line = 3 * width
        q_image = QImage(resized_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

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
            R, G, B = img_data[:,:,0], img_data[:,:,1], img_data[:,:,2]
            update = R * 0.299 + G * 0.587 + B * 0.114
            # Ensure that the values are normalized to the range [0, 255]
            update = np.clip(update, 0, 255).astype(np.uint8)
            return np.array(update)

         

    def resize_image(self, target_width, target_height):
        # Resize the image to the specified width and height
        resized_image = cv2.resize(self.image_data, (target_width, target_height))

        # Convert the resized image from BGR to RGB
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        height, width, channel = resized_image_rgb.shape  # Get image shape
        bytes_per_line = 3 * width  # RGB image has 3 channels

        # Update the QImage with the resized RGB image data
        self.dimage = QtGui.QImage(resized_image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

if __name__ == "__main__":
    Main_App = QtWidgets.QApplication(sys.argv)
    App=mainwindow()
    App.show()
    sys.exit(Main_App.exec_())
