from PyQt5 import QtWidgets,uic,QtGui
from PyQt5.QtWidgets import QGraphicsPixmapItem, QFileDialog, QGraphicsScene
import sys
import cv2
class mainwindow (QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        uic.loadUi("mainwindow.ui",self)
        self.original_img =[]
        self.processed_img=[]
        self.filters_combo.currentIndexChanged.connect(self.handlecomboBoxChange)
        self.comboBox_2.currentIndexChanged.connect(self.handlecomboBoxChange2)
        self.graphicsView.mouseDoubleClickEvent = lambda event: self.browse_and_display_image()
        self.label_3.setVisible(True)
        self.comboBox_2.setVisible(True)
        self.comboBox_2.clear()
        new_options = ["Uniform", "Gaussian", "Salt & pepper"]
        self.comboBox_2.addItems(new_options)

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
            if self.comboBox_2.currentIndex()==0: 
                pass
            else:
                pass
        elif self.filters_combo.currentIndex() ==4:
             self.label_3.setVisible(False)
             self.comboBox_2.setVisible(False)
        else:
            self.label_3.setVisible(False)
            self.comboBox_2.setVisible(False)
            
    def handlecomboBoxChange2(self):
        if self.filters_combo.currentIndex() ==0:
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
            if self.comboBox_2.currentIndex()==0: 
                pass
            elif self.comboBox_2.currentIndex()==1:
                pass
            elif self.comboBox_2.currentIndex()==2:
                pass
            else:
                pass
    def browse_and_display_image(self):
            file_dialog = QFileDialog()
            file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.gif)")
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            
            if file_dialog.exec_():
                selected_file = file_dialog.selectedFiles()[0]
                # Create an instance of the Image class with the selected image path
                self.original_img=  cv2.imread(selected_file)
                # Resize the image to a specific width and height
                target_width = 300  
                target_height = 200  
                self.original_img.resize_image(target_width, target_height)

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
