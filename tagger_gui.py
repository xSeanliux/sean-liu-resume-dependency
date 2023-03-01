from PyQt6.QtWidgets import (
    QApplication, 
    QWidget, 
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton
)
from PyQt6.QtCore import Qt
from PyQt6 import QtGui
from tagger_program import get_file_name
from annotation_object import AnnotationObject, deserialize, serialize
import sys
import os
import atexit

from math import ceil

anno = None
dots_per_meter = None 

def save_and_exit():
    print("Exiting. Saving file.")
    if anno is not None:
        print(f"Size of anno is {sys.getsizeof(anno.__dict__)}")
        serialize(anno)
    else:
        print("Anno is none!")
    return
atexit.register(save_and_exit)

def init_anno(pdf_file_path):
    assert len(pdf_file_path) > 4 and pdf_file_path[-4:] == ".pdf"
    global anno 
    anno = AnnotationObject()
    print("FILEPATH WAS", pdf_file_path)
    file_path = pdf_file_path[:-2] + "kl"
    print("FILEPATH =", file_path)
    if os.path.exists(file_path):
        print(f"{file_path}.pkl exists. Loading.")
        anno = deserialize(file_path)
        print(f"Finished loading. Starting from index {anno.current_idx}")
    else:
        print("Loading from file path.")
        anno = AnnotationObject(pdf_file_path)

class SinglePageDisplay(QWidget):
    def __init__(self, title):
        super().__init__()
        self.height = 840   # height is constant
        self.width = int(ceil(self.height * anno.wrapper.page_width / anno.wrapper.page_height))
        self.pixelRatio = 4

        layout = QVBoxLayout()

        self.title = QLabel()
        self.title.setText(title)

        self.label = QLabel()
        canvas = QtGui.QPixmap(self.width * self.pixelRatio, self.height * self.pixelRatio)
        canvas.setDevicePixelRatio(self.pixelRatio) # using a retina display
        canvas.fill(Qt.GlobalColor.white)
        self.label.setPixmap(canvas)
        self.draw_page()
        layout.addWidget(self.title)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def draw_page(self, index: int = 1, colour: QtGui.QColor = QtGui.QColor("blue")):
        element = anno.json_format[index]

        canvas = self.label.pixmap()
        painter = QtGui.QPainter(canvas)

        page_idx = element['page']
        x, y, width, height = element['x'], element['y'], element['width'], element['height']
        x = round(x * self.width / 100)
        y = round(y * self.height / 100)
        width = round(width * self.width / 100)
        height = round(height * self.height / 100)

        scaledImage = anno.qt_image_list[page_idx].scaled(
            int(self.width * self.pixelRatio),
            int(self.height * self.pixelRatio),
            aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio
        )
        scaledImage.setDevicePixelRatio(self.pixelRatio)
        painter.drawImage(0, 0, scaledImage)

        if colour is not None:
            painter.setPen(colour)
        painter.drawRect(x, y, width, height)
        painter.end()
        self.label.setPixmap(canvas)

class OptionsMenu(QWidget):
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()
        self.footer = QLabel()
        layout.addWidget(self.footer)
        self.setLayout(layout)

    def set_text(self, text):
        self.footer.setText(text)   

class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        pages_layout = QHBoxLayout()
        footer_layout = QHBoxLayout()
        self.options = OptionsMenu()
        self.stackpage = SinglePageDisplay("Stack View")
        self.bufferpage = SinglePageDisplay("Buffer View")

        pages_layout.addWidget(self.stackpage)
        pages_layout.addWidget(self.bufferpage)

        footer_layout.addWidget(self.options)

        layout.addLayout(pages_layout)
        layout.addLayout(footer_layout)
        self.setLayout(layout)
    
    def set_footer_text(self, text):
        self.options.set_text(text)

class MainWindow(QMainWindow):

    instructions = "[s]ubordinate/1; [m]erge/2; [p]op/3; [u]ndo/4; [d]iscard/5; [q]uit"

    def __init__(self):
        super().__init__()

        # self.setWindowTitle(f"Annotating: {file_path}")
        self.mainWidget = MainWidget()
        self.setCentralWidget(self.mainWidget)

        self.mainWidget.set_footer_text(self.instructions)
        self.mainWidget.stackpage.draw_page(anno.stack[-1])
        self.mainWidget.bufferpage.draw_page(anno.current_idx)

    def keyPressEvent(self, e):
        res = e.text().lower()
        res_string = ""
        if(res == 's' or res == '1'):
            res_string = "SUBORDINATE"
            anno.subordinate_action()
        if(res == 'm' or res == '2'):
            res_string = "MERGE"
            anno.merge_action()
        if(res == 'p' or res == '3'):
            res_string = "POP"
            anno.pop_action()
        if(res == 'u' or res == '4'):
            res_string = "UNDO"
            anno.undo()
        if(res == 'd' or res == '5'):
            res_string = "DISCARD"
            anno.discard()
        if(res == 'q' or e.key() == Qt.Key.Key_Escape.value):
            res_string = "QUIT - Saving File..."
            self.setWindowTitle(res_string)
            sys.exit(0)

        self.mainWidget.stackpage.draw_page(anno.stack[-1])
        self.mainWidget.bufferpage.draw_page(anno.current_idx)

        self.setWindowTitle(self.instructions + res_string)
            
if __name__ == "__main__":
    # file_name, file_path = get_file_name()
    # print(f"File_path = {file_path}")
    # You need one (and only one) QApplication instance per application.
    # Pass in sys.argv to allow command line arguments for your app.
    # If you know you won't use command line arguments QApplication([]) works too.
    app = QApplication(sys.argv)
    init_anno(pdf_file_path = "/Users/liusean/Desktop/Projects/Coding/ML/ForwardLab/SP23/data/pdf/cv_7.pdf")
    # Create a Qt widget, which will be our window.
    assert anno is not None
    window = MainWindow()
    window.show()  # IMPORTANT!!!!! Windows are hidden by default.

    # Start the event loop.
    app.exec()


# Your application won't reach here until you exit and the event
# loop has stopped.