from PyQt6.QtWidgets import (
    QApplication, 
    QWidget, 
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton
)
from PyQt6.QtCore import Qt, QPoint
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
        # print(f"Size of anno is {sys.getsizeof(anno.__dict__)}")
        serialize(anno)
    else:
        print("Anno is None!")
    return
atexit.register(save_and_exit)

def init_anno(pdf_file_path):
    assert len(pdf_file_path) > 4 and pdf_file_path[-4:] == ".pdf"
    global anno 
    anno = AnnotationObject()
    # print("FILEPATH WAS", pdf_file_path)
    pkl_file_path = pdf_file_path.replace("pdf", "pkl")
    # print("FILEPATH =", pkl_file_path)
    if os.path.exists(pkl_file_path):
        print(f"{pkl_file_path} exists. Loading.")
        anno = deserialize(pkl_file_path)
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

    def draw_page(self, index: int = 0, colour: QtGui.QColor = QtGui.QColor("blue"), draw_tree_edges = False):

        element = None
        if(index < 0):
            element = anno.json_format[0]
        elif(index >= anno.n_lines):
            element = anno.json_format[-1]
        else:
            element = anno.json_format[index]
        page_idx = element['page']

        canvas = self.label.pixmap()
        painter = QtGui.QPainter(canvas)

        scaledImage = anno.qt_image_list[page_idx].scaled(
            int(self.width * self.pixelRatio),
            int(self.height * self.pixelRatio),
            aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio
        )
        scaledImage.setDevicePixelRatio(self.pixelRatio)

        painter.drawImage(0, 0, scaledImage)

        def draw_box(element_, d):
            painter.setBrush(QtGui.QColor.fromHsvF(d/10, 0.8, 0.8, 0.1)) #assuming that the max. dep does not go over 10
            
            x_, y_, width_, height_ = element_['x'], element_['y'], element_['width'], element_['height']
            x_ = round(x_ * self.width / 100)
            y_ = round(y_ * self.height / 100)
            width_= round(width_ * self.width / 100)
            height_ = round(height_ * self.height / 100)
            painter.drawRect(x_, y_, width_, height_)


        # highlighting
        painter.setPen(QtGui.QColor.fromHsvF(0, 0, 0, 0)) #no boundary on highlights
        for i in range(max(0, index), anno.n_lines):
            d = anno.depth[i]
            element_ = anno.json_format[i]
            if(element_['page'] != page_idx):
                break 
            if(d == -1):
                continue
            draw_box(element_, d)

        for i in range(index - 1, -1, -1):
            d = anno.depth[i]
            element_ = anno.json_format[i]
            if(element_['page'] != page_idx):
                break 
            if(d == -1):
                continue
            draw_box(element_, d)

        
        # drawing parent nodes

        def get_el_info(idx):
            x, y, page = None, None, None 
            if idx == -1:
                page = 0
                x = int(round(self.width * 0.4))
                y = int(round(self.height * 0.05))
            else:
                page = anno.json_format[idx]['page']
                x = int(anno.json_format[idx]['x'] * self.width / 100)
                y = int(anno.json_format[idx]['y'] * self.height / 100)
            return x, y, page
        
        print("Drawing page ", page_idx)
        for entry in anno.record:
            if(entry['type'] == 'merge' or entry['type'] == 'subordinate'):
                # print("type = ", entry['type'])
                if(entry['type'] == 'merge'):
                    painter.setPen(QtGui.QColor.fromRgb(255, 136, 0))
                if(entry['type'] == 'subordinate'):
                    painter.setPen(QtGui.QColor.fromRgb(0, 206, 38))
                from_idx = entry['from']
                to_idx = entry['to']
                fx, fy, page_from = get_el_info(from_idx)
                tx, ty, page_to = get_el_info(to_idx)
                print(page_from)
                if(page_from == page_to and page_from == page_idx):
                    print(f"Drawing! type = {entry['type']}, width: {self.width}, height: {self.height}, fx = {fx}, fy = {fy}, tx = {tx}, ty = {ty}")
                    painter.drawLine(fx, fy, tx, ty)
        # Draw $ROOT element on first page
        x = y = width = height = None
        painter.setBrush(QtGui.QColor.fromHsvF(0, 0, 0, 0))
        painter.setPen(colour)
        if(page_idx == 0):
            font_sz = 30
            x, y, width, height = round(self.width * 0.4), round(self.height * 0.05), font_sz * 5, font_sz
            font = painter.font()
            font.setPixelSize(font_sz)
            painter.setFont(font)
            painter.drawText(QPoint(x, y + height), "$ROOT")

        if(index >= 0):
            painter.setPen(colour)
            x, y, width, height = element['x'], element['y'], element['width'], element['height']
            x = round(x * self.width / 100)
            y = round(y * self.height / 100)
            width = round(width * self.width / 100)
            height = round(height * self.height / 100)

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
    
    file_name, pdf_file_path = get_file_name()
    init_anno(pdf_file_path = pdf_file_path)    

    assert anno is not None
    app = QApplication(sys.argv)
    # Create a Qt widget, which will be our window.
    
    window = MainWindow()
    window.show()  # IMPORTANT!!!!! Windows are hidden by default.

    # Start the event loop.
    app.exec()


# Your application won't reach here until you exit and the event
# loop has stopped.