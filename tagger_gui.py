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
from annotation_object import AnnotationObject, deserialize, serialize
import sys
import os
import atexit

from math import ceil

# This file runs the graphical annotator/viewer program
# Usage: ./tagger_gui.py

anno = None
dots_per_meter = None 

# Gets the path of the PDF file to be viewed/annotated through user input in the terminal
# Is ran at the start of the program.
def get_file_name():
    
    while(True):
        file_name = input("What file would you like to parse? (default path is in ./data/pdf/)>").strip()
        if "/" not in file_name:
            file_path = "./data/pdf/" + file_name
        else:
            file_path = file_name
            file_name = file_path.split('/')[-1]
        
        print(f"file_path = {file_path}")
        if os.path.isfile(file_path):
            break
        else:
            print("Invalid file name. Please try again. >")

    return file_name, file_path


# Cleanup function for when program exits. 
def save_and_exit():
    print("Exiting. Saving file.")
    if anno is not None:
        serialize(anno)
    else:
        print("Anno is None!")
    return
atexit.register(save_and_exit)

# Initialises the annotation object (anno) given a PDF file path
# If a corresponding .pkl file is found, loads it;
# Else, initialise from scratch
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

# The class for a single page (the stack/buffer views)
class SinglePageDisplay(QWidget):
    # initialising function
    # @param title: a the title of the page (stack/buffer) to be displayed
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
        layout.addWidget(self.title)
        layout.addWidget(self.label)
        self.setLayout(layout)


    # draws the page itself, with all its lines, hierarchy lines, etc.
    # @param page_idx       : the index of the page to draw 
    # @param item_idx       : the index of the item (line) to draw (if the line is not on the page to draw, it is not drawn)
    # @param colour         : the colour of the box around the line to be highlighted (item_idx) 
    # @param draw_tree_edges: True iff tree edges are to be drawn

    def draw_page(self, page_idx: int = 0, item_idx: int = 0, colour: QtGui.QColor = QtGui.QColor("blue"), draw_tree_edges = True):

        print(f"Drawing page {page_idx}")
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
            # highlighting the box
            # print(d)
            if(d == -1): # discard operation
                return
            painter.setBrush(QtGui.QColor.fromHsvF(min(d/10, 1), 0.8, 0.8, 0.1)) #assuming that the max. dep does not go over 10
            
            x_, y_, width_, height_ = element_['x'], element_['y'], element_['width'], element_['height']
            x_ = round(x_ * self.width / 100)
            y_ = round(y_ * self.height / 100)
            width_= round(width_ * self.width / 100)
            height_ = round(height_ * self.height / 100)
            painter.drawRect(x_, y_, width_, height_)


        # highlighting
        painter.setPen(QtGui.QColor.fromHsvF(0, 0, 0, 0)) #no boundary on highlights
        for i in range(anno.n_lines):
            d = anno.depth[i]
            element_ = anno.json_format[i]
            if(element_['page'] > page_idx):
                break 
            if(element_['page'] == page_idx):
                draw_box(element_, d)

        # helper function to get ELement INFO.
        # @param    idx : the index of the line to return info about 
        # @return   x, y: positions (real value from 0 to 1) of the upper-left position 
        # @retunrn  page: the page that the element is on
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
        
        # Draw tree edges
        if(draw_tree_edges):
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
                    if(page_from == page_to and page_from == page_idx):
                        # print(f"Drawing! type = {entry['type']}, width: {self.width}, height: {self.height}, fx = {fx}, fy = {fy}, tx = {tx}, ty = {ty}")
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
        # draw box around buffer element
        
        def draw_box_element(index):
            # drawing a box around the current element (line)
            to_draw_page = 0
            if(index >= len(anno.json_format)):
                index = len(anno.json_format) - 1
            if(0 <= index ):
                to_draw_page = anno.json_format[index]['page']
            if(to_draw_page != page_idx):
                return
            x, y, width, height = None, None, None, None
            painter.setPen(colour)
            if(index >= 0):
                element = anno.json_format[index]
                x, y, width, height = element['x'], element['y'], element['width'], element['height']
                x = round(x * self.width / 100)
                y = round(y * self.height / 100)
                width = round(width * self.width / 100)
                height = round(height * self.height / 100)
            else:
                x, y, width, height = round(self.width * 0.4), round(self.height * 0.05), font_sz * 5, font_sz

            painter.drawRect(x, y, width, height)
            
        draw_box_element(item_idx)
        painter.end()
        self.label.setPixmap(canvas)

# Displays the options (helper text)
class OptionsMenu(QWidget):
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()
        self.footer = QLabel()
        layout.addWidget(self.footer)
        self.setLayout(layout)

    def set_text(self, text):
        self.footer.setText(text)   

# The wrapper around all widgets
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

# The main application, in charge of housing widgets and carrying out IO
class MainWindow(QMainWindow):

    instructions = "[s]ubordinate/1; [m]erge/2; [p]op/3; [u]ndo/4; [d]iscard/5; [q]uit"

    def __init__(self):
        super().__init__()

        # self.setWindowTitle(f"Annotating: {file_path}")
        self.mainWidget = MainWidget()
        self.setCentralWidget(self.mainWidget)

        self.stack_idx = 0
        self.buffer_idx = 0

        self.mainWidget.set_footer_text(self.instructions)
        self.mainWidget.stackpage.draw_page(page_idx = self.stack_idx, item_idx = anno.stack[-1])
        self.mainWidget.bufferpage.draw_page(page_idx = self.buffer_idx, item_idx = anno.current_idx)

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
        

        if(e.key() == Qt.Key.Key_Left.value):
            print(f"Buffer idx was {self.buffer_idx}")
            self.buffer_idx = max(self.buffer_idx - 1, 0)
            print(f"Now {self.buffer_idx}")
        elif(e.key() == Qt.Key.Key_Right.value):
            print(f"Buffer idx was {self.buffer_idx}")
            self.buffer_idx = min(self.buffer_idx + 1, anno.n_pages - 1)
            print(f"Now {self.buffer_idx}")
        else:
            self.stack_idx = 0 if anno.stack[-1] == -1 else anno.json_format[anno.stack[-1]]['page']
            self.buffer_idx = anno.json_format[min(anno.current_idx, len(anno.json_format) - 1)]['page']
        self.mainWidget.stackpage.draw_page(page_idx = self.stack_idx, item_idx = anno.stack[-1])
        self.mainWidget.bufferpage.draw_page(page_idx = self.buffer_idx, item_idx = anno.current_idx)


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