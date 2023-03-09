import pdfminer
import sys
import matplotlib.patches as patches
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLineHorizontal
from tqdm import tqdm
from pdfminer.layout import LAParams
from PIL.ImageQt import ImageQt
from PyQt6.QtGui import QImage
from math import ceil
import functools

def convert_to_ls(x, y, width, height, original_width, original_height):
    return x / original_width * 100.0, y / original_height * 100.0, \
           width / original_width * 100.0, height / original_height * 100

class PDFWrapper: 
    
    # Assumptions:
    # 1. The given CV's are in single-column format 
    # 2. As such, when sorting the lines by their coordinates (LAParams(boxes_flow = None) on line 29), individual entries are in consecutive lines and all we have to do is to merge them


    elements = []
    lines = []
    page_height = 0
    page_width = 0
    page_count = 0
    render_dpi = 100
    pdfminer_factor = render_dpi / 72   # pdfminer gives all bounding boxes assuming 72dpi, so we have to apply this correction
    dots_per_meter = ceil(render_dpi * 39.37) # qimage uses dots_per_meter, and there are roughly 39.37 inches in a meter...


    pil_image_list = None # A list of PIL images

    def line_cmp(self, l1, l2):
        bbox1 = l1.bbox 
        bbox2 = l2.bbox 
        x1, y1 = bbox1[0], bbox1[1]
        x2, y2 = bbox2[0], bbox2[1]
        if abs(y1 - y2) < 5:
            return x1 - x2
        return y2 - y1 # higher y values are higher up and should be first

    def __init__(self, fname, laparams_ = LAParams(boxes_flow = None)):

        print("Using laparams = ", laparams_)
        for page_layout in tqdm(extract_pages(fname, laparams = laparams_, caching = False)):
            page_elements = []
            page_lines = []
            self.page_height, self.page_width = page_layout.height * self.pdfminer_factor, page_layout.width * self.pdfminer_factor
            for element in page_layout:
                
                if isinstance(element, LTTextContainer):
                    page_elements.append(element)
                    for line in element:
                        if(isinstance(line, LTTextLineHorizontal)):
                            page_lines.append(line)
        
            self.elements.append(page_elements)

            page_lines = sorted(page_lines, key = functools.cmp_to_key(self.line_cmp))

            self.lines.append(page_lines)
        
        self.pil_image_list = convert_from_path(fname, dpi = self.render_dpi) # This returns a list even for a 1 page pdf
        
        self.page_count = len(self.pil_image_list)

    def get_bounding_box(self, element):
        bbox = element.bbox
        x0 = bbox[0] * self.pdfminer_factor
        y0 = bbox[1] * self.pdfminer_factor
        width = element.width * self.pdfminer_factor
        height = element.height * self.pdfminer_factor
        y0 = self.page_height - y0 - height
        return x0, y0, width, height

    def render_page(self, page, ax = None, render_boxes = True, render_arrows = True):
        assert 0 <= page < self.page_count

        img = self.pil_image_list[page]
        objs = self.lines[page]
        ax = ax or plt.gca()
        ax.imshow(img)

        for i, element in enumerate(objs):
            x0, y0, width, height = self.get_bounding_box(element)
            if(render_boxes):
                ax.add_patch(patches.Rectangle((x0, y0), width, height, linewidth=0.1, edgecolor='r', facecolor='none'))
        
            if(render_arrows and i < len(objs) - 1):
                x0, y0, w, h = self.get_bounding_box(element)
                x1, y1, _, _ = self.get_bounding_box(objs[i + 1])
                ax.arrow(x0 + w, y0 + h, x1 - x0 - w, y1 - y0 - h, width = 1e-3)
            
        return ax
    
    def get_ydiff_distribution(self):
        # @param objs: a 2D array (n_pages, n_elements). 
        # @return: a list of all the y-differences collected per page and then aggregated.

        y_diffs = [] 
        for page_elements in self.elements:
            #print(len(page_elements))
            y_values = np.array([ 
                [obj.bbox[1] * self.pdfminer_factor, obj.bbox[3] * self.pdfminer_factor] \
                for obj in page_elements \
            ])
            
            for i in range(y_values.shape[0] - 1):
                
                y_dist = np.array([np.abs(a - b) for a in y_values[i] for b in y_values[i + 1]]).min()
                cur_height = y_values[i][1] - y_values[i][0]
                nxt_height = y_values[i + 1][1] - y_values[i + 1][0]
                y_diffs.append(np.round(y_dist/cur_height, 3)) #round to 2 values
                y_diffs.append(np.round(y_dist/nxt_height, 3)) #round to 2 values
        return np.array(y_diffs)
    
    def get_lines(self):
        return self.lines
    
    def get_json(self):
        # gets the JSON content as a string from wrapper_obj
        lines = self.get_lines()
        annotations = []
        for n_page, page_line in enumerate(lines): 
            for i, line in enumerate(page_line):
                x0, y0, width, height = self.get_bounding_box(line)
                x, y, width_t, height_t = convert_to_ls(x0, y0, width, height, self.page_width, self.page_height)
                annotations.append({
                    'text': line.get_text().strip(),
                    'page': n_page,
                    'idx_in_page': i,
                    'original_dimensions': {
                        'x0': x0,
                        'y0': y0,
                        'width': width,
                        'height': height
                    },
                    'x': x,
                    'y': y,
                    'width': width_t,
                    'height': height_t,
                })

        return annotations