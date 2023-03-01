
from pdf_wrapper import PDFWrapper
import pickle as pkl
from PIL.ImageQt import ImageQt
import sys
import os

def serialize(anno_object):
    print(f"Serializing... {anno_object}")
    if anno_object is not None:
        pkl_file_name = anno_object.file_path[:-2] + "kl"
        with open(pkl_file_name, "wb") as handle:
            pkl.dump(anno_object, handle)
            print(f"Dumped to {pkl_file_name}.")
    else:
        print("Error! anno object is None!")

        
def deserialize(pkl_file_path):
    print(f"{pkl_file_path} exists: {os.path.exists(pkl_file_path)}, stats = {os.stat(pkl_file_path)}")
    assert len(pkl_file_path) > 4 and pkl_file_path[-4:] == ".pkl"
    with open(pkl_file_path, "rb") as handle:
        anno = pkl.load(handle)
        return anno


class AnnotationObject: 
    file_path = None
    record = []
    wrapper = None
    json_format = None
    is_done = False
    # record contains objects which look like 
    # {
    #     'from': (number, index)
    #     'to': (number, index)
    #     type: string, one of ('subordinate', 'merge', 'pop')
    # }
    current_idx = 1
    n_lines = 0
    # current_idx: the current object being processed
    stack = []
    # stack is the stack being used, records the indices only, -1 is root
    qt_image_list = None
    def subordinate_action(self):
        if(self.is_done):
            return
        
        self.record.append({ 
            'from': self.current_idx, 
            'to': self.stack[-1], 
            'type': 'subordinate' 
        })
        self.stack.append(self.current_idx)
        self.current_idx += 1

        self.is_done = (self.current_idx == self.n_lines)
    def merge_action(self):
        if(self.is_done):
            return
        
        self.record.append({ 
            'from': self.current_idx, 
            'to': self.stack[-1], 
            'type': 'merge' 
        })
        self.stack.pop() # representative element is the last line of the whole entry
        self.stack.append(self.current_idx)
        self.current_idx += 1

        self.is_done = (self.current_idx == self.n_lines)
    def pop_action(self):
        if(len(self.stack) <= 1):
            print("Tried to pop ROOT or stack was empty.")
            return
        self.record.append({ 
            'from': self.current_idx, 
            'to': self.stack[-1], 
            'type': 'pop' 
        })
        self.stack.pop()
    def undo(self):
        if(len(self.record) == 0):
            print("Unable to undo as there are currently no recorded actions.")
            return
        last_obj = self.record.pop()
        if(last_obj['type'] != 'discard'):
            self.stack.pop() 
            
        if(last_obj['type'] == 'pop'):
            self.stack.append(last_obj['to'])
        else:
            self.current_idx -= 1

        self.is_done = (self.current_idx == self.n_lines)
    
    def discard(self):
        if(self.is_done):
            return
        self.record.append({
            'from': self.current_idx,
            'to': self.current_idx,
            'type': 'discard'
        })
        self.current_idx += 1
        self.is_done = (self.current_idx == self.n_lines)
    def get_prompt(self):
        top_element = self.json_format[self.stack[-1]]['text'] if self.stack[-1] != -1 else "ROOT"
        buffered_element = self.json_format[self.current_idx]['text'] if self.current_idx < self.n_lines else "END"
        return f"ACTION #{len(self.record)} TOP: <{top_element}> / BUF: <{buffered_element}>"
    
    def __init__(self, file_path_ = None):

        if file_path_ is not None:
            self.file_path = file_path_
            self.record = []
            self.stack = [0] #root, can only accept subordinate relations 
            self.current_idx = 1

            self.wrapper = PDFWrapper(self.file_path)
            self.json_format = self.wrapper.get_json()
            self.n_lines = len(self.json_format)
            self.qt_image_list = [ImageQt(img) for img in self.wrapper.pil_image_list]
            print(f"Size of one qt image: {sys.getsizeof(self.qt_image_list[0])}, PIL image size: {sys.getsizeof(self.wrapper.pil_image_list[0])}")

    def get_qt_and_box(self, line_idx):
        # @param line_idx: the index of the line being rendered, in the annotations
        if(line_idx == -1):
            # $ROOT
            line_idx = 0

        line = self.json_format[line_idx]
        page_idx = line['page']
        idx_in_page = line['idx_in_page']
        res = ImageQt(self.wrapper.pil_image_lst[page_idx])

    def __getstate__(self):
        cpy = self.__dict__.copy()
        del cpy['qt_image_list']
        return cpy
    
    def __setstate__(self, d):
        self.__dict__ = d
        self.qt_image_list = [ImageQt(img) for img in self.wrapper.pil_image_list]



    def print_data(self):
        print(f"STACK: {self.stack[:3]}, currentidx = {self.current_idx}")
