
from pdf_wrapper import PDFWrapper
import pickle as pkl
from PIL.ImageQt import ImageQt
import os

def serialize(anno_object):
    print(f"Serializing... {anno_object}")
    if anno_object is not None:
        pkl_file_name = anno_object.file_path.replace("pdf", "pkl")
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
    
    
    def subordinate_action(self):
        if(self.is_done):
            return 0
        
        self.record.append({ 
            'from': self.current_idx, 
            'to': self.stack[-1], 
            'type': 'subordinate' 
        })

        self.depth[self.current_idx] = 1 if self.stack[-1] == -1 else self.depth[self.stack[-1]] + 1
        # root has depth 0

        self.stack.append(self.current_idx)
        self.current_idx += 1

        self.is_done = (self.current_idx == self.n_lines)
        return 0
    
    def merge_action(self):
        if(self.is_done):
            return 0
        if(self.stack[-1] == -1):
            return 1
        
        self.record.append({ 
            'from': self.current_idx, 
            'to': self.stack[-1], 
            'type': 'merge' 
        })
        self.depth[self.current_idx] = 0 if self.stack[-1] == -1 else self.depth[self.stack[-1]]
        self.stack.pop() # representative element is the last line of the whole entry
        self.stack.append(self.current_idx)
        self.current_idx += 1

        self.is_done = (self.current_idx == self.n_lines)
        return 0

    def pop_action(self):
        if(len(self.stack) <= 1):
            print("Tried to pop ROOT or stack was empty.")
            return 1
        self.record.append({ 
            'from': self.current_idx, 
            'to': self.stack[-1], 
            'type': 'pop' 
        })
        self.stack.pop()
        return 0
    
    def discard(self):
        if(self.is_done):
            return 0
        self.record.append({
            'from': self.current_idx,
            'to': self.stack[-1],
            'type': 'discard'
        })
        self.depth[self.current_idx] = -1
        self.current_idx += 1
        self.is_done = (self.current_idx == self.n_lines)
        return 0
    
    def undo(self):
        if(len(self.record) == 0):
            print("Unable to undo as there are currently no recorded actions.")
            return 1
        last_obj = self.record.pop()
        operation_type = last_obj['type']
        if(operation_type == 'pop'):
            self.stack.append(last_obj['to'])
        else:
            self.current_idx -= 1
            self.depth[last_obj['from']] = -1

        if(operation_type == 'merge'):
            
            self.stack.pop()
            self.stack.append(last_obj['to'])
        elif(operation_type == 'subordinate'):
            self.stack.pop()
            
        self.is_done = (self.current_idx == self.n_lines)
        return 0

    def get_prompt(self):
        top_element = self.json_format[self.stack[-1]]['text'] if self.stack[-1] != -1 else "ROOT"
        buffered_element = self.json_format[self.current_idx]['text'] if self.current_idx < self.n_lines else "END"
        return f"ACTION #{len(self.record)} TOP: <{top_element}> / BUF: <{buffered_element}>"
    
    def __init__(self, file_path_ = None):
        self.file_path = None
        self.record = []
        self.depth = []
        self.wrapper = None
        self.json_format = None
        self.is_done = False
        # record contains objects which look like 
        # {
        #     'from': (number, index)
        #     'to': (number, index)
        #     type: string, one of ('subordinate', 'merge', 'pop')
        # }
        self.current_idx = 0 # current_idx: the current object being processed
        self.n_lines = 0
        self.n_pages = 0
        self.stack = [] # stack is the stack being used, records the indices only, -1 is root
        self.qt_image_list = None
        
        if file_path_ is not None:
            self.file_path = file_path_
            self.record = []
            self.stack = [-1] #root, can only accept subordinate relations 
            self.current_idx = 0

            self.wrapper = PDFWrapper(self.file_path)
            self.json_format = self.wrapper.get_json()
            
            self.n_lines = len(self.json_format)
            self.n_pages = len(self.wrapper.lines)
            print("Document has ", self.n_pages, "pages")
            self.depth = [-1] * self.n_lines
            self.qt_image_list = [ImageQt(img) for img in self.wrapper.pil_image_list]
            # print(f"Size of one qt image: {sys.getsizeof(self.qt_image_list[0])}, PIL image size: {sys.getsizeof(self.wrapper.pil_image_list[0])}")
            print(f"Length of dep: {len(self.depth)}")
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
