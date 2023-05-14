
from pdf_wrapper import PDFWrapper
import pickle as pkl
from PIL.ImageQt import ImageQt
import os

'''
This file is the annotation object
A wrapper for the pdf wrapper to support the annotation of files (shift-based)
The main class is AnnotationObject. 
'''

'''
This function serialises the annotation object into a .pkl file, with the filename being the same as
the original pdf file just with a different extension.
@param anno_object: the AnnotationObject to serialise
'''
def serialize(anno_object):
    print(f"Serializing... {anno_object}")
    if anno_object is not None:
        pkl_file_name = anno_object.file_path.replace("pdf", "pkl")
        with open(pkl_file_name, "wb") as handle:
            pkl.dump(anno_object, handle)
            print(f"Dumped to {pkl_file_name}.")
    else:
        print("Error! anno object is None!")

'''
Deserialises an AnnotationObject from a given .pkl path.
'''
def deserialize(pkl_file_path):
    print(f"{pkl_file_path} exists: {os.path.exists(pkl_file_path)}, stats = {os.stat(pkl_file_path)}")
    assert len(pkl_file_path) > 4 and pkl_file_path[-4:] == ".pkl"
    with open(pkl_file_path, "rb") as handle:
        anno = pkl.load(handle)
        return anno


class AnnotationObject: 


    '''
    Implements the SUBORDINATE action.
    '''
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
    
    '''
    Implements the MERGE action.
    '''
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

    '''
    Implements the POP action.
    '''
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
    
    '''
    Implements the DISCARD action.
    '''
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
    
    '''
    Implements the UNDO action. 

    Undos the last operation. Note that this operation is not stored, so you cannot undo undos. 
    This function is for human use only; machines do not make mistakes (only happy errors) and thus
    do not need UNDOs. 
    '''
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

    '''
    Returns a prompt hint; was more useful when we had a text-only interface, but now is more decorational than practical.
    '''
    def get_prompt(self):
        top_element = self.json_format[self.stack[-1]]['text'] if self.stack[-1] != -1 else "ROOT"
        buffered_element = self.json_format[self.current_idx]['text'] if self.current_idx < self.n_lines else "END"
        return f"ACTION #{len(self.record)} TOP: <{top_element}> / BUF: <{buffered_element}>"
    

    '''
    Initialiser of the class which takes in a path to a PDF file.
    
    @param file_path_: path to .pdf file. 
    '''
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
            print(f"Length of dep: {len(self.depth)}")


    '''
    Custom getstate without qt_image_list (because the size of that list is HUGE) for serialise()
    '''
    def __getstate__(self):
        cpy = self.__dict__.copy()
        del cpy['qt_image_list']
        return cpy
    
    '''
    Corresponding setter for the above
    '''
    def __setstate__(self, d):
        self.__dict__ = d
        self.qt_image_list = [ImageQt(img) for img in self.wrapper.pil_image_list]