import sys
import os
import argparse 
from annotation_object import AnnotationObject, serialize, deserialize
import atexit

file_name = None
anno = None

def run_annotation(file_path):
    print("Running annotation.")
    anno = AnnotationObject()
    if os.path.exists(file_path + ".pkl"):
        print(f"{file_path}.pkl exists. Loading.")
        anno = deserialize(file_path)
        print(f"Finished loading. Starting from index {anno.current_idx}")
    else:
        print("Loading from PDF file path.")
        anno = AnnotationObject(file_path)

    anno.print_data()

    while(not anno.is_done):
        print(anno.get_prompt())
        res = input("[s]ubordinate/0; [m]erge/1; [p]op/2; [u]ndo/3; [q]uit ")
        if(len(res) > 1):
            continue
        if(res == 's' or res == '0'):
            print("SUBORDINATE")
            anno.subordinate_action()
        if(res == 'm' or res == '1'):
            print("MERGE")
            anno.merge_action()
        if(res == 'p' or res == '2'):
            print("POP")
            anno.pop_action()
        if(res == 'u' or res == '3'):
            print("UNDO")
            anno.undo()
        if(res == 'q'):
            break
        


    anno.serialize()

    return
    # raise NotImplementedError



if __name__ == "__main__":
    while(True):
        file_name, file_path = get_file_name()
        print(f"File_path = {file_path}")
        run_annotation(file_path)
        while(True):
            res = input(f"Parsing for file [{file_path}] has finished. Would you like to continue (y/n)? >")
            if res[0] == 'y':
                print("Continuing. Please enter a new filename:")
                file_name = get_file_name()
                break
            elif res[0] == 'n':
                sys.exit(0)

        
