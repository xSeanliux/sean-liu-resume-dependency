# Resume Dependency Parser 

## Overview
This project comprises of two main parts: 
* Annotator/Viewer: they're found in the root folder, and lets you view/annotate PDF files. The backend is mainly written using [pdfminer.six](https://pypi.org/project/pdfminer.six/) to extract information from PDFs, and the GUI uses [PyQt6](https://pypi.org/project/PyQt6/).
* ML model: found in `model/`, this is in charge of the ML pipeline: data aggregation, model initialization & training, and evaluation. Base models (when they're used) are from [HuggingFace](https://huggingface.co/) and faster training is enabled by [PyTorch Lightning](https://www.pytorchlightning.ai/index.html).
## Setup

### Environment

This was developed using Python version 3.10 and 3.11. Also provided is a `requirements.txt` file - please run 
```conda create --name resume_dependency --file ./requirements.txt```
. An ARM Macbook Pro & UNIX machine were used in developement. In addition, if you want to finetune the model or perform evaluation, please make sure you have the correct `.ckpt` file under `model/`. 

### Folder structure 

```
.
└── sean-liu-resume-dependency/
    ├── data/
    │   ├── json/
    │   │   └── ...
    │   ├── pdf/
    │   │   └── your PDF files here!
    │   ├── pkl/
    │   │   └── ...
    │   └── ...
    ├── model/
    │   ├── data.py
    │   ├── experiment.ipynb
    │   ├── model.py
    │   ├── utils.py
    │   └── your .ckpt checkpoint file
    ├── report/
    │   ├── main.tex
    │   └── main.pdf
    ├── pdf_wrapper.py
    ├── annotation_object.py
    ├── frontend.ipynb
    ├── extract_pdf.ipynb
    ├── pkl_to_json.ipynb
    ├── tagger_gui.py
    ├── README.md
    └── requirements.txt
```

note: ASCII tree was drawn using [this](https://tree.nathanfriend.io/) software.

## Functional Design (Usage)


### Annotation / Viewing: 
In the root folder, run 
```python3 tagger_gui.py```
. You will be prompted to enter the name of a PDF file that's present in `data/pdf/`, and it'll start running.

#### Interface
There are two pages shown - the left is the "stack view" which defaults to showing the page that the current stack element is on; on the right is the "buffer view" which shows the page that the current buffer element is on by default. The relevant (stack/buffer) element is shown with a blue box around it. 
* Line colors: every annotated, non-discarded line has a color indicating its depth
* Green lines indicate a merge relation between elements, while
* orange lines indicate subordinate relations. 
#### Controls: 
For annotation (the key to press is in square brackets): 
* [s]ubordinate (defines a subordinate relation)
* [m]erge (defines a merge relation)
* [p]op (pops current stack element from stack)
* [u]undo (undos last operation, note that you cannot undo undos)
* [d]iscard (undos current buffer element, useful for things like page numbers)
* [q]uit (quits and saves program)
And you may navigate using the arrow keys. 

### Training
#### Data preprocessing 
After annotation, a corresponding `.pkl` file will be generated with the same file name (but different extension) inside `data/pkl`. Run `pkl_to_json.ipynb` to aggregate all the PDF(s) into a training JSON file. You may need to change some values such as file/folder names in there if you would like to have different JSON files present (for example, for different coarsenesses). 
#### Training 
You will find most of the configuration and training scripts inside `model/experiment.ipynb`, while the model itself and the LightningModule is defined inside `model/model.py`. Run `model/experiment.ipynb` to start training. 

**IMPORTANT**: I couldn't get multicore training to work, so in the trainer definition towards the end of the file,
```python
trainer = pl.Trainer(accelerator="gpu", devices=[2], val_check_interval = 0.5)
```
, please change `devices` to the number of your own GPU that you'd like to use, for example, to `devices=[0]`. You could alternatively just use the CPU to train. You may find more information in the PyTorch Lightning [docs](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html).

#### Evaluation 

If you're not using a pretrained checkpoint, you can get the training weights in `model/lightning_logs/<run name>/` after running training. Move the desired `.ckpt` checkpoint file to `model/` and follow the instructions in `frontend.ipynb`. 

To view the parsing results, you can run the viewer using `python3 tagger_gui.py`. 
## Demo video

TODO

## Algorithmic Design 
We make some assumptions about the nature of the PDFs that we extract: 
1. The text of the resumes can be extracted without the use of OCR software - that is, they were computer-born (for example, generated using typesetting software). 
2. The resumes themeselves are single-column. 
3. There is an underlying hierarchical tree structure underlying the resume, with individual items (such as publications) being subordinated under headers and sub-headers, exempli gratia, `$ROOT -> Publications -> Books -> To Be Published -> Harry Potter and Goblet of Azkaban`. In addition, that the lines are *some* preorder ordering of this underlying tree (that is, projective structure, etc.)
### PDF Extraction 

We first extract all the lines of said PDF and sort by the $y$-position

## Issues and Future Work

## Change log