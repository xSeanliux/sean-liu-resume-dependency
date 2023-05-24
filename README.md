# Resume Dependency Parser 

## Overview
This project comprises of two main parts: 
* Annotator/Viewer: they're found in the root folder, and lets you view/annotate PDF files. 
* ML model: found in `model/`, this is in charge of the ML pipeline: data aggregation, model initialization & training, and evaluation. 
## Setup

### Environment

This was developed using python version 3.10 and 3.11. Also provided is a `requirements.txt` file - please run 
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

[software to draw the ASCII tree](https://tree.nathanfriend.io/)

## Functional Design (Usage)

### Annotation / Viewing: 
In the root 

## Demo video

## Algorithmic Design 

## Issues and Future Work

## Change log