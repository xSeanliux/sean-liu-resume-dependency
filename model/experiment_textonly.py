#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd
import torch
from torchmetrics import Accuracy
import lightning.pytorch as pl
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from lightning.pytorch import loggers as pl_loggers
from data import ResumeDataset, ResumePromptDataset
from model_t5 import T5FineTuner
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import (
  AutoTokenizer
)
from utils import label_idx, idx_label, all_labels, n_classes



import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
  model_name = "t5-large"
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  tokenizer_args = {
      'padding': True,
      'return_tensors': 'pt',
  }

  output_dir = "~/sp23/sean-liu-resume-dependency/model/output"
  
  checkpoint_callback = ModelCheckpoint(dirpath=output_dir, save_top_k=2, monitor="Validation Loss")

  args_dict = dict(
      output_dir=output_dir,
      input_json_path = "../data/json/data_fine.json",
      tokenizer_args=tokenizer_args,
      model_name_or_path=model_name,
      tokenizer_name_or_path=model_name,
      max_seq_length=512,
      learning_rate=3e-5,
      weight_decay=0.1,
      adam_epsilon=1e-8,
      warmup_steps=0,
      num_train_epochs=3,
      train_batch_size=8,
      eval_batch_size=8,
      gradient_accumulation_steps=2,
      early_stop_callback=False,
      fp_16=False, # Setting this to true results in NaN values for the loss - this is widely documented for the T5 family unfortunately :(
      precision=32, # if you want to enable 16-bit training then install apex and set this to true
      opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
      max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
      seed=42
  )

  args = argparse.Namespace(**args_dict)
  print(args_dict)


  tb_logger = pl_loggers.TensorBoardLogger(save_dir="/home/zxliu2/sp23/sean-liu-resume-dependency/model/")

  train_params = dict(
      accumulate_grad_batches=args.gradient_accumulation_steps,
      max_epochs=args.num_train_epochs,
      precision= 16 if args.fp_16 else 32,
      accelerator="gpu",
      devices = [0],
      gradient_clip_val=args.max_grad_norm,
      val_check_interval = 0.5, # validate every half epoch
      detect_anomaly=True,
      logger=tb_logger,
      callbacks=[checkpoint_callback]
  )
  model = T5FineTuner(args)
  trainer = pl.Trainer(**train_params)
  tuner = Tuner(trainer)
  trainer.fit(model)
