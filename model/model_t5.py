import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, ReLU, Embedding, CrossEntropyLoss
import math 
from torchmetrics import Accuracy
import transformers
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from tqdm import tqdm
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from data import ResumeDataset, ResumePromptDataset
from utils import all_labels

class T5FineTuner(pl.LightningModule):
  def __init__(self, hparams):
    super(T5FineTuner, self).__init__()
    self.hparams.update(vars(hparams))
    


    self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
    if hasattr(hparams, "tokenizer_args"):
      self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name_or_path, **(hparams.tokenizer_args))
    else:
      self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name_or_path, padding=True, return_tensors = 'pt')
   
    action_ids = self.tokenizer(all_labels, max_length=2)

    self.action_to_token_id = torch.LongTensor(action_ids["input_ids"])[:, 0]
    self.accuracy = Accuracy(task="multiclass", num_classes = self.tokenizer.vocab_size) 
    self.build_datasets() 
    self.save_hyperparameters()

  def build_datasets(self):
    input_json_path = self.hparams.input_json_path if hasattr(self.hparams, "input_json_path") else "../data/json/data_fine.json"
    resume_dataset = ResumePromptDataset(self.tokenizer, input_json_path)
    total_count = len(resume_dataset)
    train_count = int(0.85 * total_count)
    valid_count = int(0.1 * total_count)
    test_count = total_count - train_count - valid_count
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        resume_dataset, (train_count, valid_count, test_count)
    )
    self.train_dataset = train_dataset
    self.valid_dataset = valid_dataset
    self.test_dataset  = test_dataset

  def is_logger(self):
    return self.trainer.proc_rank <= 0
  
  def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
  ):
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,
    )

  def _step(self, batch):
    labels = batch["target_ids"]
    labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

    outputs = self(
        input_ids=batch["source_ids"],
        attention_mask=batch["source_mask"],
        labels=labels,
        decoder_attention_mask=batch['target_mask']
    )

    # logits : B x L x V
    return outputs["logits"][:, 0, :], outputs["loss"]

  def get_logits(self, batch):
    labels = batch["target_ids"]
    labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

    outputs = self(
        input_ids=batch["source_ids"],
        attention_mask=batch["source_mask"],
        labels=labels,
        decoder_attention_mask=batch['target_mask']
    )

    # logits : B x L x V
    print(f'logit shape: {outputs["logits"].shape}')
    return outputs["logits"][:, 0, self.action_to_token_id]

  def training_step(self, batch, batch_idx):
    _, loss = self._step(batch)

    tensorboard_logs = {"train_loss": loss}
    self.log("Training Loss", loss)
    return {"loss": loss, "log": tensorboard_logs}
  
  def validation_step(self, batch, batch_idx):
    logits, loss = self._step(batch)
    labels = batch["target_ids"]
    # print(f"logit shape: {logits.shape}, labels shape : {labels.shape}, labels: {labels}, vocab size: {len(tokenizer)}")
    acc = self.accuracy(logits[:, :self.tokenizer.vocab_size], labels[:, 0]) # logits has 28 extra labels for whatever reason; labels: B x L 
    self.log("Validation Loss", loss, on_epoch=True) # accumulate loss over the whole validation epoch
    self.log("Validation Accuracy", acc, on_epoch=True) # accumulate accuracy over the whole validation epoch
    return {"val_loss": loss}
  
  def test_step(self, batch, batch_idx):
    logits, loss = self._step(batch)
    labels = batch["target_ids"]
    # print(f"logit shape: {logits.shape}, labels shape : {labels.shape}, labels: {labels}, vocab size: {len(tokenizer)}")
    acc = self.accuracy(logits[:, :self.tokenizer.vocab_size], labels[:, 0]) # logits has 28 extra labels for whatever reason; labels: B x L 
    self.log("Test Loss", loss, on_epoch=True) # accumulate loss over the whole validation epoch
    self.log("Test Accuracy", acc, on_epoch=True) # accumulate accuracy over the whole validation epoch
    return {"val_loss": loss}
  
  def configure_optimizers(self):
    "Prepare optimizer and schedule (linear warmup and decay)"

    model = self.model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.hparams.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
    self.opt = optimizer
    return [optimizer]
  
  # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
  #   optimizer.step()
  #   optimizer.zero_grad()
  #   self.lr_scheduler.step()
  
  def get_tqdm_dict(self):
    tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

    return tqdm_dict

  def train_dataloader(self):
    dataloader = DataLoader(self.train_dataset, drop_last=True, shuffle=True, num_workers=4)
    t_total = (
        (len(dataloader.dataset) // (self.hparams.train_batch_size))
        // self.hparams.gradient_accumulation_steps
        * float(self.hparams.num_train_epochs)
    )
    scheduler = get_linear_schedule_with_warmup(
        self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
    )
    self.lr_scheduler = scheduler
    return dataloader

  def val_dataloader(self):
    return DataLoader(self.valid_dataset, num_workers=4)
  
  def test_dataloader(self):
    return DataLoader(self.test_dataset, num_workers=4)

