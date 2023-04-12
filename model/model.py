import pandas as pd 
import torch
from torchmetrics import Accuracy
import transformers
import lightning.pytorch as pl
from tqdm import tqdm
import torch.nn as nn
from torch.nn import Linear, Dropout, ReLU, Embedding, CrossEntropyLoss
import math 

class ResumeParser(pl.LightningModule):
    def __init__(self, backend, args):
        super().__init__()
        self.backend = backend 
        self.classifier = nn.Sequential(
            Linear(in_features = self.backend.config.hidden_size + 4 * args['positional_dim'], out_features = args['hidden_dim']),
            Dropout(p = args['classifier_dropout']),
            ReLU(),
            Linear(in_features = args['hidden_dim'], out_features = args['hidden_dim']), #n_hidden = 1 hardcoded
            Dropout(p = args['classifier_dropout']),
            ReLU(),
            Linear(in_features = args['hidden_dim'], out_features = args['num_classes']),
            Dropout(p = args['classifier_dropout']),
        )  
        # self.pos_embeddings = Embedding(num_embeddings = 100, embedding_dim = args['positional_dim'])
        self.pos_embeddings = positionalencoding1d(args['positional_dim'], 101)
        # self.tokenizer = tokenizer
        self.metric = Accuracy(task = "multiclass", num_classes = n_classes)
        self.running_loss = None

        self.ce_loss = CrossEntropyLoss()
        
    def get_logits_and_loss(self, batch):
        inp_buf, inp_stk, pos, typ = batch 
        pos_emb = self.pos_embeddings[pos.cpu()].to(self.device) # B x 4 x D_pos
        pos_emb = pos_emb.reshape((-1, 4 * args['positional_dim'])) # concatenate all positional embeddings
        # print("pos_emb before shape: ", pos_emb.shape)
        # pos_emb = pos_emb.sum(dim = 1) # sum the positional embeddings (B x D_pos)
        # print("pos_emb after shape: ", pos_emb.shape)
        # print("inp ids shape:", inp_buf['input_ids'].shape, "inp_buf type: ", type(inp_buf))
        emb_buf = self.backend(**inp_buf)['pooler_output'] # B x D_backend
        emb_stk = self.backend(**inp_stk)['pooler_output'] # B x D_backend
        # print("Pos embedding shape: ", pos_emb.shape, ", emb_buf shape: ", emb_buf.shape, " emb_stk shape: ", emb_stk.shape)
        classifier_inp = torch.cat((emb_buf + emb_stk, pos_emb), 1) # B x (D_backend + D_pos)
        logits = self.classifier(classifier_inp)
        loss = self.ce_loss(logits, typ)
        return logits, loss

    def training_step(self, batch, batch_idx):
        _, loss = self.get_logits_and_loss(batch)
        
        if self.running_loss == None:
            self.running_loss = loss
        self.running_loss = 0.95 * self.running_loss + 0.05 * loss

        return loss

    def validation_step(self, batch, batch_idx):
        _, _, _, typ = batch 
        logits, loss = self.get_logits_and_loss(batch)
        preds = torch.argmax(logits, dim = 1)
        # print("logits shape:", logits.shape, ", preds.shape: ", preds.shape)
        
        self.log("Validation Accuracy", self.metric(preds, typ))
        self.log("Validation Loss", loss)
        if self.running_loss is not None:
            self.log("Training Loss", self.running_loss)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer
    def __init__(self, model, tokenizer, n_classes):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.metric = Accuracy(task = "multiclass", num_classes = n_classes)
        self.running_loss = None
        
    def training_step(self, batch, batch_idx):
        buf, stk, typ = batch 
        strs = [[a, b] for a, b in zip(buf, stk)]
        inputs = self.tokenizer(strs, return_tensors="pt", padding=True).to(self.device)
        output = self.model(**inputs, labels = typ)
        
        if self.running_loss == None:
            self.running_loss = output.loss
        self.running_loss = 0.95 * self.running_loss + 0.05 * output.loss

        return output.loss

    def validation_step(self, batch, batch_idx):
        
        buf, stk, typ = batch 
        strs = [[a, b] for a, b in zip(buf, stk)]
        inputs = self.tokenizer(strs, return_tensors="pt", padding=True).to(self.device)
        output = self.model(**inputs, labels = typ)
        logits = output.logits 
        preds = logits.argmax(dim = -1)
        
        self.log("accuracy", self.metric(preds, typ))
        self.log("val_loss", output.loss)
        if self.running_loss is not None:
            self.log("train_loss", self.running_loss)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer