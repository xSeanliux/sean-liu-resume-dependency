import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, ReLU, Embedding, CrossEntropyLoss
import math 
from torchmetrics import Accuracy
import transformers
import lightning.pytorch as pl
from tqdm import tqdm





class ResumeParser(pl.LightningModule):
    
    def positionalencoding1d(self, d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe
    
    def __init__(self, backend, args):
        super().__init__()
        self.backend = backend 
        self.args = args

        self.input_dim = 4 * args['positional_dim'] + 4 # 4 for the style
        if args['use_llm']:
            args += self.backend.config.hidden_size
        
        self.classifier = nn.Sequential(
            Linear(in_features = self.input_dim, out_features = args['hidden_dim']),
            Dropout(p = args['classifier_dropout']),
            ReLU(),
            Linear(in_features = args['hidden_dim'], out_features = args['num_classes']), #n_hidden = 1 hardcoded
            Dropout(p = args['classifier_dropout']),
        )  
        # self.pos_embeddings = Embedding(num_embeddings = 100, embedding_dim = args['positional_dim'])
        self.pos_embeddings = self.positionalencoding1d(args['positional_dim'], 101).to('cuda:2') #is a hacky workaround
        self.register_buffer('bbox_pos_embeddings', self.pos_embeddings, persistent=False)
        print("Device: ", self.device)
        # self.tokenizer = tokenizer
        self.metric = Accuracy(task = "multiclass", num_classes = self.args['num_classes'])
        self.running_loss = None

        self.ce_loss = CrossEntropyLoss()

    def get_logits(self, batch):
        inp_buf, inp_stk, pos, sty, _ = batch 
        pos_emb = self.pos_embeddings[pos] # B x 6 x D_pos
        pos_emb = pos_emb.reshape((-1, 4 * self.args['positional_dim'])) # concatenate all positional embeddings
        classifier_inp = None
        if self.args['use_llm']:
            emb_buf = self.backend(**inp_buf)['pooler_output'] # B x D_backend
            emb_stk = self.backend(**inp_stk)['pooler_output'] # B x D_backend
            classifier_inp = torch.cat((emb_buf + emb_stk, sty, pos_emb), 1) # B x (D_backend + D_pos)
        else:
            classifier_inp = torch.cat((sty, pos_emb), 1) 

        
        # print("pos_emb before shape: ", pos_emb.shape)
        # pos_emb = pos_emb.sum(dim = 1) # sum the positional embeddings (B x D_pos)
        # print("pos_emb after shape: ", pos_emb.shape)
        # print("inp ids shape:", inp_buf['input_ids'].shape, "inp_buf type: ", type(inp_buf))
        
        # print("Pos embedding shape: ", pos_emb.shape, ", emb_buf shape: ", emb_buf.shape, " emb_stk shape: ", emb_stk.shape)
        
        logits = self.classifier(classifier_inp)
        return logits
        
    def get_logits_and_loss(self, batch):
        _, _, _, _, typ = batch
        logits = self.get_logits(batch)
        loss = self.ce_loss(logits, typ)
        return logits, loss

    def training_step(self, batch, batch_idx):
        _, loss = self.get_logits_and_loss(batch)
        
        if self.running_loss == None:
            self.running_loss = loss
        self.running_loss = 0.95 * self.running_loss + 0.05 * loss

        return loss

    def validation_step(self, batch, batch_idx):
        typ = batch[-1]
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