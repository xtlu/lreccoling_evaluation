
from transformers import BertTokenizer
import torch
import torchtext
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch.nn as nn
from transformers import BertModel


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained('bert-base-uncased',output_attentions=True)




class BERTSentiment(nn.Module):
    def __init__(self,
                 bert,
                 output_dim):
        
        super().__init__()
        
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.out = nn.Linear(embedding_dim, output_dim)
    def forward(self, input_ids):
        #embedded = [batch size, emb dim]
        embedded =  self.bert(input_ids  ,attention_mask=self.attention_mask )[1]
        #output = [batch size, out dim]
        output = self.out(embedded)
        
        return output
    
OUTPUT_DIM = 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = BERTSentiment(bert, OUTPUT_DIM).to(device)    
# Pretrained model BertIBDM.m
model.load_state_dict(torch.load('BertIBDM.m'))



def get_iptsandattmaks(text):
    token = my_token(text)
    input_ids = token["input_ids"].to(device)
    attention_mask = token["attention_mask"].to(device)
    return input_ids,attention_mask


from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso


def compute_bert_outputs(model_bert, embedding_output, attention_mask=None, head_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    extended_attention_mask = extended_attention_mask.to(dtype=next(model_bert.parameters()).dtype) # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(model_bert.config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        head_mask = head_mask.to(dtype=next(model_bert.parameters()).dtype) # switch to fload if need + fp16 compatibility
    else:
        head_mask = [None] * model_bert.config.num_hidden_layers

    encoder_outputs = model_bert.encoder(embedding_output,
                                         extended_attention_mask,
                                         head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = model_bert.pooler(sequence_output)
    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)    

class BertModelWrapper(nn.Module):
    
    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, embeddings):        
        output = compute_bert_outputs(self.model.bert, embeddings,attention_mask=self.model.attention_mask)[1]
        output = self.model.out(output)
        return output
  

from captum.attr import Lime, LimeBase,IntegratedGradients,Saliency,DeepLift,InputXGradient


def get_attention(input_ids,attention_mask):
    model = BERTSentiment(bert, OUTPUT_DIM).to(device)    
    model.load_state_dict(torch.load('BertIBDM.m'))
    with torch.no_grad():
        model.attention_mask=attention_mask
        t= model.bert(input_ids)
        all_attention = torch.stack(t[2])
        print(all_attention.shape)
        all_attention = all_attention.squeeze()

        all_attention_sum = torch.sum(all_attention,dim=0)
        all_attention_sum = torch.sum(all_attention_sum,dim=0)
        all_attention_sum = torch.sum(all_attention_sum,dim=0)

        last_attention = all_attention[-1]
        last_attention = torch.sum(last_attention,dim=0)
        last_attention = torch.sum(last_attention,dim=0)

        return all_attention_sum.squeeze(),last_attention.squeeze()

def logits(input_ids,attention_mask):
    model = BERTSentiment(bert, OUTPUT_DIM).to(device)    
    model.load_state_dict(torch.load('BertIBDM.m'))  
    with torch.no_grad():
        model.attention_mask=attention_mask
        print(model(input_ids))
        
        
def get_ig(input_ids,attention_mask,target):
    model = BERTSentiment(bert, OUTPUT_DIM).to(device)    
    model.load_state_dict(torch.load('BertIBDM.m'))
    model.attention_mask=attention_mask
    wrap = BertModelWrapper(model)
    input_embedding = wrap.model.bert.embeddings(input_ids)
    wrap.eval()
    wrap.zero_grad()
    ig = IntegratedGradients(wrap)

    attributions_ig, delta = ig.attribute(input_embedding, n_steps=100, return_convergence_delta=True,target=target)
    mapig= torch.sum(attributions_ig,dim=2)
    mapig=mapig.squeeze()
    return mapig


def get_lime(input_ids,attention_mask,target):
    model = BERTSentiment(bert, OUTPUT_DIM).to(device)    
    model.load_state_dict(torch.load('BertIBDM.m'))
    model.attention_mask=attention_mask
    wrap = BertModelWrapper(model)
    input_embedding = wrap.model.bert.embeddings(input_ids)
    exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)
    lr_lime = Lime(
        wrap, 
        interpretable_model=SkLearnLinearRegression(),  # build-in wrapped sklearn Linear Regression
        similarity_func=exp_eucl_distance
    )

    attrs = lr_lime.attribute(
        input_embedding,
        target=target,
        n_samples=40,
        perturbations_per_eval=16,
        show_progress=True
    )

    attrs = torch.sum(attrs,dim=2).squeeze()
    return attrs

def get_vg(input_ids,attention_mask,target):
    
    model = BERTSentiment(bert, OUTPUT_DIM).to(device)    
    model.load_state_dict(torch.load('BertIBDM.m'))
    model.attention_mask=attention_mask
    wrap = BertModelWrapper(model)
    input_embedding = wrap.model.bert.embeddings(input_ids)
    
    saliency = Saliency(wrap)

    attrs = saliency.attribute(input_embedding, target=target)
    attrs = torch.sum(attrs,dim=2).squeeze()
    return attrs
   

def get_deeplift(input_ids,attention_mask,target):
    
    model = BERTSentiment(bert, OUTPUT_DIM).to(device)    
    model.load_state_dict(torch.load('BertIBDM.m'))
    model.attention_mask=attention_mask
    wrap = BertModelWrapper(model)
    input_embedding = wrap.model.bert.embeddings(input_ids)
    
    dl = DeepLift(wrap)

    attrs = dl.attribute(input_embedding, target=target)
    attrs = torch.sum(attrs,dim=2).squeeze()
    return attrs

def get_ixg(input_ids,attention_mask,target):
    
    model = BERTSentiment(bert, OUTPUT_DIM).to(device)    
    model.load_state_dict(torch.load('BertIBDM.m'))
    model.attention_mask=attention_mask
    wrap = BertModelWrapper(model)
    input_embedding = wrap.model.bert.embeddings(input_ids)
    ixg = InputXGradient(wrap)
    attrs = ixg.attribute(input_embedding, target=target)
    attrs = torch.sum(attrs,dim=2).squeeze()
    return attrs






