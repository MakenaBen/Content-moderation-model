import torch
from transformers import LlamaSequenceClassification, LlamaTokenizer

def get_llama_model(model_name):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaSequenceClassification.from_pretrained(model_name)
    return tokenizer, model