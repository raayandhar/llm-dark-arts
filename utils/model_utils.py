import transformers
import numpy as np
import torch

def load_model_and_tokenizer(
        model_name_or_path,
        dtype='auto',
        device_map='auto',
        
