import numpy as np
import pytorch
import os
import transformers
import fastchat

class Attack:

    def __init__(self, target_model, config, **kwargs):
        """
        :param target_mode: a dictionary specifying the target_model (kwargs to load_model_and_tokenizer)
        """
        raise NotImplementedError

    def attack(self, args):
        raise NotImplementedError

    def log(self, save_dir, model_name, attack_name):
        raise NotImplementedError

"""
NOTES:
- reference harmbench, nano_gcg and the struq gcg code
- recall that you need to add in these specialm delimiters and stuff -- it's not going to be super simple,
  so reference the struq gcg
- you want to keep the loss function options clear, understand where the loss comes from
- understand each repositories approach to GCG code pretty well first before you tackle it seriously, map this out
"""
