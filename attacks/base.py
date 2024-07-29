"""
This repository is heavily inspired by HarmBench's repository, in structure and some code,
so much credit must go to them.
"""

import numpy as np
import pytorch
import os
import transformers
import fastchat

class Attack:

    def __init__(self, target_model, **kwargs):
        """
        :param target_mode: a dictionary specifying the target_model (kwargs to load_model_and_tokenizer)
        """
        raise NotImplementedError

    def attack(self, args):
        raise NotImplementedError

    def log(self, save_dir, model_name, attack_name):
        raise NotImplementedError
