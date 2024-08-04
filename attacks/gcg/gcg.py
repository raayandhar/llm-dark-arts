import transformers
import numpy as np
import torch

from typing import List
from dataclasses import dataclass

from ..base import Attack
from .utils import GCGResult
from .utils import AttackBuffer

class GCG(Attack):

    def __init__(self, target_model, config):
    """
    :param target_model: a dictionary specifying the target_model (kwargs to load_model_and_tokenizer)
    :param config: a dictionary specifying the different options for the GCG attack; expects dictionary
    Our configs params are as follows:
    :config_param num_steps: number of optimization steps to use; expects int
    :config_param search_width: the number of candidates sequences we are sampling at each step; expects int
    :config_param batch_size: how many of the search_width candidate sequences are evaluated at a time in a single GCG iteration; expects int
    :config_param topk: number of token substitutions to consider in a single GCG iteration; expects int
    :config_param buffer_size: size of attack buffer; expects int
    :config_param n_replace

    :config_param adv_init: the initial adversarial string; expects string
    :config_param allow_non_ascii: are we going to allow non ascii tokens or not; expects boolean
    :config_param filter_ids: if you want to retain candidate sequences that are the same after tokenization and retokenization; expects boolean
    :config_param asbt: add a space before the target string; expects string
    :config_param seed: default 42, the answer to the life, the universe, and everything
    :config_param verbose: be verbose when running the attack

    :config_param use_mellowmax: use the mellowmax loss function
    :config_param mellowmax_alpha: value of alpha in mellowmax loss function
    """

    # Core optimization and performance hyperparameters
    self.num_steps = config.get("num_steps", 250)
    self.search_width = config.get("search_width", 512)
    self.batch_size = config.get("batch_size", 0)
    self.topk = config.get("topk", 256)
    self.buffer_size = config.get("buffer_size", 0)
    self.n_replace = config.get("n_replace", 1)

    # Not so important
    self.adv_init = config.get("adv_init", "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
    self.allow_non_ascii = config.get("allow_non_ascii", False)
    self.filter_ids = config.get("filter_ids", True)
    # asbt = add_space_before_target
    self.asbt = config.get("abst", False)
    self.seed = config.get("seed", 42)
    self.verbose = config.get("verbose", False)

    # Research choices
    self.use_mellowmax = config.get("use_mellowmax", False)
    self.mellowmax_alpha = config.get("mellowmax_alpha", 0)
    # make sure to add other options for loss functions
    # TODO

    


    def attack(self, args):
        pass
