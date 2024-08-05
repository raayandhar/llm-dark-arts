import functools
import gc
import inspect
import torch
from torch import Tensor

def get_not_allowed_ids(tokenizer, device, allow_non_ascii, not_allowed_tokens):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    not_allowed_tokens = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            not_allowed_tokens.append(i)

    if tokenizer.bos_token_id is not None:
        nonascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        nonascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        nonascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        nonascii_toks.append(tokenizer.unk_token_id)

    # ADD other tokens that we cannot sample (delimiters!)

    return torch.tensor(nonascii_toks, device=device)
