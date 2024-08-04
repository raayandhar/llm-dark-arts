import transformers
import numpy as np
import torch

from typing import List
from dataclasses import dataclass

@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]

class AttackBuffer:
    def __init__(self, size:int):
        # buffer elems are (loss: float, optim_ids: Tensor)
        self.buffer = []
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
            return

        self.buffer[-1] = (loss, optim_ids)
        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_min_loss(self) -> float:
        return self.buffer[0][0]

    def get_max_loss(self) -> float:
        return self.buffer[-1][0]

    def print_buffer(self, tokenizer):
        print("Buffer")
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            print(f"loss: {loss}" + f" | optim string: {optim_str}")
        print ("End buffer")

class GCGUtil:
    
