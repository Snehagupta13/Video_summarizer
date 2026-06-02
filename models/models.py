# --------------------------- model/model.py ---------------------------
# Placeholder: You can replace this with LayoutLM or Donut
import torch.nn as nn
import torch

class DummyOCRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 2)  # dummy output

    def forward(self, x):
        return self.linear(x)