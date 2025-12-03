import torch
import torch.nn as nn
from transformer_lens import HookedTransformer, utils

class PredictSafety(nn.Module):
 def __init__(self):
  super().__init__()
  self.network = nn.Sequential(
   nn.AdaptiveAvgPool2d((1, 100)),
   nn.Linear(100, 300),
   nn.LeakyReLU(),
   nn.Linear(300, 600),
   nn.LeakyReLU(),
   nn.Linear(600, 300),
   nn.LeakyReLU(),
   nn.Linear(300, 100),
   nn.LeakyReLU(),
   nn.Linear(100, 50),
   nn.LeakyReLU(),
   nn.Linear(50, 25),
   nn.LeakyReLU(),
   nn.Linear(25, 12),
   nn.LeakyReLU(),
   nn.Linear(12, 6),
   nn.LeakyReLU(),
   nn.Linear(6, 3),
   nn.LeakyReLU(),
   nn.Linear(3, 1),
   nn.Sigmoid()
  )

 def forward(self, x):
  return self.network(x).squeeze()
 

model = PredictSafety()

test_model = HookedTransformer.from_pretrained("gpt2-small")

logits, activations = test_model.run_with_cache('hello')

print(model(torch.tensor(logits)))