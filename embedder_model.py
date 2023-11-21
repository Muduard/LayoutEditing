import torch
import torch.nn as nn
import torch.nn.functional as F
class EmbedderModel(nn.Module):
    def __init__(self):
      super(EmbedderModel, self).__init__()
      self.fc1 = nn.Linear(768, 1200)
      
      self.fc2 = nn.Linear(1200, 1200)

      self.fc3 = nn.Linear(1200, 768)
    def forward(self, x):
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      x = F.relu(x)
      x = self.fc3(x)
      output = torch.sigmoid(x)
      return output
    
