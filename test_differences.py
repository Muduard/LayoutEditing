import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(8888)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

def hook_fn(module, input, output):
    # Modify the output in some way
    output *= 2

model = MyModel()

# Register the hook to the desired layer
hook_handle = model.fc.register_forward_hook(hook_fn)








# Now when you pass input through the model, the output will be modified by the hook
input_data = torch.randn(1, 10)
output = model(input_data)

print(output)  # Modified output

# Don't forget to remove the hook when you're done
#hook_handle.remove()
#h1 = cv2.imread("cat4.png")

#h1 = np.where(h1 < 100, 0, 255)
#h1[:,:,0] = 0
#h1[:,:,1] = 0
#cv2.imwrite("cat5.png",h1)

