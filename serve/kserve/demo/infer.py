import torch
import torch.nn as nn
from model import Net
from os import listdir
from os.path import isfile, join
from PIL import Image
from torchvision import datasets, transforms

model = Net()
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()

inputs = listdir("inputs")
for input in inputs:
    img = Image.open("inputs/"+input)

    # Preprocess
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor,0)
    
    # Predict
    output = model(img_tensor)    

    # Postprocess
    pred = output.argmax(dim=1, keepdim=True)
    print(input, pred)

