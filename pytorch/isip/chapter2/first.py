from torchvision import models
from torchvision import transforms
from PIL import Image
import torch

dir(models)

alexnet = models.AlexNet()

resnet = models.resnet101(weights='ResNet101_Weights.DEFAULT')

img = Image.open("bobby.jpg")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

img_t = preprocess(img)

batch_t = torch.unsqueeze(img_t, 0)

resnet.eval()

out = resnet(batch_t)

with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

_, index = torch.max(out, 1)

print(labels[index[0]])