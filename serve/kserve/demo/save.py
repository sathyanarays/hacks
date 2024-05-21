from torchvision import datasets, transforms
import torch

transform=transforms.Compose([
        transforms.ToTensor(),        
        ])
dataset = datasets.MNIST('../data', train=False, transform=transform)
test_kwargs = {'batch_size': 1}
loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

i = 0
for image, target in loader:
    print(image[0][0].shape)
    trans = transforms.ToPILImage()
    img = trans(image[0][0])
    img.save(str(i)+".png", "PNG")
    if i == 10:
        break
    i = i + 1
    

    