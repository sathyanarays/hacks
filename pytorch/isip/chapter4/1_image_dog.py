import imageio.v2 as imageio
import torch

img_arr = imageio.imread('../../../../dlwpt-code/data/p1ch4/image-dog/bobby.jpg')
print(img_arr.shape)

img = torch.from_numpy(img_arr)
out = img.permute(2,0,1)
print(out.shape)

batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)

import os

data_dir = '../../../../dlwpt-code/data/p1ch4/image-cats/'
filenames = [name for name in os.listdir(data_dir)
             if os.path.splitext(name)[-1] == '.png']

for i, filename in enumerate(filenames):
    img_arr = imageio.imread(os.path.join(data_dir, filename))
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2,0,1)
    img_t = img_t[:3]
    batch[i] = img_t

print(batch.shape)

batch = batch.float()
n_channels = batch.shape[1]
for c in range(n_channels):
    mean = torch.mean(batch[:,c])
    std = torch.std(batch[:,c])
    batch[:,c] = (batch[:,c] - mean) / std

print(batch)