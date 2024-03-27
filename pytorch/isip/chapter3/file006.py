import torch

zeroes = torch.zeros([3,4])
print(zeroes)

torch.save(zeroes, 'zeroes.t')

points = torch.load('zeroes.t')
print(points)

import h5py

f = h5py.File('file.hdf5', 'w')
dset = f.create_dataset('zeroes', data=points)
f.close()