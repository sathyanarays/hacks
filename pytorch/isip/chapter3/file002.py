import torch

points = torch.tensor([[4.0,1.0], [5.0,3.0], [2.0,1.0]])
second_point = points[1]
print(second_point.storage_offset())

third_point = points[2]
print(third_point.storage_offset())

print(second_point.shape)
print(points.shape)
print(points.stride())

# Accessing an element [i][j] is equivalent to accessing i * stride[0] + j * string[1] in storage

# This indirection between tensor and storage makes some operations inexpensive like
# transposing a tesnor or extracting a subtensor, because they do not lead to memory reallocations.

print(second_point.size())
print(second_point.storage_offset())
print(second_point.stride())

second_point[0] = 10.0
print(points)

second_point = points[1].clone()
second_point[0] = 3.0
print(points)
print(second_point)
