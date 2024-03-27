import torch

points = torch.tensor([[4.0,1.0], [5.0,3.0], [2.0,1.0]])
print(points)

points_t = points.t()
print(points_t)

print(id(points.untyped_storage()) == id(points_t.untyped_storage()))

print(points.stride())
print(points_t.stride())
