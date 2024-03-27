import torch

some_t = torch.ones(3,4,5)
transpose_t = some_t.transpose(0,2)
print(some_t.shape)
print(transpose_t.shape)

print(some_t.stride())
print(transpose_t.stride())

print(some_t.is_contiguous())
print(transpose_t.is_contiguous())

transpose_t_c = transpose_t.contiguous()
print(transpose_t_c.is_contiguous())
print(transpose_t_c.stride())