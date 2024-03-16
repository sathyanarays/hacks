import torch

tensor = torch.tensor([[0,1],[2,3],[4,5]], dtype=torch.float64)

tensor.storage()[1] = 7
print(tensor.storage())
print(tensor)