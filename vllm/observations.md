### Observations

1. OPT model has 12 attention units
2. Each attention unit has 768 dimensions
3. Cache tensor shape: torch.Size([2, 4351, 12288])
4. (12 * 768 / 1024) = 9 and 12288 / 1024 = 12
5. This shows page size is 4 * 1024 * dtype_size
