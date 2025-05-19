import torch

tensor = torch.tensor([30,30,30,
                       30,30,30,
                       30,30,30,
                       30,30], dtype=torch.int8)

a = torch.combinations(tensor, 10, with_replacement=True)
print(a)