import torch
import numpy as np

rng = np.random.default_rng(663)

input_tensor = torch.ones(32,2, dtype=torch.float64) * -11.
alpha = 324112638312866870.
inplace = False

output_cpu = torch.nn.functional.celu(input_tensor, alpha=alpha, inplace=inplace)
output_gpu = torch.nn.functional.celu(input_tensor.cuda(), alpha=alpha, inplace=inplace)

print(output_cpu[0, 0])  # tensor(0., dtype=torch.float64)
print(output_gpu[0, 0])  # tensor(-11.0000, device='cuda:0', dtype=torch.float64)