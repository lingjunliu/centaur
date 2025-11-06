
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def fold_inputs():
    def compute_L_dim(out_dim, k, d, p, s):
        return int(np.floor((out_dim + 2 * p - d * (k - 1) - 1) / s + 1))

    list_of_inputs = []

    # Input 1
    output_size = (4, 5)
    kernel_size = (2, 2)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (1, 1)
    Lh = compute_L_dim(output_size[0], kernel_size[0], dilation[0], padding[0], stride[0])
    Lw = compute_L_dim(output_size[1], kernel_size[1], dilation[1], padding[1], stride[1])
    L = Lh * Lw
    N, C = 1, 3
    input_tensor = torch.randn((N, C * kernel_size[0] * kernel_size[1], L), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }))

    # Input 2 (unbatched)
    output_size = (6, 6)
    kernel_size = (3, 3)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (3, 3)
    Lh = compute_L_dim(output_size[0], kernel_size[0], dilation[0], padding[0], stride[0])
    Lw = compute_L_dim(output_size[1], kernel_size[1], dilation[1], padding[1], stride[1])
    L = Lh * Lw
    C = 2
    input_tensor = torch.randn((C * kernel_size[0] * kernel_size[1], L), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }))

    # Input 3 (padding, stride)
    output_size = (7, 7)
    kernel_size = (3, 3)
    dilation = (1, 1)
    padding = (1, 1)
    stride = (2, 2)
    Lh = compute_L_dim(output_size[0], kernel_size[0], dilation[0], padding[0], stride[0])
    Lw = compute_L_dim(output_size[1], kernel_size[1], dilation[1], padding[1], stride[1])
    L = Lh * Lw
    N, C = 2, 1
    input_tensor = torch.randn((N, C * kernel_size[0] * kernel_size[1], L), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }))

    # Input 4 (dilation)
    output_size = (10, 10)
    kernel_size = (3, 3)
    dilation = (2, 2)
    padding = (0, 0)
    stride = (1, 1)
    Lh = compute_L_dim(output_size[0], kernel_size[0], dilation[0], padding[0], stride[0])
    Lw = compute_L_dim(output_size[1], kernel_size[1], dilation[1], padding[1], stride[1])
    L = Lh * Lw
    N, C = 1, 4
    input_tensor = torch.randn((N, C * kernel_size[0] * kernel_size[1], L), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }))

    # Input 5 (large stride)
    output_size = (15, 8)
    kernel_size = (2, 3)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (4, 5)
    Lh = compute_L_dim(output_size[0], kernel_size[0], dilation[0], padding[0], stride[0])
    Lw = compute_L_dim(output_size[1], kernel_size[1], dilation[1], padding[1], stride[1])
    L = Lh * Lw
    N, C = 3, 5
    input_tensor = torch.randn((N, C * kernel_size[0] * kernel_size[1], L), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }))

    # Input 6 (dilation and asymmetric padding/stride)
    output_size = (8, 9)
    kernel_size = (2, 2)
    dilation = (3, 2)
    padding = (1, 0)
    stride = (2, 3)
    Lh = compute_L_dim(output_size[0], kernel_size[0], dilation[0], padding[0], stride[0])
    Lw = compute_L_dim(output_size[1], kernel_size[1], dilation[1], padding[1], stride[1])
    L = Lh * Lw
    N, C = 1, 7
    input_tensor = torch.randn((N, C * kernel_size[0] * kernel_size[1], L), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }))

    # Input 7 (non-square kernel, padding on one dim)
    output_size = (5, 12)
    kernel_size = (1, 4)
    dilation = (1, 1)
    padding = (0, 1)
    stride = (1, 2)
    Lh = compute_L_dim(output_size[0], kernel_size[0], dilation[0], padding[0], stride[0])
    Lw = compute_L_dim(output_size[1], kernel_size[1], dilation[1], padding[1], stride[1])
    L = Lh * Lw
    N, C = 1, 2
    input_tensor = torch.randn((N, C * kernel_size[0] * kernel_size[1], L), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }))

    # Input 8 (large padding)
    output_size = (3, 3)
    kernel_size = (3, 3)
    dilation = (1, 1)
    padding = (2, 2)
    stride = (2, 2)
    Lh = compute_L_dim(output_size[0], kernel_size[0], dilation[0], padding[0], stride[0])
    Lw = compute_L_dim(output_size[1], kernel_size[1], dilation[1], padding[1], stride[1])
    L = Lh * Lw
    N, C = 2, 1
    input_tensor = torch.randn((N, C * kernel_size[0] * kernel_size[1], L), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }))

    # Input 9 (flooring due to stride)
    output_size = (9, 7)
    kernel_size = (3, 2)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (4, 3)
    Lh = compute_L_dim(output_size[0], kernel_size[0], dilation[0], padding[0], stride[0])
    Lw = compute_L_dim(output_size[1], kernel_size[1], dilation[1], padding[1], stride[1])
    L = Lh * Lw
    N, C = 1, 6
    input_tensor = torch.randn((N, C * kernel_size[0] * kernel_size[1], L), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }))

    # Input 10 (dilation > 1 with padding and stride)
    output_size = (12, 12)
    kernel_size = (5, 3)
    dilation = (2, 3)
    padding = (1, 2)
    stride = (2, 2)
    Lh = compute_L_dim(output_size[0], kernel_size[0], dilation[0], padding[0], stride[0])
    Lw = compute_L_dim(output_size[1], kernel_size[1], dilation[1], padding[1], stride[1])
    L = Lh * Lw
    N, C = 4, 3
    input_tensor = torch.randn((N, C * kernel_size[0] * kernel_size[1], L), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }))

    # Input 11 (unbatched minimal L=1)
    output_size = (2, 2)
    kernel_size = (2, 2)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (1, 1)
    Lh = compute_L_dim(output_size[0], kernel_size[0], dilation[0], padding[0], stride[0])
    Lw = compute_L_dim(output_size[1], kernel_size[1], dilation[1], padding[1], stride[1])
    L = Lh * Lw
    C = 1
    input_tensor = torch.randn((C * kernel_size[0] * kernel_size[1], L), dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }))

    # Input 12 (larger batch, non-overlapping)
    output_size = (20, 20)
    kernel_size = (4, 4)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (4, 4)
    Lh = compute_L_dim(output_size[0], kernel_size[0], dilation[0], padding[0], stride[0])
    Lw = compute_L_dim(output_size[1], kernel_size[1], dilation[1], padding[1], stride[1])
    L = Lh * Lw
    N, C = 5, 2
    input_tensor = torch.randn((N, C * kernel_size[0] * kernel_size[1], L), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": input_tensor
    }))

    return list_of_inputs

generated_inputs["torch.nn.Fold_2"] = fold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Fold_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Fold_2'.")


check_valid('torch.nn.Fold', generated_inputs['torch.nn.Fold_2'], lib="torch", suffix=2)
