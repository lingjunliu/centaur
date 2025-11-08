
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, math

def fold_inputs():
    list_of_inputs = []

    # Input 1
    output_size = (4, 5)
    kernel_size = (2, 2)
    dilation = 1
    padding = 0
    stride = 1
    H, W = output_size
    kH, kW = kernel_size
    H_blocks = (H + 2*padding - dilation*(kH - 1) - 1) // stride + 1
    W_blocks = (W + 2*padding - dilation*(kW - 1) - 1) // stride + 1
    L = int(H_blocks * W_blocks)
    N, C = 1, 3
    inp = torch.randn(N, C * (kH * kW), L, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": inp
    }))

    # Input 2
    output_size = (7, 7)
    kernel_size = (3, 3)
    dilation = 1
    padding = 1
    stride = 2
    H, W = output_size
    kH, kW = kernel_size
    H_blocks = (H + 2*padding - dilation*(kH - 1) - 1) // stride + 1
    W_blocks = (W + 2*padding - dilation*(kW - 1) - 1) // stride + 1
    L = int(H_blocks * W_blocks)
    N, C = 2, 4
    inp = torch.randn(N, C * (kH * kW), L, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": inp
    }))

    # Input 3
    output_size = (7, 6)
    kernel_size = (2, 2)
    dilation = 2
    padding = 0
    stride = 1
    H, W = output_size
    kH, kW = kernel_size
    H_blocks = (H + 2*padding - dilation*(kH - 1) - 1) // stride + 1
    W_blocks = (W + 2*padding - dilation*(kW - 1) - 1) // stride + 1
    L = int(H_blocks * W_blocks)
    N, C = 1, 2
    inp = torch.randn(N, C * (kH * kW), L, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": inp
    }))

    # Input 4 (unbatched)
    output_size = (10, 8)
    kernel_size = (3, 2)
    dilation = 3
    padding = 2
    stride = 2
    H, W = output_size
    kH, kW = kernel_size
    H_blocks = (H + 2*padding - dilation*(kH - 1) - 1) // stride + 1
    W_blocks = (W + 2*padding - dilation*(kW - 1) - 1) // stride + 1
    L = int(H_blocks * W_blocks)
    C = 3
    inp = torch.randn(C * (kH * kW), L, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": inp
    }))

    # Input 5
    output_size = (5, 4)
    kernel_size = (4, 5)
    dilation = 1
    padding = 3
    stride = 1
    H, W = output_size
    kH, kW = kernel_size
    H_blocks = (H + 2*padding - dilation*(kH - 1) - 1) // stride + 1
    W_blocks = (W + 2*padding - dilation*(kW - 1) - 1) // stride + 1
    L = int(H_blocks * W_blocks)
    N, C = 3, 1
    inp = torch.randn(N, C * (kH * kW), L, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": inp
    }))

    # Input 6 (float64)
    output_size = (9, 7)
    kernel_size = (1, 1)
    dilation = 1
    padding = 0
    stride = 3
    H, W = output_size
    kH, kW = kernel_size
    H_blocks = (H + 2*padding - dilation*(kH - 1) - 1) // stride + 1
    W_blocks = (W + 2*padding - dilation*(kW - 1) - 1) // stride + 1
    L = int(H_blocks * W_blocks)
    N, C = 4, 2
    inp = torch.randn(N, C * (kH * kW), L, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": inp
    }))

    # Input 7
    output_size = (12, 13)
    kernel_size = (3, 3)
    dilation = 4
    padding = 5
    stride = 2
    H, W = output_size
    kH, kW = kernel_size
    H_blocks = (H + 2*padding - dilation*(kH - 1) - 1) // stride + 1
    W_blocks = (W + 2*padding - dilation*(kW - 1) - 1) // stride + 1
    L = int(H_blocks * W_blocks)
    N, C = 2, 5
    inp = torch.randn(N, C * (kH * kW), L, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": inp
    }))

    # Input 8 (unbatched)
    output_size = (8, 9)
    kernel_size = (3, 4)
    dilation = 2
    padding = 2
    stride = 1
    H, W = output_size
    kH, kW = kernel_size
    H_blocks = (H + 2*padding - dilation*(kH - 1) - 1) // stride + 1
    W_blocks = (W + 2*padding - dilation*(kW - 1) - 1) // stride + 1
    L = int(H_blocks * W_blocks)
    C = 2
    inp = torch.randn(C * (kH * kW), L, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": inp
    }))

    # Input 9 (float16)
    output_size = (1, 1)
    kernel_size = (1, 1)
    dilation = 1
    padding = 0
    stride = 1
    H, W = output_size
    kH, kW = kernel_size
    H_blocks = (H + 2*padding - dilation*(kH - 1) - 1) // stride + 1
    W_blocks = (W + 2*padding - dilation*(kW - 1) - 1) // stride + 1
    L = int(H_blocks * W_blocks)
    N, C = 1, 1
    inp = torch.randn(N, C * (kH * kW), L, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": inp
    }))

    # Input 10
    output_size = (15, 10)
    kernel_size = (5, 4)
    dilation = 1
    padding = 1
    stride = 3
    H, W = output_size
    kH, kW = kernel_size
    H_blocks = (H + 2*padding - dilation*(kH - 1) - 1) // stride + 1
    W_blocks = (W + 2*padding - dilation*(kW - 1) - 1) // stride + 1
    L = int(H_blocks * W_blocks)
    N, C = 1, 7
    inp = torch.randn(N, C * (kH * kW), L, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": inp
    }))

    # Input 11
    output_size = (9, 9)
    kernel_size = (2, 3)
    dilation = 2
    padding = 1
    stride = 2
    H, W = output_size
    kH, kW = kernel_size
    H_blocks = (H + 2*padding - dilation*(kH - 1) - 1) // stride + 1
    W_blocks = (W + 2*padding - dilation*(kW - 1) - 1) // stride + 1
    L = int(H_blocks * W_blocks)
    N, C = 5, 1
    inp = torch.randn(N, C * (kH * kW), L, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": inp
    }))

    # Input 12 (unbatched, float64)
    output_size = (20, 20)
    kernel_size = (2, 2)
    dilation = 1
    padding = 0
    stride = 5
    H, W = output_size
    kH, kW = kernel_size
    H_blocks = (H + 2*padding - dilation*(kH - 1) - 1) // stride + 1
    W_blocks = (W + 2*padding - dilation*(kW - 1) - 1) // stride + 1
    L = int(H_blocks * W_blocks)
    C = 3
    inp = torch.randn(C * (kH * kW), L, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "output_size": output_size,
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
        "input": inp
    }))

    return list_of_inputs

generated_inputs["torch.nn.Fold_1"] = fold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Fold_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Fold_1'.")


check_valid('torch.nn.Fold', generated_inputs['torch.nn.Fold_1'], lib="torch", suffix=1)
