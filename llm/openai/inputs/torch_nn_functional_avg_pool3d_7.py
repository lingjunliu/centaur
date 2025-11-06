
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def avg_pool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.arange(4*4*4, dtype=torch.float32).reshape(1,1,4,4,4).numpy()
    kernel_size = (2, 2, 2)
    stride = [2, 2, 2]
    padding = [0, 0, 0]
    ceil_mode = False
    count_include_pad = True
    divisor_override = 8
    input_dict = {
        "input": input_arr,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.randn(2, 3, 5, 7, 9, dtype=torch.double).numpy()
    kernel_size = (3, 3, 3)
    stride = [2, 2, 2]
    padding = [1, 1, 1]
    ceil_mode = False
    count_include_pad = False
    divisor_override = 3
    input_dict = {
        "input": input_arr,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.tensor([[[[[1., -1., 2.],
                                 [3., -3., 4.],
                                 [5., -5., 6.]],
                                [[-2., 2., -3.],
                                 [4., -4., 5.],
                                 [6., -6., 7.]],
                                [[-1., 1., -2.],
                                 [2., -2., 3.],
                                 [4., -4., 5.]]],
                               [[[-1., 2., -2.],
                                 [0., 1., -1.],
                                 [3., -3., 4.]],
                                [[2., -2., 3.],
                                 [-4., 4., -5.],
                                 [6., -6., 7.]],
                                [[1., -1., 2.],
                                 [-2., 2., -3.],
                                 [3., -3., 4.]]]]], dtype=torch.float32).numpy()
    kernel_size = (1, 2, 3)
    stride = [1, 2, 3]
    padding = [0, 0, 0]
    ceil_mode = True
    count_include_pad = False
    divisor_override = 1
    input_dict = {
        "input": input_arr,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.randn(4, 1, 8, 8, 8, dtype=torch.float32).numpy()
    kernel_size = (2, 3, 4)
    stride = [2, 3, 4]
    padding = [1, 0, 2]
    ceil_mode = False
    count_include_pad = True
    divisor_override = 6
    input_dict = {
        "input": input_arr,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.ones(1, 1, 2, 3, 4, dtype=torch.float32).numpy()
    kernel_size = (2, 2, 2)
    stride = [1, 1, 2]
    padding = [0, 1, 0]
    ceil_mode = True
    count_include_pad = True
    divisor_override = 2
    input_dict = {
        "input": input_arr,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.linspace(-10, 10, steps=3*2*6*5*4, dtype=torch.float32).reshape(3, 2, 6, 5, 4).numpy()
    kernel_size = (3, 2, 2)
    stride = [1, 2, 2]
    padding = [1, 0, 1]
    ceil_mode = False
    count_include_pad = False
    divisor_override = 4
    input_dict = {
        "input": input_arr,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = torch.rand(2, 1, 9, 7, 5, dtype=torch.float32).numpy()
    kernel_size = (1, 1, 1)
    stride = [1, 1, 1]
    padding = [0, 0, 0]
    ceil_mode = False
    count_include_pad = True
    divisor_override = 1
    input_dict = {
        "input": input_arr,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = (-2.0) * torch.ones(1, 4, 10, 10, 10, dtype=torch.float32).numpy()
    kernel_size = (5, 5, 5)
    stride = [3, 3, 3]
    padding = [2, 2, 2]
    ceil_mode = True
    count_include_pad = False
    divisor_override = 7
    input_dict = {
        "input": input_arr,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (fix padding to satisfy kernel_size=1 constraint)
    input_arr = torch.randn(2, 3, 3, 4, 5, dtype=torch.double).numpy()
    kernel_size = (2, 1, 3)
    stride = [2, 1, 1]
    padding = [0, 0, 1]
    ceil_mode = False
    count_include_pad = True
    divisor_override = 2
    input_dict = {
        "input": input_arr,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = torch.randn(1, 2, 7, 6, 5, dtype=torch.float32).numpy()
    kernel_size = (3, 4, 2)
    stride = [1, 3, 2]
    padding = [1, 2, 0]
    ceil_mode = True
    count_include_pad = False
    divisor_override = 5
    input_dict = {
        "input": input_arr,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = torch.arange(5*1*4*6*8, dtype=torch.float32).reshape(5, 1, 4, 6, 8).numpy()
    kernel_size = (4, 1, 2)
    stride = [2, 1, 2]
    padding = [0, 0, 1]
    ceil_mode = False
    count_include_pad = True
    divisor_override = 2
    input_dict = {
        "input": input_arr,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = torch.tensor([[[[[1., 2., 3.],
                                 [4., 5., 6.],
                                 [7., 8., 9.]],
                                [[10., 11., 12.],
                                 [13., 14., 15.],
                                 [16., 17., 18.]],
                                [[19., 20., 21.],
                                 [22., 23., 24.],
                                 [25., 26., 27.]]]]], dtype=torch.float32).numpy()
    kernel_size = (3, 3, 3)
    stride = [3, 3, 3]
    padding = [0, 0, 0]
    ceil_mode = False
    count_include_pad = False
    divisor_override = 27
    input_dict = {
        "input": input_arr,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.avg_pool3d_7"] = avg_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.avg_pool3d_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool3d_7'.")


check_valid('torch.nn.functional.avg_pool3d', generated_inputs['torch.nn.functional.avg_pool3d_7'], lib="torch", suffix=7)
