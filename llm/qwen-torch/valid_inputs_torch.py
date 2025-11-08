generated_inputs = {}
import torch, copy

def amin_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    dim = 1
    keepdim = False
    out = torch.zeros(2).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]]).numpy()
    dim = 0
    keepdim = True
    out = torch.zeros(1).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn(4, 5, 6).numpy()
    dim = 2
    keepdim = False
    out = torch.zeros(4, 5).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dim = 1
    keepdim = True
    out = torch.zeros(2, 1).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]]).numpy()
    dim = 0
    keepdim = False
    out = torch.zeros(2).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dim = 0
    keepdim = False
    out = torch.zeros(2).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[-1.0, 2.0], [-3.0, 4.0], [-5.0, 6.0]]).numpy()
    dim = 1
    keepdim = True
    out = torch.zeros(3, 1).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(3, 4, 5).numpy()
    dim = 2
    keepdim = True
    out = torch.zeros(3, 4).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    dim = 1
    keepdim = False
    out = torch.zeros(3).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]]).numpy()
    dim = 0
    keepdim = False
    out = torch.zeros(2).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.amin_1"] = amin_inputs()

import torch, copy

def are_deterministic_algorithms_enabled_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    
    input_dict = {
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    
    input_dict = {
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.zeros((1, 2, 3)).numpy()
    
    input_dict = {
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    
    input_dict = {
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]).numpy()
    
    input_dict = {
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0]).numpy()
    
    input_dict = {
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((3,)).numpy()
    
    input_dict = {
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.zeros((2, 2, 2)).numpy()
    
    input_dict = {
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    
    input_dict = {
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1.0, 0.0, 1.0]).numpy()
    
    input_dict = {
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.are_deterministic_algorithms_enabled"] = are_deterministic_algorithms_enabled_inputs()

import torch, copy

def argsort_inputs():
    list_of_inputs = []

    
    # Input 1, valid
    input = torch.tensor([[ 0.0785, 1.5267, -0.8521, 0.4065],
                        [ 0.1598, 0.0788, -0.0745, -1.2700],
                        [ 1.2208, 1.0722, -0.7064, 1.2564],
                        [ 0.0669, -0.2318, -0.8229, -0.9280]]).numpy()
    dim = 1
    descending = False
    stable = False
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[-0.8521, 0.4065, 0.0785, 1.5267],
                        [-1.2700, -0.0745, 0.0788, 0.1598],
                        [-0.7064, 1.2564, 1.0722, 1.2208],
                        [-0.9280, -0.8229, -0.2318, 0.0669]]).numpy()
    dim = 0
    descending = True
    stable = False
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    dim = 0
    descending = False
    stable = True
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    dim = 1
    descending = True
    stable = True
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[1.0], [2.0], [3.0]]).numpy()
    dim = 0
    descending = False
    stable = False
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    dim = 0
    descending = True
    stable = False
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([-0.1, 0.2, -0.3, 0.4]).numpy()
    dim = 0
    descending = False
    stable = True
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]).numpy()
    dim = 1
    descending = False
    stable = True
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    dim = 0
    descending = True
    stable = True
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    dim = 0
    descending = False
    stable = False
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.argsort"] = argsort_inputs()

import torch, copy

def bitwise_left_shift_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1, 2, 3]).numpy()
    other = torch.tensor([0, 1, 2]).numpy()
    out = torch.tensor([0, 0, 0], dtype=torch.int64).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3), dtype=torch.int64).numpy()
    other = torch.zeros((2, 3), dtype=torch.int64).numpy()
    out = torch.tensor([0, 0, 0], dtype=torch.int64).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([10, 20, 30], dtype=torch.int64).numpy()
    other = torch.tensor([1, 2, 3], dtype=torch.int64).numpy()
    out = torch.tensor([0, 0, 0], dtype=torch.int64).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1, 2, 3, 4], dtype=torch.int64).numpy()
    other = torch.tensor([1, 2, 3, 4], dtype=torch.int64).numpy()
    out = torch.tensor([0, 0, 0, 0], dtype=torch.int64).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1, 2], dtype=torch.int64).numpy()
    other = torch.tensor([0, 1], dtype=torch.int64).numpy()
    out = torch.tensor([0, 0], dtype=torch.int64).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([-1, -2, -3], dtype=torch.int64).numpy()
    other = torch.tensor([1, 2, 3], dtype=torch.int64).numpy()
    out = torch.tensor([0, 0, 0], dtype=torch.int64).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64).numpy()
    other = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64).numpy()
    out = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int64).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([100, 200, 300], dtype=torch.int64).numpy()
    other = torch.tensor([1, 2, 3], dtype=torch.int64).numpy()
    out = torch.tensor([0, 0, 0], dtype=torch.int64).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1, 2, 3], dtype=torch.int64).numpy()
    other = torch.tensor([0, 1, 2], dtype=torch.int64).numpy()
    out = torch.tensor([0, 0, 0], dtype=torch.int64).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64).numpy()
    other = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64).numpy()
    out = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int64).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.bitwise_left_shift"] = bitwise_left_shift_inputs()

import torch, copy

def dequantize_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    tensor = torch.tensor([0.1, 0.2, 0.3]).numpy()
    
    input_dict = {
        "tensor": tensor
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    tensor = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]]).numpy()
    
    input_dict = {
        "tensor": tensor
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    tensor = torch.tensor([[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]],
                          [[7.0, 8.0, 9.0],
                           [10.0, 11.0, 12.0]]]).numpy()
    
    input_dict = {
        "tensor": tensor
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    tensor = torch.tensor([[[[1.0, 2.0],
                           [3.0, 4.0]],
                          [[5.0, 6.0],
                           [7.0, 8.0]]],
                         [[[9.0, 10.0],
                          [11.0, 12.0]],
                          [[13.0, 14.0],
                          [15.0, 16.0]]]]).numpy()
    
    input_dict = {
        "tensor": tensor
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0]],
                          [[9.0, 10.0, 11.0, 12.0],
                           [13.0, 14.0, 15.0, 16.0]]]).numpy()
    
    input_dict = {
        "tensor": tensor
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    tensor = torch.tensor([[[1.0, 2.0, 3.0]]]).numpy()
    
    input_dict = {
        "tensor": tensor
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    tensor = torch.tensor([[[[1.0, 2.0],
                           [3.0, 4.0]]]]).numpy()
    
    input_dict = {
        "tensor": tensor
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    tensor = torch.tensor([[1.0], [2.0]]).numpy()
    
    input_dict = {
        "tensor": tensor
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    tensor = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0],
                          [5.0, 6.0]]).numpy()
    
    input_dict = {
        "tensor": tensor
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    tensor = torch.tensor([[[1.0, 2.0],
                           [3.0, 4.0],
                           [5.0, 6.0]]]).numpy()
    
    input_dict = {
        "tensor": tensor
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.dequantize_1"] = dequantize_inputs()

import torch, copy

def equal_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1, 2, 3]).numpy()
    other = torch.tensor([1, 2, 3]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    other = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1, 2, 3], dtype=torch.float32).numpy()
    other = torch.tensor([1, 2, 3], dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.ones((2, 3)).numpy()
    other = torch.ones((2, 3)).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.zeros((3, 4)).numpy()
    other = torch.zeros((3, 4)).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1, 2, 3], dtype=torch.int64).numpy()
    other = torch.tensor([1, 2, 3], dtype=torch.int64).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-1, -2, -3]).numpy()
    other = torch.tensor([-1, -2, -3]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.0]).numpy()
    other = torch.tensor([0.0]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1, 2, 3], dtype=torch.float64).numpy()
    other = torch.tensor([1, 2, 3], dtype=torch.float64).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.equal"] = equal_inputs()

import torch, copy

def fix_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.5, 2.7, 3.9]).numpy()   # tensor
    out = torch.zeros((3,)).numpy()    # tensor

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([-1.5, -2.7, -3.9]).numpy()   # tensor
    out = torch.zeros((3,)).numpy()    # tensor

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((2, 3)).numpy()
    out = torch.zeros((2, 3)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.zeros((1, 4, 5)).numpy()
    out = torch.zeros((1, 4, 5)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.0]).numpy()
    out = torch.zeros((1,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([-0.1]).numpy()
    out = torch.zeros((1,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.99]).numpy()
    out = torch.zeros((1,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-1.99]).numpy()
    out = torch.zeros((1,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.5]).numpy()
    out = torch.zeros((1,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-0.5]).numpy()
    out = torch.zeros((1,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.fix"] = fix_inputs()

import torch, copy

def ge_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[1, 2], [3, 4]]).numpy()
    other = torch.tensor([[1, 1], [4, 4]]).numpy()
    out = torch.tensor([[True, True], [False, True]]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1, 2, 3]).numpy()
    other = torch.tensor([0, 1, 2]).numpy()
    out = torch.tensor([True, True, True]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    other = torch.tensor([[1, 1, 1], [1, 1, 1]]).numpy()
    out = torch.tensor([[True, True, True], [True, True, True]]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[-1, -2, -3], [0, 1, 2]]).numpy()
    other = torch.tensor([[0, 0, 0], [-1, -1, -1]]).numpy()
    out = torch.tensor([[False, False, True], [True, True, True]]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1.0]).numpy()
    other = torch.tensor([1.0]).numpy()
    out = torch.tensor([True]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0, 1, 2, 3]).numpy()
    other = torch.tensor([0, 1, 2, 3]).numpy()
    out = torch.tensor([True, True, True, True]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[0, 1], [2, 3]]).numpy()
    other = torch.tensor([[0, 0], [2, 2]]).numpy()
    out = torch.tensor([[True, True], [True, True]]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-10, -5, 0]).numpy()
    other = torch.tensor([-10, -5, 0]).numpy()
    out = torch.tensor([True, True, True]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[1, 2], [3, 4], [5, 6]]).numpy()
    other = torch.tensor([[1, 1], [4, 4], [6, 6]]).numpy()
    out = torch.tensor([[True, True], [False, True], [False, True]]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    other = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    out = torch.tensor([[True, True, True], [True, True, True]]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.ge"] = ge_inputs()

import torch, copy

def imag_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1+2j, 3+4j, 5+6j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([1+2j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([0+0j, 1+1j, 2+2j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([-1-1j, -2-2j, -3-3j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.5+0.5j, 1.5+1.5j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[0.1+0.2j, 0.3+0.4j], [0.5+0.6j, 0.7+0.8j]], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1j, 2j, 3j, 4j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1+1j, 2+2j, 3+3j, 4+4j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0+2j, 3.0+4j, 5.0+6j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.imag"] = imag_inputs()

import torch, copy

def is_floating_point_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor(1.5, dtype=torch.float32).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([0.0, -1.0, 2.0], dtype=torch.float64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.ones((3, 4), dtype=torch.float32).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0], dtype=torch.float64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor(1.0, dtype=torch.float64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.is_floating_point"] = is_floating_point_inputs()

import torch, copy

def is_grad_enabled_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.zeros((1, 2, 3)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-1.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.is_grad_enabled"] = is_grad_enabled_inputs()

import torch, copy

def is_storage_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "obj": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    input_dict = {
        "obj": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {
        "obj": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.zeros((1, 2, 3)).numpy()
    input_dict = {
        "obj": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.0]).numpy()
    input_dict = {
        "obj": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "obj": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    input_dict = {
        "obj": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    input_dict = {
        "obj": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0]).numpy()
    input_dict = {
        "obj": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([0.0]).numpy()
    input_dict = {
        "obj": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.is_storage"] = is_storage_inputs()

import torch, copy

def isreal_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1, 2, 3]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1+1j, 2+0j, 3+1j]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([0+0j, 1+0j, 2+0j]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([-1, -2, -3]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.5, 1.5, 2.5]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[1, 2], [3, 4]]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1+1j, 2+0j, 3+1j]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[0.0, 1.0], [2.0, 3.0]]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.isreal"] = isreal_inputs()

import torch, copy

def lcm_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([5, 10, 15]).numpy()   # tensor
    other = torch.tensor([3, 4, 5]).numpy()    # tensor
    out = torch.tensor([15, 20, 15]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([10, 20, 30]).numpy()   # tensor
    other = torch.tensor([3, 4, 5]).numpy()     # tensor
    out = torch.tensor([30, 20, 30]).numpy()     # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([1, 2, 3]).numpy()   # tensor
    other = torch.tensor([4, 5, 6]).numpy()  # tensor
    out = torch.tensor([4, 10, 6]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([0, 1, 2]).numpy()   # tensor
    other = torch.tensor([3, 4, 5]).numpy()  # tensor
    out = torch.tensor([0, 4, 10]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0, 0, 0]).numpy()   # tensor
    other = torch.tensor([1, 2, 3]).numpy()  # tensor
    out = torch.tensor([0, 0, 0]).numpy()    # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1, 2, 3]).numpy()   # tensor
    other = torch.tensor([0, 0, 0]).numpy()  # tensor
    out = torch.tensor([0, 0, 0]).numpy()    # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([10, 20]).numpy()   # tensor
    other = torch.tensor([5, 10]).numpy()   # tensor
    out = torch.tensor([10, 20]).numpy()    # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([5, 10, 15, 20]).numpy()   # tensor
    other = torch.tensor([3, 4, 5, 6]).numpy()      # tensor
    out = torch.tensor([15, 20, 15, 60]).numpy()    # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([2, 4, 6]).numpy()   # tensor
    other = torch.tensor([3, 5, 7]).numpy()  # tensor
    out = torch.tensor([6, 20, 42]).numpy()  # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-5, -10, -15]).numpy()   # tensor
    other = torch.tensor([3, 4, 5]).numpy()         # tensor
    out = torch.tensor([15, 20, 15]).numpy()        # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.lcm"] = lcm_inputs()

import torch, copy

def lu_solve_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([1, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([2, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.lu_solve"] = lu_solve_inputs()

import torch, copy

def median_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[0.0, 1.0, 2.0],
                         [3.0, 4.0, 5.0],
                         [6.0, 7.0, 8.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([-1.0, 0.0, 1.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[1.0, 2.0],
                         [3.0, 4.0],
                         [5.0, 6.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                         [5.0, 6.0, 7.0, 8.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-1.0, 0.0, 1.0, 2.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[0.0, 1.0, 2.0],
                         [3.0, 4.0, 5.0],
                         [6.0, 7.0, 8.0],
                         [9.0, 10.0, 11.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.median_1"] = median_inputs()

import torch, copy

def median_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    dim = 1
    keepdim = False
    out = (torch.zeros(2).numpy(), torch.zeros(2).numpy().astype('int64'))
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[-1.0, 0.0, 1.0, 2.0], [3.0, 4.0, 5.0, 6.0]]).numpy()
    dim = 0
    keepdim = True
    out = (torch.zeros(2).numpy(), torch.zeros(2).numpy().astype('int64'))
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    dim = -1
    keepdim = False
    out = (torch.zeros(1).numpy(), torch.zeros(1).numpy().astype('int64'))
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0, -2.0], [3.0, -4.0], [-5.0, 6.0]]).numpy()
    dim = 1
    keepdim = True
    out = (torch.zeros(3).numpy(), torch.zeros(3).numpy().astype('int64'))
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[-1.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()
    dim = 0
    keepdim = False
    out = (torch.zeros(4).numpy(), torch.zeros(4).numpy().astype('int64'))
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]).numpy()
    dim = -1
    keepdim = True
    out = (torch.zeros(3).numpy(), torch.zeros(3).numpy().astype('int64'))
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    dim = 0
    keepdim = False
    out = (torch.zeros(3).numpy(), torch.zeros(3).numpy().astype('int64'))
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]).numpy()
    dim = -1
    keepdim = True
    out = (torch.zeros(3).numpy(), torch.zeros(3).numpy().astype('int64'))
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).numpy()
    dim = 1
    keepdim = False
    out = (torch.zeros(4).numpy(), torch.zeros(4).numpy().astype('int64'))
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]).numpy()
    dim = 0
    keepdim = True
    out = (torch.zeros(2).numpy(), torch.zeros(2).numpy().astype('int64'))
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.median_2"] = median_inputs()

import torch, copy

def nanssum_inputs():
    list_of_inputs = []

    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, float('nan'), 4.0]).numpy()
    dtype = None
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[1.0, 2.0], [3.0, float('nan')]]).numpy()
    dtype = None
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, float('nan')], [7.0, 8.0]]]).numpy()
    dtype = None
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1.0, float('nan'), float('nan'), 4.0]).numpy()
    dtype = None
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    dtype = None
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[-1.0, -2.0, float('nan'), -4.0]]).numpy()
    dtype = None
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[-1.0, float('nan'), float('nan'), -4.0]]).numpy()
    dtype = None
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[[1.0, float('nan')], [3.0, 4.0]], [[5.0, float('nan')], [7.0, 8.0]]]).numpy()
    dtype = None
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0, float('nan'), float('nan')]).numpy()
    dtype = None
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    dtype = None
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nansum_1"] = nanssum_inputs()

import torch, copy

def nansum_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, float('nan'), 4.0]).numpy()
    dim = 0
    keepdim = False
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[1.0, 2.0], [3.0, float('nan')]]).numpy()
    dim = 0
    keepdim = False
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[1.0, float('nan')], [3.0, 4.0]]).numpy()
    dim = 1
    keepdim = False
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, float('nan')], [7.0, 8.0]]]).numpy()
    dim = 0
    keepdim = True
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[[1.0, float('nan')], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dim = 1
    keepdim = True
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0, 2.0, float('nan'), 4.0]).numpy()
    dim = 0
    keepdim = True
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[[1.0, float('nan')], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dim = -1
    keepdim = False
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[[1.0, float('nan')], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dim = 0
    keepdim = False
    dtype = torch.float32
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[[1.0, float('nan')], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dim = 1
    keepdim = False
    dtype = torch.float64
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[[1.0, float('nan')], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dim = 2
    keepdim = False
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nansum_2"] = nansum_inputs()

import torch, copy

def avgpool1d_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7.]]]).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8.]]]).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9.]]]).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]]]).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 3,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]]]).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": 2,
        "padding": 2,
        "ceil_mode": True,
        "count_include_pad": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]]]).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.]]]).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.]]]).numpy()
    input_dict = {
        "kernel_size": 6,
        "stride": 3,
        "padding": 2,
        "ceil_mode": True,
        "count_include_pad": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]]]).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.]]]).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.AvgPool1d_1"] = avgpool1d_inputs()

import torch, copy

def avgpool1d_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7.]]]).numpy()
    kernel_size = (3,)
    stride = (2,)
    padding = (0,)
    ceil_mode = False
    count_include_pad = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9.]]]).numpy()
    kernel_size = (3,)
    stride = (2,)
    padding = (1,)
    ceil_mode = False
    count_include_pad = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]]]).numpy()
    kernel_size = (4,)
    stride = (3,)
    padding = (0,)
    ceil_mode = False
    count_include_pad = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]]]).numpy()
    kernel_size = (3,)
    stride = (2,)
    padding = (0,)
    ceil_mode = True
    count_include_pad = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]]]).numpy()
    kernel_size = (5,)
    stride = (4,)
    padding = (1,)
    ceil_mode = False
    count_include_pad = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.]]]).numpy()
    kernel_size = (3,)
    stride = (2,)
    padding = (0,)
    ceil_mode = False
    count_include_pad = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.]]]).numpy()
    kernel_size = (2,)
    stride = (1,)
    padding = (0,)
    ceil_mode = False
    count_include_pad = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]]]).numpy()
    kernel_size = (3,)
    stride = (2,)
    padding = (1,)
    ceil_mode = True
    count_include_pad = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.]]]).numpy()
    kernel_size = (4,)
    stride = (3,)
    padding = (0,)
    ceil_mode = False
    count_include_pad = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.]]]).numpy()
    kernel_size = (3,)
    stride = (2,)
    padding = (1,)
    ceil_mode = True
    count_include_pad = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.AvgPool1d_2"] = avgpool1d_inputs()

import torch, copy

def featurealphadropout_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.randn(20, 16, 4, 32, 32).numpy()
    input_dict = {
        "input": input,
        "p": 0.2,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.randn(10, 8, 16, 16).numpy()
    input_dict = {
        "input": input,
        "p": 0.5,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn(5, 32, 8, 8).numpy()
    input_dict = {
        "input": input,
        "p": 0.7,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(1, 64, 32, 32).numpy()
    input_dict = {
        "input": input,
        "p": 0.1,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.randn(32, 1, 64, 64).numpy()
    input_dict = {
        "input": input,
        "p": 0.3,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(8, 128, 4, 4).numpy()
    input_dict = {
        "input": input,
        "p": 0.9,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(16, 32, 8, 8).numpy()
    input_dict = {
        "input": input,
        "p": 0.4,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(128, 16, 32, 32).numpy()
    input_dict = {
        "input": input,
        "p": 0.6,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(10, 32, 8, 8).numpy()
    input_dict = {
        "input": input,
        "p": 0.2,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(64, 8, 16, 16).numpy()
    input_dict = {
        "input": input,
        "p": 0.8,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.FeatureAlphaDropout"] = featurealphadropout_inputs()

import torch, copy

def huberloss_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    target = torch.tensor([1.5, 2.5, 3.5]).numpy()
    reduction = 'mean'
    delta = 1.0
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    target = torch.ones((2, 3)).numpy()
    reduction = 'sum'
    delta = 0.5
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    target = torch.tensor([[0.5, 1.5], [2.5, 3.5]]).numpy()
    reduction = 'none'
    delta = 2.0
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    target = torch.tensor([0.5, 1.5, 2.5, 3.5]).numpy()
    reduction = 'mean'
    delta = 1.0
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    target = torch.tensor([-1.5, -2.5, -3.5]).numpy()
    reduction = 'sum'
    delta = 1.0
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[-1.0, 2.0], [3.0, -4.0]]).numpy()
    target = torch.tensor([[-1.5, 2.5], [3.5, -4.5]]).numpy()
    reduction = 'none'
    delta = 1.5
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([0.0]).numpy()
    target = torch.tensor([0.0]).numpy()
    reduction = 'mean'
    delta = 1.0
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([2.0, 2.0]).numpy()
    target = torch.tensor([1.0, 1.0]).numpy()
    reduction = 'sum'
    delta = 0.5
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    target = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5]).numpy()
    reduction = 'mean'
    delta = 2.0
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.5, 2.5]).numpy()
    target = torch.tensor([1.0, 2.0]).numpy()
    reduction = 'none'
    delta = 1.0
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.HuberLoss"] = huberloss_inputs()

import torch, copy

def logsigmoid_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((1, 1)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.0]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(3).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(5).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0, 2.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([-1.0, -2.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.ones((3, 4)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.LogSigmoid"] = logsigmoid_inputs()

import torch, copy

def maxpool2d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic case with square kernel and stride
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Non-square window with different stride
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": (3, 2),
        "stride": (2, 1),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - With padding (valid padding value)
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - With dilation
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - With return_indices
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - With ceil_mode
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - With negative values in input tensor
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - With different dimensions
    input = torch.randn(5, 8, 20, 15).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - With large padding (valid, half of kernel size)
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - With large dilation
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_1"] = maxpool2d_inputs()

import torch, copy

def maxpool2d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic case with square kernel and stride
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Non-square kernel and stride
    input = torch.randn(10, 8, 64, 32).numpy()
    input_dict = {
        "kernel_size": (3, 2),
        "stride": (2, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - With padding
    input = torch.randn(5, 4, 100, 50).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - With dilation
    input = torch.randn(5, 4, 20, 20).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - With return_indices True
    input = torch.randn(1, 2, 10, 10).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - With ceil_mode True
    input = torch.randn(1, 1, 10, 10).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Mixed parameters
    input = torch.randn(3, 4, 256, 128).numpy()
    input_dict = {
        "kernel_size": (4, 4),
        "stride": (2, 2),
        "padding": (2, 2),
        "dilation": (2, 2),
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Negative values in input tensor
    input = torch.randn(1, 2, 10, 10).numpy()
    input[0, 0, 0, 0] = -1.0
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Different dimensions in input tensor
    input = torch.randn(2, 4, 64, 32).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - Large input tensor
    input = torch.randn(2, 8, 1024, 512).numpy()
    input_dict = {
        "kernel_size": (5, 5),
        "stride": (3, 3),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_2"] = maxpool2d_inputs()

import torch, copy

def maxpool3d_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.randn(20, 16, 50, 44, 31).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.randn(10, 8, 30, 20, 15).numpy()
    input_dict = {
        "kernel_size": (3, 2, 2),
        "stride": (2, 1, 2),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn(5, 4, 60, 50, 40).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(15, 10, 25, 35, 45).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 3,
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.randn(8, 6, 40, 30, 25).numpy()
    input_dict = {
        "kernel_size": (4, 3, 2),
        "stride": (1, 2, 1),
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(12, 8, 50, 45, 35).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(6, 5, 35, 25, 20).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (1, 1, 1),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(3, 2, 60, 50, 40).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(10, 7, 45, 35, 30).numpy()
    input_dict = {
        "kernel_size": (5, 4, 3),
        "stride": (2, 1, 2),
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(5, 3, 30, 25, 20).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_1"] = maxpool3d_inputs()

import torch, copy

def multilabelmarginloss_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target = torch.tensor([[3, 0, -1, 1]]).numpy()
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target = torch.tensor([[3, 0, -1, 1]]).numpy()
    input_dict = {
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target = torch.tensor([[3, 0, -1, 1]]).numpy()
    input_dict = {
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.3, 0.5, 0.7, 0.9]]).numpy()
    target = torch.tensor([[3, 0, -1, 1], [2, 1, -1, 0]]).numpy()
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.3, 0.5, 0.7, 0.9]]).numpy()
    target = torch.tensor([[3, 0, -1, 1], [2, 1, -1, 0]]).numpy()
    input_dict = {
        "size_average": False,
        "reduce": False,
        "reduction": "sum",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.3, 0.5, 0.7, 0.9]]).numpy()
    target = torch.tensor([[3, 0, -1, 1], [2, 1, -1, 0]]).numpy()
    input_dict = {
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.3, 0.5, 0.7, 0.9], [0.2, 0.4, 0.6, 0.8]]).numpy()
    target = torch.tensor([[3, 0, -1, 1], [2, 1, -1, 0], [0, 2, -1, 3]]).numpy()
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.3, 0.5, 0.7, 0.9], [0.2, 0.4, 0.6, 0.8]]).numpy()
    target = torch.tensor([[3, 0, -1, 1], [2, 1, -1, 0], [0, 2, -1, 3]]).numpy()
    input_dict = {
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.3, 0.5, 0.7, 0.9], [0.2, 0.4, 0.6, 0.8]]).numpy()
    target = torch.tensor([[3, 0, -1, 1], [2, 1, -1, 0], [0, 2, -1, 3]]).numpy()
    input_dict = {
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.3, 0.5, 0.7, 0.9], [0.2, 0.4, 0.6, 0.8]]).numpy()
    target = torch.tensor([[3, 0, -1, 1], [2, 1, -1, 0], [0, 2, -1, 3]]).numpy()
    input_dict = {
        "size_average": False,
        "reduce": False,
        "reduction": "sum",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.MultiLabelMarginLoss"] = multilabelmarginloss_inputs()

import torch, copy

def multi_margin_loss_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target = torch.tensor([3]).numpy()
    input_dict = {
        "p": 1,
        "margin": 1.0,
        "weight": torch.ones(4).numpy(),
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target = torch.tensor([3]).numpy()
    input_dict = {
        "p": 2,
        "margin": 1.0,
        "weight": torch.ones(4).numpy(),
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.5, 0.6, 0.7, 0.9]]).numpy()
    target = torch.tensor([3, 2]).numpy()
    input_dict = {
        "p": 1,
        "margin": 2.0,
        "weight": torch.ones(4).numpy(),
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.5, 0.6, 0.7, 0.9]]).numpy()
    target = torch.tensor([3, 2]).numpy()
    input_dict = {
        "p": 2,
        "margin": 1.0,
        "weight": torch.ones(4).numpy(),
        "size_average": False,
        "reduce": False,
        "reduction": "none",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target = torch.tensor([3]).numpy()
    input_dict = {
        "p": 1,
        "margin": 1.0,
        "weight": torch.ones(4).numpy(),
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target = torch.tensor([3]).numpy()
    input_dict = {
        "p": 1,
        "margin": 0.5,
        "weight": torch.ones(4).numpy(),
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target = torch.tensor([3]).numpy()
    input_dict = {
        "p": 1,
        "margin": 1.0,
        "weight": torch.ones(4).numpy(),
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.5, 0.6, 0.7, 0.9]]).numpy()
    target = torch.tensor([3, 2]).numpy()
    input_dict = {
        "p": 2,
        "margin": 1.0,
        "weight": torch.ones(4).numpy(),
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target = torch.tensor([3]).numpy()
    input_dict = {
        "p": 2,
        "margin": 2.0,
        "weight": torch.ones(4).numpy(),
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target = torch.tensor([3]).numpy()
    input_dict = {
        "p": 1,
        "margin": 1.0,
        "weight": torch.ones(4).numpy(),
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.MultiMarginLoss"] = multi_margin_loss_inputs()

import torch, copy

def rrelu_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, -2.0, 3.0]).numpy()
    lower = 0.1
    upper = 0.3
    inplace = False
    
    input_dict = {
        "input": input,
        "lower": lower,
        "upper": upper,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    lower = 0.5
    upper = 0.8
    inplace = True
    
    input_dict = {
        "input": input,
        "lower": lower,
        "upper": upper,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((2, 3)).numpy()
    lower = 0.125
    upper = 0.3333333333333333
    inplace = False
    
    input_dict = {
        "input": input,
        "lower": lower,
        "upper": upper,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn((1, 2, 3)).numpy()
    lower = 0.1
    upper = 0.3
    inplace = True
    
    input_dict = {
        "input": input,
        "lower": lower,
        "upper": upper,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.0]).numpy()
    lower = 0.2
    upper = 0.4
    inplace = False
    
    input_dict = {
        "input": input,
        "lower": lower,
        "upper": upper,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0, -1.0]).numpy()
    lower = 0.1
    upper = 0.3
    inplace = False
    
    input_dict = {
        "input": input,
        "lower": lower,
        "upper": upper,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((3, 4)).numpy()
    lower = 0.125
    upper = 0.3333333333333333
    inplace = False
    
    input_dict = {
        "input": input,
        "lower": lower,
        "upper": upper,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn((4, 5)).numpy()
    lower = 0.1
    upper = 0.3
    inplace = True
    
    input_dict = {
        "input": input,
        "lower": lower,
        "upper": upper,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0, -2.0, 3.0, -4.0]).numpy()
    lower = 0.2
    upper = 0.4
    inplace = False
    
    input_dict = {
        "input": input,
        "lower": lower,
        "upper": upper,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([0.0]).numpy()
    lower = 0.1
    upper = 0.3
    inplace = True
    
    input_dict = {
        "input": input,
        "lower": lower,
        "upper": upper,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.RReLU"] = rrelu_inputs()

import torch, copy

def replicationpad3d_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = 3
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.randn(1, 1, 2, 3, 4).numpy()
    padding = 1
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn(2, 3, 5, 6, 7).numpy()
    padding = 2
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(4, 2, 10, 15, 20).numpy()
    padding = 4
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.randn(8, 5, 1, 2, 3).numpy()
    padding = 5
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(1, 1, 10, 10, 10).numpy()
    padding = 0
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(3, 2, 5, 6, 7).numpy()
    padding = -1
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(5, 3, 2, 3, 4).numpy()
    padding = 6
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(7, 1, 3, 2, 1).numpy()
    padding = 7
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(6, 4, 8, 9, 10).numpy()
    padding = 8
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad3d_1"] = replicationpad3d_inputs()

import torch, copy

def silu_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((2, 3)).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(5).numpy()
    input_dict = {
        "input": input,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]).numpy()
    input_dict = {
        "input": input,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.zeros((3, 4, 5)).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([0.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(2, 2, 2).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1.5, -2.5, -3.5]).numpy()
    input_dict = {
        "input": input,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.SiLU"] = silu_inputs()

import torch, copy

def smoothl1loss_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    target = torch.tensor([0.1, 0.2, 0.3]).numpy()
    beta = 1.0
    size_average = True
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "beta": beta,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    target = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]]).numpy()
    beta = 0.5
    size_average = False
    reduce = True
    reduction = 'sum'
    
    input_dict = {
        "input": input,
        "target": target,
        "beta": beta,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([-1.0, 0.0, 1.0]).numpy()
    target = torch.tensor([-0.5, 0.0, 0.5]).numpy()
    beta = 2.0
    size_average = True
    reduce = False
    reduction = 'none'
    
    input_dict = {
        "input": input,
        "target": target,
        "beta": beta,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((3, 2, 2)).numpy()
    target = torch.zeros((3, 2, 2)).numpy()
    beta = 1.5
    size_average = False
    reduce = False
    reduction = 'none'
    
    input_dict = {
        "input": input,
        "target": target,
        "beta": beta,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1.0]).numpy()
    target = torch.tensor([2.0]).numpy()
    beta = 0.0
    size_average = True
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "beta": beta,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.5, 1.5, 2.5]).numpy()
    target = torch.tensor([0.1, 0.2, 0.3]).numpy()
    beta = 0.2
    size_average = False
    reduce = True
    reduction = 'sum'
    
    input_dict = {
        "input": input,
        "target": target,
        "beta": beta,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((2, 2)).numpy()
    target = torch.zeros((2, 2)).numpy()
    beta = 1.0
    size_average = True
    reduce = False
    reduction = 'none'
    
    input_dict = {
        "input": input,
        "target": target,
        "beta": beta,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0, 2.0]).numpy()
    target = torch.tensor([1.0, 2.0]).numpy()
    beta = 3.0
    size_average = True
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "beta": beta,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    target = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    beta = 0.5
    size_average = False
    reduce = True
    reduction = 'sum'
    
    input_dict = {
        "input": input,
        "target": target,
        "beta": beta,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.ones((4, 3)).numpy()
    target = torch.zeros((4, 3)).numpy()
    beta = 2.0
    size_average = True
    reduce = False
    reduction = 'none'
    
    input_dict = {
        "input": input,
        "target": target,
        "beta": beta,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.SmoothL1Loss"] = smoothl1loss_inputs()

import torch, copy

def softmax_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    input_dict = {
        "input": input,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn(4, 5, 6).numpy()
    input_dict = {
        "input": input,
        "dim": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0, 2.0],
                         [3.0, 4.0]]).numpy()
    input_dict = {
        "input": input,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]],
                         [[7.0, 8.0, 9.0],
                          [10.0, 11.0, 12.0]]]).numpy()
    input_dict = {
        "input": input,
        "dim": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(3).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.ones((2, 2, 2)).numpy()
    input_dict = {
        "input": input,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(3, 4).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[[1.0, 2.0],
                          [3.0, 4.0],
                          [5.0, 6.0]]]).numpy()
    input_dict = {
        "input": input,
        "dim": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.Softmax"] = softmax_inputs()

import torch, copy

def softmin_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    input_dict = {
        "input": input,
        "dim": 0
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    input_dict = {
        "input": input,
        "dim": 1
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn(5).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]]).numpy()
    input_dict = {
        "input": input,
        "dim": 1
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((3, 4, 5)).numpy()
    input_dict = {
        "input": input,
        "dim": 2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([0.0, 1.0, 2.0]).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(4, 3).numpy()
    input_dict = {
        "input": input,
        "dim": 1
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.5, 1.5]).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.Softmin"] = softmin_inputs()

import torch, copy

def softshrink_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    lambd = 0.5
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    lambd = 0.5
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((2, 3)).numpy()
    lambd = 0.5
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0, -2.0], [3.0, -4.0]]).numpy()
    lambd = 0.5
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.0]).numpy()
    lambd = 0.5
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(1).numpy()
    lambd = 0.5
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([2.0, -2.0]).numpy()
    lambd = 1.0
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.5, -1.5]).numpy()
    lambd = 0.3
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.1, 0.2]).numpy()
    lambd = 0.0
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0, 2.0]).numpy()
    lambd = 0.5
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.Softshrink"] = softshrink_inputs()

import torch, copy

def celu_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    alpha = 1.0 # float
    inplace = False # boolean
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    alpha = 1.5 # float
    inplace = True # boolean
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    alpha = 0.5 # float
    inplace = False # boolean
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    alpha = 2.0 # float
    inplace = False # boolean
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.0]).numpy()
    alpha = 1.0 # float
    inplace = True # boolean
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((1, 2, 3)).numpy()
    alpha = 0.75 # float
    inplace = False # boolean
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[-1.0, 2.0, -3.0]]).numpy()
    alpha = 1.0 # float
    inplace = True # boolean
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0, -2.0, 3.0]).numpy()
    alpha = 2.5 # float
    inplace = False # boolean
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.ones((3, 2)).numpy()
    alpha = 0.5 # float
    inplace = True # boolean
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([0.5, -0.5, 1.5]).numpy()
    alpha = 1.0 # float
    inplace = False # boolean
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.celu"] = celu_inputs()

import torch, copy

def hardshrink_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, -2.0, 3.0]).numpy()   # tensor
    lambd = 0.5 # float
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    lambd = 0.5
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    lambd = 0.5
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([0.0, 0.0, 0.0]).numpy()
    lambd = 0.5
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1.0, 2.0, 3.0, -4.0]).numpy()
    lambd = 0.5
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.5, -2.5, 3.5]).numpy()
    lambd = 0.5
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[1.0, -2.0], [3.0, -4.0]]).numpy()
    lambd = 0.5
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0]).numpy()
    lambd = 0.5
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0]).numpy()
    lambd = 0.5
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1.0, 2.0, -3.0]).numpy()
    lambd = 0.5
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.hardshrink"] = hardshrink_inputs()

import torch, copy

def hardswish_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    inplace = False  # boolean
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()   # tensor
    inplace = False  # boolean
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((2, 3)).numpy()
    inplace = False  # boolean
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([0.0, 1.5, -1.5]).numpy()   # tensor
    inplace = False  # boolean
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([3.0, -3.0, 0.0]).numpy()   # tensor
    inplace = False  # boolean
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((1, 2, 3)).numpy()
    inplace = False  # boolean
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([-4.0, -5.0]).numpy()   # tensor
    inplace = False  # boolean
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([4.0, 5.0]).numpy()   # tensor
    inplace = False  # boolean
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.0, 1.0, 2.0, 3.0]).numpy()   # tensor
    inplace = False  # boolean
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([0.5, -0.5]).numpy()   # tensor
    inplace = False  # boolean
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.hardswish"] = hardswish_inputs()

import torch, copy

def kl_div_inputs():
    list_of_inputs = []
    
    # Input 1
    input = torch.tensor([0.1, 0.2, 0.3]).numpy()
    target = torch.tensor([0.1, 0.2, 0.3]).numpy()
    reduction = 'none'
    log_target = False
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    target = torch.tensor([0.1, 0.2, 0.3]).numpy()
    reduction = 'none'
    log_target = False
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    target = torch.tensor([[0.1, 0.2], [0.3, 0.4]]).numpy()
    reduction = 'none'
    log_target = False
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input = torch.tensor([0.1, 0.2, 0.3]).numpy()
    target = torch.tensor([0.1, 0.2, 0.3]).numpy()
    reduction = 'mean'
    log_target = False
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input = torch.tensor([0.1, 0.2, 0.3]).numpy()
    target = torch.tensor([0.1, 0.2, 0.3]).numpy()
    reduction = 'sum'
    log_target = False
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input = torch.tensor([0.1, 0.2, 0.3]).numpy()
    target = torch.tensor([0.1, 0.2, 0.3]).numpy()
    reduction = 'none'
    log_target = True
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    target = torch.tensor([[0.1, 0.2], [0.3, 0.4]]).numpy()
    reduction = 'mean'
    log_target = True
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    target = torch.tensor([[0.1, 0.2], [0.3, 0.4]]).numpy()
    reduction = 'sum'
    log_target = True
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    target = torch.tensor([0.1, 0.2, 0.3]).numpy()
    reduction = 'none'
    log_target = True
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input = torch.tensor([0.1, 0.2, 0.3]).numpy()
    target = torch.tensor([0.1, 0.2, 0.3]).numpy()
    reduction = 'mean'
    log_target = True
    
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "log_target": log_target
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.kl_div"] = kl_div_inputs()

import torch, copy

def pdist_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()   # tensor
    p = 2.0  # float
    
    input_dict = {
        "input": input,
        "p": p
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()   # tensor
    p = 1.0  # float
    
    input_dict = {
        "input": input,
        "p": p
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[0.0, 0.0], [1.0, 1.0]]).numpy()   # tensor
    p = 0.0  # float
    
    input_dict = {
        "input": input,
        "p": p
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0], [2.0], [3.0]]).numpy()   # tensor
    p = 3.0  # float
    
    input_dict = {
        "input": input,
        "p": p
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()   # tensor
    p = 2.5  # float
    
    input_dict = {
        "input": input,
        "p": p
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()   # tensor
    p = 0.5  # float
    
    input_dict = {
        "input": input,
        "p": p
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]).numpy()   # tensor
    p = 1.5  # float
    
    input_dict = {
        "input": input,
        "p": p
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]).numpy()   # tensor
    p = 2.0  # float
    
    input_dict = {
        "input": input,
        "p": p
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).numpy()   # tensor
    p = 0.0  # float
    
    input_dict = {
        "input": input,
        "p": p
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]).numpy()   # tensor
    p = 1.0  # float
    
    input_dict = {
        "input": input,
        "p": p
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.pdist"] = pdist_inputs()

import torch, copy

def relu6_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([-1.0, 0.0, 5.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((2, 3)).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.5, 2.5, 3.5]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.ones((3, 4)).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([-1.0, 0.0, 6.0, 7.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.relu6"] = relu6_inputs()

import torch, copy

def selu_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[1.0, -2.0, 3.0],
                         [4.0, -5.0, 6.0]]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.ones((1, 4)).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.0, 1.0, -1.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[0.0, 1.0],
                         [2.0, 3.0]]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.5, -1.5, 2.5]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.ones((3, 2, 4)).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-0.5, 0.5, -1.5]).numpy()
    input_dict = {
        "input": input,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.selu"] = selu_inputs()

import torch, copy

def numel_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.zeros((4, 4)).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((2, 3, 4)).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(1, 2, 3, 4, 5).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.zeros((3, 3, 3)).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((1, 1, 1, 1)).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[[[1, 2]]]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.zeros((5, 5, 5, 5, 5)).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.ones((2, 2, 2, 2, 2, 2)).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.numel"] = numel_inputs()

import torch, copy

def permute_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.randn(2, 3, 5).numpy()
    dims = (2, 0, 1)
    input_dict = {
        "input": input,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.randn(1, 4, 6).numpy()
    dims = (2, 1, 0)
    input_dict = {
        "input": input,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn(3, 2, 4, 5).numpy()
    dims = (3, 0, 2, 1)
    input_dict = {
        "input": input,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(2, 3, 4, 5, 6).numpy()
    dims = (4, 3, 2, 1, 0)
    input_dict = {
        "input": input,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.randn(1, 2, 3).numpy()
    dims = (1, 0, 2)
    input_dict = {
        "input": input,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(4, 2, 3, 5).numpy()
    dims = (0, 2, 1, 3)
    input_dict = {
        "input": input,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(5, 6, 7).numpy()
    dims = (2, 1, 0)
    input_dict = {
        "input": input,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(2, 3, 4, 5, 6, 7).numpy()
    dims = (5, 4, 3, 2, 1, 0)
    input_dict = {
        "input": input,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(1, 1, 1).numpy()
    dims = (0, 1, 2)
    input_dict = {
        "input": input,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(2, 3, 4, 5).numpy()
    dims = (3, 2, 1, 0)
    input_dict = {
        "input": input,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.permute"] = permute_inputs()

import torch, copy

def positive_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.zeros((1, 4)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(5).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(3, 4).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([-0.5, 0.5]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(2, 2, 2).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.positive"] = positive_inputs()

import torch, copy

def sin_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    out = torch.zeros(3).numpy()    # tensor

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    out = torch.zeros(6).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    input = torch.randn(4).numpy()
    out = torch.zeros(4).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    out = torch.zeros(3).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    input = torch.tensor([0.0]).numpy()
    out = torch.zeros(1).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    input = torch.randn(1).numpy()
    out = torch.zeros(1).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    input = torch.ones((3, 2)).numpy()
    out = torch.zeros(6).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    input = torch.tensor([0.5, 1.5, 2.5]).numpy()
    out = torch.zeros(3).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    input = torch.randn(5).numpy()
    out = torch.zeros(5).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    input = torch.ones((1, 4)).numpy()
    out = torch.zeros(4).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.sin"] = sin_inputs()

import torch, copy

def i0e_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0.1, 0.2, 0.3]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.zeros((3, 4)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.5, 2.5, 3.5]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((1, 2, 3)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([0.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([10.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.ones((5, 6)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.special.i0e"] = i0e_inputs()

import torch, copy

def i1e_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([-1.0]).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.zeros((3, 4)).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([0.0]).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.ones((1, 2, 3)).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.5, 1.5, 2.5]).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1e-6, 1e-5, 1e-4]).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.special.i1e"] = i1e_inputs()

import torch, copy

def polygamma_inputs():
    list_of_inputs = []
    
    # Input 1
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    n = 0
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input = torch.ones((2, 3)).numpy()
    n = 1
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input = torch.tensor([0.5, 1.5, 2.5]).numpy()
    n = 2
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input = torch.ones((3, 4)).numpy()
    n = 3
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    n = 4
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input = torch.tensor([1.0]).numpy()
    n = 5
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input = torch.ones((1, 2)).numpy()
    n = 6
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input = torch.tensor([0.1, 0.2]).numpy()
    n = 7
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input = torch.ones((4, 5)).numpy()
    n = 8
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input = torch.tensor([2.0, 3.0]).numpy()
    n = 9
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.special.polygamma"] = polygamma_inputs()

import torch, copy

def sinc_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([0.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((2, 3)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((3, 4)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([0.5, 1.5, 2.5]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([-0.5, -1.5, -2.5]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.ones((1, 2, 3)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.special.sinc"] = sinc_inputs()

import torch, copy

def xlog1py_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    tensor1 = torch.tensor([0.1, 0.2, 0.3]).numpy() # tensor
    tensor2 = torch.tensor([10.0, 20.0, 30.0]).numpy()  # tensor
    
    input_dict = {
        "x": input,
        "y": tensor1
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    tensor1 = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]]).numpy()
    tensor2 = torch.tensor([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6]]).numpy()
    
    input_dict = {
        "x": input,
        "y": tensor1
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0.0]).numpy()
    tensor1 = torch.tensor([0.0]).numpy()
    tensor2 = torch.tensor([1.0]).numpy()
    
    input_dict = {
        "x": input,
        "y": tensor1
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-1.0, -2.0]).numpy()
    tensor1 = torch.tensor([0.5, 0.7]).numpy()
    tensor2 = torch.tensor([2.0, 3.0]).numpy()
    
    input_dict = {
        "x": input,
        "y": tensor1
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1.0]).numpy()
    tensor1 = torch.tensor([2.0]).numpy()
    tensor2 = torch.tensor([3.0]).numpy()
    
    input_dict = {
        "x": input,
        "y": tensor1
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((4, 5)).numpy()
    tensor1 = torch.ones((4, 5)).numpy()
    tensor2 = torch.ones((4, 5)).numpy()
    
    input_dict = {
        "x": input,
        "y": tensor1
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    tensor1 = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    tensor2 = torch.tensor([10.0, 20.0, 30.0, 40.0]).numpy()
    
    input_dict = {
        "x": input,
        "y": tensor1
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([0.5]).numpy()
    tensor1 = torch.tensor([0.5]).numpy()
    tensor2 = torch.tensor([0.5]).numpy()
    
    input_dict = {
        "x": input,
        "y": tensor1
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    tensor1 = torch.tensor([0.5, 0.7, 0.9]).numpy()
    tensor2 = torch.tensor([0.1, 0.2, 0.3]).numpy()
    
    input_dict = {
        "x": input,
        "y": tensor1
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    tensor1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    tensor2 = torch.tensor([0.1, 0.2, 0.3]).numpy()
    
    input_dict = {
        "x": input,
        "y": tensor1
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.special.xlog1py"] = xlog1py_inputs()

import torch, copy

def sqrt_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([4.0, 9.0, 16.0]).numpy()  # tensor
    out = torch.zeros(3).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([-4.0, -9.0, -16.0]).numpy()  # tensor
    out = torch.zeros(3).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0.0]).numpy()  # tensor
    out = torch.zeros(1).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((2, 3)).numpy()  # tensor
    out = torch.zeros((2, 3)).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.randn(4).numpy()  # tensor
    out = torch.zeros(4).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0]).numpy()  # tensor
    out = torch.zeros(5).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([0.25, 0.5, 0.75]).numpy()  # tensor
    out = torch.zeros(3).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.zeros((3, 4)).numpy()  # tensor
    out = torch.zeros((3, 4)).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([100.0]).numpy()  # tensor
    out = torch.zeros(1).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(3).numpy()  # tensor
    out = torch.zeros(3).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.sqrt"] = sqrt_inputs()

