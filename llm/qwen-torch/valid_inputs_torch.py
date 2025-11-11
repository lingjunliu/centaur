generated_inputs = {}
import torch, copy

def addcdiv_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    tensor1 = torch.tensor([0.1, 0.2, 0.3]).numpy() # tensor
    tensor2 = torch.tensor([10.0, 20.0, 30.0]).numpy()  # tensor
    value = 2.0 # float
    out = torch.zeros((3,)).numpy()    # tensor

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    tensor1 = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]]).numpy()
    tensor2 = torch.tensor([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6]]).numpy()
    value = 0.5
    out = torch.zeros((2, 3)).numpy()

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    tensor1 = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]).numpy()
    tensor2 = torch.tensor([[2.0, 4.0], [6.0, 8.0]]).numpy()
    value = -1.5
    out = torch.zeros((2, 2)).numpy()

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([0.5]).numpy()
    tensor1 = torch.tensor([1.0]).numpy()
    tensor2 = torch.tensor([2.0]).numpy()
    value = 3.0
    out = torch.zeros((1,)).numpy()

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.zeros((3, 2)).numpy()
    tensor1 = torch.ones((3, 2)).numpy()
    tensor2 = torch.ones((3, 2)).numpy()
    value = 0.0
    out = torch.zeros((3, 2)).numpy()

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(1, 3).numpy()
    tensor1 = torch.randn(3, 1).numpy()
    tensor2 = torch.randn(1, 3).numpy()
    value = 0.1
    out = torch.zeros((1, 3)).numpy()

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]]).numpy()
    tensor1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    tensor2 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    value = 2.5
    out = torch.zeros((2, 3)).numpy()

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0]).numpy()
    tensor1 = torch.tensor([2.0]).numpy()
    tensor2 = torch.tensor([0.5]).numpy()
    value = 1.0
    out = torch.zeros((1,)).numpy()

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[-2.0, -3.0], [-4.0, -5.0]]).numpy()
    tensor1 = torch.tensor([1.0, 2.0]).numpy()
    tensor2 = torch.tensor([2.0, 4.0]).numpy()
    value = -0.5
    out = torch.zeros((2, 2)).numpy()

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0, 2.0]).numpy()
    tensor1 = torch.tensor([3.0, 4.0]).numpy()
    tensor2 = torch.tensor([5.0, 6.0]).numpy()
    value = 0.75
    out = torch.zeros((2,)).numpy()

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.addcdiv"] = addcdiv_inputs()

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
    input = torch.tensor([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]]).numpy()
    dim = 0
    keepdim = True
    out = torch.zeros(3).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn((3, 4, 5)).numpy()
    dim = 2
    keepdim = False
    out = torch.zeros(3).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dim = 0
    keepdim = True
    out = torch.zeros(2).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]).numpy()
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
    
    # Input 6, valid
    input = torch.tensor([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    dim = 0
    keepdim = False
    out = torch.zeros(3).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[[1.0, -2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dim = 2
    keepdim = True
    out = torch.zeros(2).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn((2, 3, 4, 5)).numpy()
    dim = 3
    keepdim = False
    out = torch.zeros(2).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]]).numpy()
    dim = 1
    keepdim = True
    out = torch.zeros(3).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]]).numpy()
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

generated_inputs["torch.amin"] = amin_inputs()

import torch, copy

def any_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([0, 1, 2]).numpy()
    out = torch.tensor([True]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[False, True], [True, False]]).numpy()
    out = torch.tensor([True]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0, 0, 0]).numpy()
    out = torch.tensor([False]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[0, 1], [2, 3]]).numpy()
    out = torch.tensor([True]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1, 0, 0]).numpy()
    out = torch.tensor([True]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([-1, 0, 1]).numpy()
    out = torch.tensor([True]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([-1, -2, -3]).numpy()
    out = torch.tensor([False]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[True, False], [False, True]]).numpy()
    out = torch.tensor([True]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[0, 0], [0, 1]]).numpy()
    out = torch.tensor([True]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[0, 0], [0, 0]]).numpy()
    out = torch.tensor([False]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.any_1"] = any_inputs()

import torch, copy

def any_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([0, 1, 2], dtype=torch.int32).numpy()
    dim = 0
    keepdim = False
    out = torch.empty(1, dtype=torch.bool).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[False, True], [True, False]], dtype=torch.bool).numpy()
    dim = 0
    keepdim = False
    out = torch.empty(1, dtype=torch.bool).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0, 0, 0], dtype=torch.int32).numpy()
    dim = 0
    keepdim = False
    out = torch.empty(1, dtype=torch.bool).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32).numpy()
    dim = 1
    keepdim = False
    out = torch.empty(2, dtype=torch.bool).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[True, False], [False, True]], dtype=torch.bool).numpy()
    dim = 1
    keepdim = True
    out = torch.empty(1, 1, dtype=torch.bool).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0, 1, 2], dtype=torch.int32).numpy()
    dim = None
    keepdim = False
    out = torch.empty(1, dtype=torch.bool).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32).numpy()
    dim = None
    keepdim = True
    out = torch.empty(1, 1, dtype=torch.bool).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([0, 0, 0], dtype=torch.int32).numpy()
    dim = 0
    keepdim = True
    out = torch.empty(1, dtype=torch.bool).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[False, False], [True, True]], dtype=torch.bool).numpy()
    dim = 0
    keepdim = True
    out = torch.empty(1, 1, dtype=torch.bool).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    dim = None
    keepdim = False
    out = torch.empty(1, dtype=torch.bool).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.any_2"] = any_inputs()

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
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[0.1, 0.2], [0.3, 0.4]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[-1.0, -2.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.ones(5).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.zeros((2, 2, 2)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.are_deterministic_algorithms_enabled"] = are_deterministic_algorithms_enabled_inputs()

import torch, copy

def bincount_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([0, 1, 2, 3, 4]).numpy()   # tensor
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).numpy() # tensor
    minlength = 5   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([0, 0, 1, 1, 2]).numpy()   # tensor
    weights = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]).numpy() # tensor
    minlength = 3   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([7, 6, 5, 4, 3]).numpy()   # tensor
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy() # tensor
    minlength = 8   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1, 2, 3, 4, 5]).numpy()   # tensor
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).numpy() # tensor
    minlength = 6   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0]).numpy()   # tensor
    weights = torch.tensor([1.0]).numpy() # tensor
    minlength = 1   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([10, 10, 10]).numpy()   # tensor
    weights = torch.tensor([2.0, 3.0, 4.0]).numpy() # tensor
    minlength = 11   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([0, 1, 2, 3, 4, 5]).numpy()   # tensor
    weights = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).numpy() # tensor
    minlength = 6   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([3, 3, 3, 3]).numpy()   # tensor
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy() # tensor
    minlength = 4   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0, 1, 2, 3, 4, 5, 6]).numpy()   # tensor
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]).numpy() # tensor
    minlength = 7   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([2, 2, 2, 2, 2]).numpy()   # tensor
    weights = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]).numpy() # tensor
    minlength = 3   # integer
    
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.bincount_1"] = bincount_inputs()

import torch, copy

def bitwise_left_shift_inputs():
    list_of_inputs = []
    
    # Input 1
    input = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    other = torch.tensor([0, 1, 2], dtype=torch.int32).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input = torch.tensor([10, 20, 30], dtype=torch.int32).numpy()
    other = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input = torch.ones((2, 3), dtype=torch.int32).numpy()
    other = torch.zeros((2, 3), dtype=torch.int32).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input = torch.tensor([100, 200], dtype=torch.int32).numpy()
    other = torch.tensor([1, 2], dtype=torch.int32).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input = torch.tensor([0, 1, 2, 3], dtype=torch.int32).numpy()
    other = torch.tensor([3, 2, 1, 0], dtype=torch.int32).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input = torch.tensor([1000, 2000], dtype=torch.int32).numpy()
    other = torch.tensor([2, 3], dtype=torch.int32).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32).numpy()
    other = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input = torch.tensor([5, 4, 3], dtype=torch.int32).numpy()
    other = torch.tensor([2, 1, 0], dtype=torch.int32).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input = torch.zeros((3, 4), dtype=torch.int32).numpy()
    other = torch.ones((3, 4), dtype=torch.int32).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input = torch.tensor([10, 20], dtype=torch.int32).numpy()
    other = torch.tensor([3, 4], dtype=torch.int32).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.bitwise_left_shift"] = bitwise_left_shift_inputs()

import torch, copy

def complex_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    real = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    imag = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32).numpy()
    out = torch.tensor([1.0, 2.0, 3.0], dtype=torch.complex64).numpy()
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    real = torch.tensor([1.0, 2.0], dtype=torch.float64).numpy()
    imag = torch.tensor([3.0, 4.0], dtype=torch.float64).numpy()
    out = torch.tensor([1.0, 2.0], dtype=torch.complex128).numpy()
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    real = torch.tensor([1.0], dtype=torch.float32).numpy()
    imag = torch.tensor([2.0], dtype=torch.float32).numpy()
    out = torch.tensor([1.0], dtype=torch.complex64).numpy()
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    real = torch.tensor([-1.0, -2.0], dtype=torch.float32).numpy()
    imag = torch.tensor([-3.0, -4.0], dtype=torch.float32).numpy()
    out = torch.tensor([1.0, 2.0], dtype=torch.complex64).numpy()
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    real = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64).numpy()
    imag = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64).numpy()
    out = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.complex128).numpy()
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    real = torch.tensor([0.0], dtype=torch.float32).numpy()
    imag = torch.tensor([0.0], dtype=torch.float32).numpy()
    out = torch.tensor([1.0], dtype=torch.complex64).numpy()
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    real = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    imag = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32).numpy()
    out = torch.tensor([1.0, 2.0, 3.0], dtype=torch.complex64).numpy()
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    real = torch.tensor([1.0], dtype=torch.float64).numpy()
    imag = torch.tensor([2.0], dtype=torch.float64).numpy()
    out = torch.tensor([1.0], dtype=torch.complex128).numpy()
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    real = torch.tensor([1.0, 2.0], dtype=torch.float32).numpy()
    imag = torch.tensor([3.0, 4.0], dtype=torch.float32).numpy()
    out = torch.tensor([1.0, 2.0], dtype=torch.complex64).numpy()
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    real = torch.tensor([-1.0, -2.0], dtype=torch.float64).numpy()
    imag = torch.tensor([-3.0, -4.0], dtype=torch.float64).numpy()
    out = torch.tensor([1.0, 2.0], dtype=torch.complex128).numpy()
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.complex_1"] = complex_inputs()

import torch, copy

def conj_physical_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0+2.0j, 3.0+4.0j]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[1.0+2.0j, 3.0+4.0j], [5.0+6.0j, 7.0+8.0j]]).numpy()
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
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0+2.0j, 3.0+4.0j, 5.0+6.0j]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[1.0+2.0j, 3.0+4.0j], [5.0+6.0j, 7.0+8.0j]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0-2.0j, 3.0-4.0j]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[1.0-2.0j, 3.0-4.0j], [5.0-6.0j, 7.0-8.0j]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.conj_physical"] = conj_physical_inputs()

import torch, copy

def cumprod_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    dim = 0   # integer
    dtype = None   # dtype
    out = torch.empty(3).numpy()    # tensor

    input_dict = {
        "input": input,
        "dim": dim,
        "dtype": dtype,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((3, 4)).numpy()
    dim = 1   # integer
    dtype = None   # dtype
    out = torch.empty((3, 4)).numpy()

    input_dict = {
        "input": input,
        "dim": dim,
        "dtype": dtype,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn(5).numpy()
    dim = 0   # integer
    dtype = None   # dtype
    out = torch.empty(5).numpy()

    input_dict = {
        "input": input,
        "dim": dim,
        "dtype": dtype,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[-1.0, 2.0], [3.0, -4.0]]).numpy()
    dim = 0   # integer
    dtype = None   # dtype
    out = torch.empty((2, 2)).numpy()

    input_dict = {
        "input": input,
        "dim": dim,
        "dtype": dtype,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.zeros((2, 3, 4)).numpy()
    dim = 2   # integer
    dtype = None   # dtype
    out = torch.empty((2, 3, 4)).numpy()

    input_dict = {
        "input": input,
        "dim": dim,
        "dtype": dtype,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.5, -0.5, 1.0]).numpy()
    dim = 0   # integer
    dtype = None   # dtype
    out = torch.empty(3).numpy()

    input_dict = {
        "input": input,
        "dim": dim,
        "dtype": dtype,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((10,)).numpy()
    dim = 0   # integer
    dtype = None   # dtype
    out = torch.empty(10).numpy()

    input_dict = {
        "input": input,
        "dim": dim,
        "dtype": dtype,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    dim = 0   # integer
    dtype = None   # dtype
    out = torch.empty(4).numpy()

    input_dict = {
        "input": input,
        "dim": dim,
        "dtype": dtype,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(7).numpy()
    dim = 0   # integer
    dtype = None   # dtype
    out = torch.empty(7).numpy()

    input_dict = {
        "input": input,
        "dim": dim,
        "dtype": dtype,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    dim = 1   # integer
    dtype = None   # dtype
    out = torch.empty((2, 2)).numpy()

    input_dict = {
        "input": input,
        "dim": dim,
        "dtype": dtype,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.cumprod"] = cumprod_inputs()

import torch, copy

def dequantize_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    tensor = torch.tensor([1, 2, 3]).numpy()
    input_dict = {
        "tensor": tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    tensor = torch.tensor([[1, 2], [3, 4]]).numpy()
    input_dict = {
        "tensor": tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    tensor = torch.tensor([0.1, 0.2, 0.3]).numpy()
    input_dict = {
        "tensor": tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    tensor = torch.tensor([-1, -2, -3]).numpy()
    input_dict = {
        "tensor": tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    tensor = torch.tensor([0.0]).numpy()
    input_dict = {
        "tensor": tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    tensor = torch.tensor([1, 2, 3, 4, 5]).numpy()
    input_dict = {
        "tensor": tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    input_dict = {
        "tensor": tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    tensor = torch.tensor([1.0]).numpy()
    input_dict = {
        "tensor": tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    tensor = torch.tensor([0.5, 1.5, 2.5]).numpy()
    input_dict = {
        "tensor": tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    tensor = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "tensor": tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.dequantize_1"] = dequantize_inputs()

import torch, copy

def diag_embed_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    k = 0   # integer
    
    input_dict = {
        "input": input,
        "k": k
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()   # tensor
    k = 1   # integer
    
    input_dict = {
        "input": input,
        "k": k
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    k = -1   # integer
    
    input_dict = {
        "input": input,
        "k": k
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((3, 4)).numpy()   # tensor
    k = 0   # integer
    
    input_dict = {
        "input": input,
        "k": k
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.ones((2, 3)).numpy()   # tensor
    k = 2   # integer
    
    input_dict = {
        "input": input,
        "k": k
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0, 2.0]).numpy()   # tensor
    k = -2   # integer
    
    input_dict = {
        "input": input,
        "k": k
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((4, 5)).numpy()   # tensor
    k = -1   # integer
    
    input_dict = {
        "input": input,
        "k": k
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()   # tensor
    k = 3   # integer
    
    input_dict = {
        "input": input,
        "k": k
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    k = 2   # integer
    
    input_dict = {
        "input": input,
        "k": k
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.ones((1, 6)).numpy()   # tensor
    k = -2   # integer
    
    input_dict = {
        "input": input,
        "k": k
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.diag_embed"] = diag_embed_inputs()

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
    other = torch.tensor([1, 2, 3], dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((2, 3)).numpy()
    other = torch.ones((2, 3)).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.zeros((1, 4)).numpy()
    other = torch.zeros((1, 4)).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([3, torch.nan]).numpy()
    other = torch.tensor([3, torch.nan]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1, 2, 3], dtype=torch.int64).numpy()
    other = torch.tensor([1, 2, 3], dtype=torch.float64).numpy()
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
    input = torch.tensor([0]).numpy()
    other = torch.tensor([0]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1, 2, 3, 4, 5]).numpy()
    other = torch.tensor([1, 2, 3, 4, 5]).numpy()
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
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out = torch.zeros((3,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    out = torch.zeros((2, 3)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    input = torch.tensor([-1.5, -2.7, -3.9]).numpy()
    out = torch.zeros((3,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    input = torch.zeros((1, 2, 3)).numpy()
    out = torch.zeros((1, 2, 3)).numpy()

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
    input = torch.tensor([1.9, 2.1, 3.8]).numpy()
    out = torch.zeros((3,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    input = torch.tensor([-1.9, -2.1, -3.8]).numpy()
    out = torch.zeros((3,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    input = torch.ones((3, 2)).numpy()
    out = torch.zeros((3, 2)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    input = torch.tensor([1.0, -2.0, 3.0]).numpy()
    out = torch.zeros((3,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    input = torch.tensor([-1.0, 2.0, -3.0]).numpy()
    out = torch.zeros((3,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.fix"] = fix_inputs()

import torch, copy

def flipud_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[1, 2], [3, 4]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1, 2, 3, 4]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1, 2], [3, 4], [5, 6]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[0, 1], [2, 3], [4, 5]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[-1, -2], [-3, -4]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.flipud_1"] = flipud_inputs()

import torch, copy

def floor_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([-0.8166, 1.5308, -0.2530, -0.2091]).numpy()
    out = torch.tensor([-1., 1., -1., -1.]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([0.0]).numpy()
    out = torch.tensor([0.0]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([1.7, 2.9, 3.1]).numpy()
    out = torch.tensor([1., 2., 3.]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-1.5, -2.7, -3.9]).numpy()
    out = torch.tensor([-2., -3., -4.]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([10.0, 20.0, 30.0]).numpy()
    out = torch.tensor([10., 20., 30.]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((2, 3)).numpy()
    out = torch.ones((2, 3)).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.zeros((3, 4)).numpy()
    out = torch.zeros((3, 4)).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[-1.5, 2.7], [-3.9, 4.1]]).numpy()
    out = torch.tensor([-2., 2., -4., 4.]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.0]).numpy()
    out = torch.tensor([0.0]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-0.99, -1.01]).numpy()
    out = torch.tensor([-2., -2.]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.floor"] = floor_inputs()

import torch, copy

def floor_divide_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([10.0, 20.0, 30.0]).numpy()
    other = torch.tensor([3.0, 4.0, 5.0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    other = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([100.0]).numpy()
    other = torch.tensor([7.0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.zeros((3, 4)).numpy()
    other = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0]]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([-10.0, -20.0, -30.0]).numpy()
    other = torch.tensor([3.0, 4.0, 5.0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([10.0, 20.0, 30.0]).numpy()
    other = torch.tensor([-3.0, -4.0, -5.0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((1, 2)).numpy()
    other = torch.tensor([[1.0, 2.0]]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([0.0]).numpy()
    other = torch.tensor([1.0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.zeros((4, 5)).numpy()
    other = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                        [6.0, 7.0, 8.0, 9.0, 10.0],
                        [11.0, 12.0, 13.0, 14.0, 15.0],
                        [16.0, 17.0, 18.0, 19.0, 20.0]]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([5.0]).numpy()
    other = torch.tensor([2.0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.floor_divide"] = floor_divide_inputs()

import torch, copy

def ge_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[1, 2], [3, 4]]).numpy()
    other = torch.tensor([[1, 1], [4, 4]]).numpy()
    out = torch.zeros((2, 2)).numpy()

    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other = torch.tensor([1.0, 1.0, 2.0]).numpy()
    out = torch.zeros((3,)).numpy()

    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((2, 3)).numpy()
    other = torch.zeros((2, 3)).numpy()
    out = torch.zeros((2, 3)).numpy()

    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-1, 0, 1]).numpy()
    other = torch.tensor([-1, -1, 0]).numpy()
    out = torch.zeros((3,)).numpy()

    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1, 2, 3]).numpy()
    other = torch.tensor([0, 1, 2]).numpy()
    out = torch.zeros((3,)).numpy()

    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([-10, -5]).numpy()
    other = torch.tensor([-10, -5]).numpy()
    out = torch.zeros((2,)).numpy()

    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[0, 1], [2, 3]]).numpy()
    other = torch.tensor([[0, 1], [2, 3]]).numpy()
    out = torch.zeros((2, 2)).numpy()

    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    other = torch.tensor([[[1, 1], [4, 4]], [[0, 0], [7, 7]]]).numpy()
    out = torch.zeros((2, 2, 2)).numpy()

    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    other = torch.tensor([[0, 1], [2, 3]]).numpy()
    out = torch.zeros((2, 2)).numpy()

    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    other = torch.tensor([[1, 1, 1], [1, 1, 1]]).numpy()
    out = torch.zeros((2, 3)).numpy()

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
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1.5+2.7j, 3.1+4.2j], dtype=torch.complex64).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0+0j, 1+1j, 2+2j, 3+3j], dtype=torch.complex64).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1+2j, 3+4j, 5+6j, 7+8j], dtype=torch.complex128).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.5+0.5j, 1.5+1.5j], dtype=torch.complex128).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((3, 4), dtype=torch.complex64).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.zeros((2, 2, 3), dtype=torch.complex64).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn((5, 1), dtype=torch.complex128).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.0+0.0j], dtype=torch.complex64).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1-2j, -3-4j, -5-6j], dtype=torch.complex64).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.imag"] = imag_inputs()

import torch, copy

def is_floating_point_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([0.1, 0.2, 0.3]).numpy()
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
    input = torch.tensor([0.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([]).numpy()
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
    input = torch.tensor([0.5]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([0.0]).numpy()
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
    input = torch.tensor([1.0, 2.0, 3.0], requires_grad=True).detach().numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1.0, 2.0, 3.0], requires_grad=False).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((2, 3), requires_grad=True).detach().numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((2, 3), requires_grad=False).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor(1.0, requires_grad=True).detach().numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor(1.0, requires_grad=False).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.zeros((3, 4, 5), requires_grad=True).detach().numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.zeros((3, 4, 5), requires_grad=False).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True).detach().numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], requires_grad=False).numpy()
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
    input = torch.zeros((1, 4, 5)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0, 2.0],
                        [3.0, 4.0]]).numpy()
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
    input = torch.tensor([0.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones(5).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1.0, -2.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.is_storage"] = is_storage_inputs()

import torch, copy

def isreal_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([1+1j, 2+2j, 3+3j]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1+0j, 2+0j, 3+0j]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1, 2+2j, 3]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1+1j, 2+0j, 3+1j]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([-1, -2, -3]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.isreal"] = isreal_inputs()

import torch, copy

def svdvals_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    driver = None
    out = torch.empty(2).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    driver = None
    out = torch.empty(3).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    A = torch.randn((4, 5)).numpy()
    driver = None
    out = torch.empty(5).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    A = torch.randn((3, 3)).numpy()
    driver = None
    out = torch.empty(3).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()
    driver = None
    out = torch.empty(2).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    A = torch.randn((2, 2)).numpy()
    driver = None
    out = torch.empty(2).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    driver = None
    out = torch.empty(3).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    A = torch.randn((5, 4)).numpy()
    driver = None
    out = torch.empty(4).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).numpy()
    driver = None
    out = torch.empty(4).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    A = torch.randn((6, 3)).numpy()
    driver = None
    out = torch.empty(3).numpy()
    
    input_dict = {
        "A": A,
        "driver": driver,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.linalg.svdvals_1"] = svdvals_inputs()

import torch, copy

def logsumexp_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()   # tensor
    dim = 1   # integer
    keepdim = False   # boolean
    out = torch.zeros((2,)).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    dim = 0   # integer
    keepdim = True   # boolean
    out = torch.zeros((1,)).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn((3, 4)).numpy()
    dim = 1   # integer
    keepdim = False   # boolean
    out = torch.zeros((3,)).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]]).numpy()
    dim = 0   # integer
    keepdim = True   # boolean
    out = torch.zeros((1,)).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.ones((2, 3, 4)).numpy()
    dim = 2   # integer
    keepdim = False   # boolean
    out = torch.zeros((2, 3)).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn((5, 2)).numpy()
    dim = 1   # integer
    keepdim = True   # boolean
    out = torch.zeros((5,)).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[-1.0, 2.0], [3.0, -4.0], [5.0, 6.0]]).numpy()
    dim = 0   # integer
    keepdim = False   # boolean
    out = torch.zeros((3,)).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()
    dim = 1   # integer
    keepdim = True   # boolean
    out = torch.zeros((3,)).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn((3, 3, 3)).numpy()
    dim = 2   # integer
    keepdim = False   # boolean
    out = torch.zeros((3, 3)).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    dim = 1   # integer
    keepdim = True   # boolean
    out = torch.zeros((3,)).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.logsumexp"] = logsumexp_inputs()

import torch, copy

def matrix_power_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[2.0, 3.0], [4.0, 5.0]]).numpy()  # tensor
    n = -3  # integer
    
    input_dict = {
        "input": input,
        "n": n
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()  # tensor
    n = 4  # integer
    
    input_dict = {
        "input": input,
        "n": n
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()  # tensor
    n = -5  # integer
    
    input_dict = {
        "input": input,
        "n": n
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()  # tensor
    n = 1  # integer
    
    input_dict = {
        "input": input,
        "n": n
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()  # tensor
    n = 2  # integer
    
    input_dict = {
        "input": input,
        "n": n
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()  # tensor
    n = 3  # integer
    
    input_dict = {
        "input": input,
        "n": n
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).numpy()  # tensor
    n = 0  # integer
    
    input_dict = {
        "input": input,
        "n": n
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()  # tensor
    n = -1  # integer
    
    input_dict = {
        "input": input,
        "n": n
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[2.0, 3.0], [4.0, 5.0]]).numpy()  # tensor
    n = -2  # integer
    
    input_dict = {
        "input": input,
        "n": n
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()  # tensor
    n = 5  # integer
    
    input_dict = {
        "input": input,
        "n": n
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.matrix_power"] = matrix_power_inputs()

import torch, copy

def median_inputs():
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
    input = torch.randn(4, 5).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-1.0, 0.0, 1.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.zeros((3, 4, 5)).numpy()
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
    input = torch.randn(1).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(5, 3).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
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
    input = torch.tensor([[1.0, 2.0, 3.0]]).numpy()
    dim = 0
    keepdim = False
    out = (torch.tensor([0.0]).numpy(), torch.tensor([0]).numpy())
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    dim = 1
    keepdim = True
    out = (torch.tensor([0.0]).numpy(), torch.tensor([0]).numpy())
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[-1.0, 0.0, 1.0, 2.0]]).numpy()
    dim = 0
    keepdim = False
    out = (torch.tensor([0.0]).numpy(), torch.tensor([0]).numpy())
    
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
    keepdim = False
    out = (torch.tensor([0.0]).numpy(), torch.tensor([0]).numpy())
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]).numpy()
    dim = -1
    keepdim = True
    out = (torch.tensor([0.0]).numpy(), torch.tensor([0]).numpy())
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]).numpy()
    dim = 0
    keepdim = False
    out = (torch.tensor([0.0]).numpy(), torch.tensor([0]).numpy())
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[-1.0, 0.0, 1.0, 2.0, 3.0]]).numpy()
    dim = -1
    keepdim = False
    out = (torch.tensor([0.0]).numpy(), torch.tensor([0]).numpy())
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).numpy()
    dim = 1
    keepdim = True
    out = (torch.tensor([0.0]).numpy(), torch.tensor([0]).numpy())
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).numpy()
    dim = 0
    keepdim = True
    out = (torch.tensor([0.0]).numpy(), torch.tensor([0]).numpy())
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]).numpy()
    dim = -1
    keepdim = False
    out = (torch.tensor([0.0]).numpy(), torch.tensor([0]).numpy())
    
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

def minimum_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1, 2, -1]).numpy()
    other = torch.tensor([3, 0, 4]).numpy()
    out = torch.tensor([1, 0, -1]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other = torch.tensor([0.1, 0.2, 0.3]).numpy()
    out = torch.tensor([0.1, 0.2, 0.3]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([-1, -2, -3]).numpy()
    other = torch.tensor([1, 2, 3]).numpy()
    out = torch.tensor([-1, -2, -3]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((2, 3)).numpy()
    other = torch.zeros((2, 3)).numpy()
    out = torch.zeros((2, 3)).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1, 2, 3, 4, 5]).numpy()
    other = torch.tensor([5, 4, 3, 2, 1]).numpy()
    out = torch.tensor([1, 2, 3, 2, 1]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.5, 1.5, 2.5]).numpy()
    other = torch.tensor([0.5, 1.5, 2.5]).numpy()
    out = torch.tensor([0.5, 1.5, 2.5]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([0, 0, 0]).numpy()
    other = torch.tensor([1, 2, 3]).numpy()
    out = torch.tensor([0, 0, 0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1, 2, 3]).numpy()
    other = torch.tensor([0, 0, 0]).numpy()
    out = torch.tensor([0, 0, 0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    out = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([5, 4, 3, 2, 1]).numpy()
    other = torch.tensor([1, 2, 3, 4, 5]).numpy()
    out = torch.tensor([1, 2, 3, 2, 1]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.minimum"] = minimum_inputs()

import torch, copy

def nextafter_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    other = torch.tensor([2.0, 1.0, 4.0]).numpy()  # tensor
    out = torch.zeros(3).numpy()                   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    other = torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]]).numpy()
    out = torch.zeros((2, 3)).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0.5]).numpy()
    other = torch.tensor([-0.5]).numpy()
    out = torch.zeros(1).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-1.0, -2.0]).numpy()
    other = torch.tensor([1.0, 2.0]).numpy()
    out = torch.zeros(2).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1e-10]).numpy()
    other = torch.tensor([1e-5]).numpy()
    out = torch.zeros(1).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.0]).numpy()
    other = torch.tensor([0.0]).numpy()
    out = torch.zeros(1).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([10.0]).numpy()
    other = torch.tensor([20.0]).numpy()
    out = torch.zeros(1).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0]).numpy()
    other = torch.tensor([2.0]).numpy()
    out = torch.zeros(1).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([2.0]).numpy()
    other = torch.tensor([1.0]).numpy()
    out = torch.zeros(1).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-3.0]).numpy()
    other = torch.tensor([-2.0]).numpy()
    out = torch.zeros(1).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nextafter"] = nextafter_inputs()

import torch, copy

def asin_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([-0.5, 0.5]).numpy()   # tensor
    out = torch.zeros(2).numpy()    # tensor

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([-1.0, 0.0, 1.0]).numpy()   # tensor
    out = torch.zeros(3).numpy()    # tensor

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
    input = torch.randn(4).numpy()
    out = torch.zeros(4).numpy()

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
    input = torch.tensor([-0.9, 0.9, -0.1]).numpy()
    out = torch.zeros(3).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    input = torch.ones((3, 2)).numpy()
    out = torch.zeros((3, 2)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    input = torch.randn(1).numpy()
    out = torch.zeros(1).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    input = torch.tensor([0.5]).numpy()
    out = torch.zeros(1).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    input = torch.tensor([-0.8, 0.7, -0.6]).numpy()
    out = torch.zeros(3).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.asin_1"] = asin_inputs()

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
    input = torch.tensor([1, 2, 3]).numpy()    # tensor
    other = torch.tensor([4, 5, 6]).numpy()    # tensor
    out = torch.tensor([4, 10, 6]).numpy()     # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0, 1, 2]).numpy()    # tensor
    other = torch.tensor([3, 4, 5]).numpy()    # tensor
    out = torch.tensor([0, 4, 10]).numpy()     # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([10, 20, 30]).numpy()   # tensor
    other = torch.tensor([5, 6, 7]).numpy()      # tensor
    out = torch.tensor([10, 60, 210]).numpy()    # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([3, 6, 9]).numpy()   # tensor
    other = torch.tensor([2, 4, 8]).numpy()   # tensor
    out = torch.tensor([6, 12, 72]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([100, 200]).numpy()   # tensor
    other = torch.tensor([150, 300]).numpy()   # tensor
    out = torch.tensor([300, 600]).numpy()     # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1, 2, 3, 4]).numpy()   # tensor
    other = torch.tensor([5, 6, 7, 8]).numpy()   # tensor
    out = torch.tensor([5, 6, 21, 8]).numpy()    # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([0, 1, 2]).numpy()   # tensor
    other = torch.tensor([1, 0, 3]).numpy()   # tensor
    out = torch.tensor([0, 0, 6]).numpy()     # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([-1, -2, -3]).numpy()   # tensor
    other = torch.tensor([4, 5, 6]).numpy()      # tensor
    out = torch.tensor([4, 10, 6]).numpy()       # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1, 2, 3]).numpy()   # tensor
    other = torch.tensor([0, 1, 2]).numpy()   # tensor
    out = torch.tensor([0, 2, 6]).numpy()     # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.lcm"] = lcm_inputs()

import torch, copy

def nansum_inputs():
    list_of_inputs = []
    
    # Input 1 - valid
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
    
    # Input 2 - valid
    input = torch.tensor([[1, 2], [3, float('nan')]]).numpy()
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
    
    # Input 3 - valid
    input = torch.tensor([[1, 2], [3, float('nan')]]).numpy()
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
    
    # Input 4 - valid
    input = torch.tensor([1.0, float('nan'), 3.0]).numpy()
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
    
    # Input 5 - valid
    input = torch.tensor([1.0, 2.0, float('nan'), 4.0]).numpy()
    dim = None
    keepdim = False
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, float('nan')]]).numpy()
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
    
    # Input 7 - valid
    input = torch.tensor([[1.0, 2.0], [3.0, float('nan')], [4.0, 5.0]]).numpy()
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
    
    # Input 8 - valid
    input = torch.tensor([1.0, float('nan'), 3.0]).numpy()
    dim = 0
    keepdim = False
    dtype = torch.float64
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - valid
    input = torch.tensor([[1, 2], [3, float('nan')]]).numpy()
    dim = 0
    keepdim = True
    dtype = torch.float32
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, float('nan')], [6.0, 7.0, 8.0]]).numpy()
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
    
    return list_of_inputs

generated_inputs["torch.nansum_2"] = nansum_inputs()

import torch, copy

def clip_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    tensor1 = torch.tensor([0.1, 0.2, 0.3]).numpy()
    tensor2 = torch.tensor([10.0, 20.0, 30.0]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    tensor1 = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]]).numpy()
    tensor2 = torch.tensor([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6]]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    tensor1 = torch.tensor([0.1, 0.2, 0.3]).numpy()
    tensor2 = torch.tensor([10.0, 20.0, 30.0]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((3, 4)).numpy()
    tensor1 = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                            [0.5, 0.6, 0.7, 0.8],
                            [0.9, 1.0, 1.1, 1.2]]).numpy()
    tensor2 = torch.tensor([[10.0, 20.0, 30.0, 40.0],
                            [50.0, 60.0, 70.0, 80.0],
                            [90.0, 100.0, 110.0, 120.0]]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.0]).numpy()
    tensor1 = torch.tensor([0.1]).numpy()
    tensor2 = torch.tensor([10.0]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0, 2.0]).numpy()
    tensor1 = torch.tensor([0.5]).numpy()
    tensor2 = torch.tensor([10.0, 20.0]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((1, 5)).numpy()
    tensor1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).numpy()
    tensor2 = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-1.0, -2.0, -3.0, -4.0]).numpy()
    tensor1 = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    tensor2 = torch.tensor([10.0, 20.0, 30.0, 40.0]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.zeros((2, 2)).numpy()
    tensor1 = torch.tensor([0.5, 0.7]).numpy()
    tensor2 = torch.tensor([0.8, 0.9]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    tensor1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).numpy()
    tensor2 = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]).numpy()
    
    input_dict = {
        "input": input,
        "min": tensor1,
        "max": tensor2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.clip_1"] = clip_inputs()

import torch, copy

def log1p_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([-0.5, 0.5, 1.0])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((2, 3))
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(5)
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.1, 0.2, 0.3])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([-1.0, -0.5, 0.0])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((1, 4))
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(3)
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.0])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.5, 2.5, 3.5])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11, valid
    input = torch.tensor([-0.9, -0.8, -0.7])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12, valid
    input = torch.ones((3, 2))
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.log1p"] = log1p_inputs()

import torch, copy

def lu_unpack_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    LU = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]]).numpy()
    Piv = torch.tensor([3, 2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    LU = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0]]).numpy()
    Piv = torch.tensor([2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    LU = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0],
                      [13.0, 14.0, 15.0, 16.0]]).numpy()
    Piv = torch.tensor([4, 3, 2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    LU = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]]).numpy()
    Piv = torch.tensor([2, 1, 3], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    LU = torch.ones((4, 4)).numpy()
    Piv = torch.tensor([1, 2, 3, 4], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    LU = torch.tensor([[0.0, 1.0],
                      [2.0, 3.0]]).numpy()
    Piv = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    LU = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]]).numpy()
    Piv = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    LU = torch.zeros((3, 3)).numpy()
    Piv = torch.tensor([3, 2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    LU = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0],
                      [7.0, 8.0]]).numpy()
    Piv = torch.tensor([1, 2, 3, 4], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    LU = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]]).numpy()
    Piv = torch.tensor([3, 2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "LU": LU,
        "Piv": Piv
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.lu_unpack"] = lu_unpack_inputs()

import torch, copy

def nansum_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, float('nan'), 4.0]).numpy()   # tensor
    dtype = None
    
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[1, 2], [3, float('nan')]]).numpy()
    dtype = None
    
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([1.0, float('nan'), 3.0]).numpy()
    dtype = None
    
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[[1.0, 2.0], [3.0, float('nan')]], [[4.0, 5.0], [6.0, 7.0]]]).numpy()
    dtype = None
    
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    dtype = None
    
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[-1.0, -2.0, float('nan'), 4.0]]).numpy()
    dtype = None
    
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.0, float('nan'), float('nan'), 4.0]).numpy()
    dtype = None
    
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0, float('nan'), 3.0, 4.0]).numpy()
    dtype = None
    
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[[1.0, float('nan')], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dtype = None
    
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([float('nan'), float('nan'), float('nan')]).numpy()
    dtype = None
    
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nansum_1"] = nansum_inputs()

import torch, copy

def batchnorm1d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic case with learnable parameters
    input = torch.randn(20, 100).numpy()
    input_dict = {
        "num_features": 100,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Without learnable parameters
    input = torch.randn(10, 50).numpy()
    input_dict = {
        "num_features": 50,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - With different momentum value
    input = torch.randn(15, 200).numpy()
    input_dict = {
        "num_features": 200,
        "eps": 1e-5,
        "momentum": 0.01,
        "affine": True,
        "track_running_stats": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - With custom epsilon value
    input = torch.randn(5, 75).numpy()
    input_dict = {
        "num_features": 75,
        "eps": 1e-3,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - With different batch size and features
    input = torch.randn(2, 30).numpy()
    input_dict = {
        "num_features": 30,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - Negative values in input tensor
    input = torch.randn(10, 10).numpy()
    input_dict = {
        "num_features": 10,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Different dimensions (3D)
    input = torch.randn(2, 5, 10).numpy()
    input_dict = {
        "num_features": 5,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - No track running stats
    input = torch.randn(5, 20).numpy()
    input_dict = {
        "num_features": 20,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Different momentum value (None)
    input = torch.randn(3, 15).numpy()
    input_dict = {
        "num_features": 15,
        "eps": 1e-5,
        "momentum": None,
        "affine": True,
        "track_running_stats": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - With large number of features
    input = torch.randn(2, 1000).numpy()
    input_dict = {
        "num_features": 1000,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.BatchNorm1d_1"] = batchnorm1d_inputs()

import torch, copy

def featurealphadropout_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.randn(20, 16, 4, 32, 32).numpy()
    input_dict = {
        "p": 0.2,
        "inplace": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.randn(10, 8, 16, 16).numpy()
    input_dict = {
        "p": 0.5,
        "inplace": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn(5, 32, 8, 8).numpy()
    input_dict = {
        "p": 0.7,
        "inplace": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(1, 1, 32, 32).numpy()
    input_dict = {
        "p": 0.1,
        "inplace": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.randn(32, 16, 8).numpy()
    input_dict = {
        "p": 0.3,
        "inplace": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(8, 4, 128, 128).numpy()
    input_dict = {
        "p": 0.4,
        "inplace": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(16, 32, 32, 32).numpy()
    input_dict = {
        "p": 0.6,
        "inplace": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(2, 1, 32, 32).numpy()
    input_dict = {
        "p": 0.8,
        "inplace": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(16, 16, 4, 32).numpy()
    input_dict = {
        "p": 0.9,
        "inplace": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(1, 32, 8, 8).numpy()
    input_dict = {
        "p": 0.05,
        "inplace": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.FeatureAlphaDropout"] = featurealphadropout_inputs()

import torch, copy

def hubloss_inputs():
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
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    target = torch.tensor([[0.5, 1.5], [2.5, 3.5]]).numpy()
    reduction = 'sum'
    delta = 2.0
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    target = torch.tensor([-0.5, -1.5, -2.5]).numpy()
    reduction = 'none'
    delta = 1.5
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((3, 4)).numpy()
    target = torch.ones((3, 4)).numpy()
    reduction = 'mean'
    delta = 0.5
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.zeros((2, 2, 3)).numpy()
    target = torch.ones((2, 2, 3)).numpy()
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
    input = torch.tensor([0.0]).numpy()
    target = torch.tensor([0.5]).numpy()
    reduction = 'none'
    delta = 1.0
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.0]).numpy()
    target = torch.tensor([-1.0]).numpy()
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
    input = torch.tensor([1.5, 2.5, 3.5]).numpy()
    target = torch.tensor([1.0, 2.0, 3.0]).numpy()
    reduction = 'sum'
    delta = 2.0
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    target = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]]).numpy()
    reduction = 'none'
    delta = 1.0
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1.0, -2.0]).numpy()
    target = torch.tensor([-0.5, -1.5]).numpy()
    reduction = 'mean'
    delta = 0.5
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.HuberLoss"] = hubloss_inputs()

import torch, copy

def logsoftmax_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    dim = 0
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]]).numpy()
    dim = 1
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dim = 2
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[-1.0, -2.0, -3.0]]).numpy()
    dim = 0
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[0.1, 0.2, 0.3],
                         [0.4, 0.5, 0.6]]).numpy()
    dim = 1
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]).numpy()
    dim = 2
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[-1.0, -2.0, -3.0]]).numpy()
    dim = 0
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dim = 2
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]).numpy()
    dim = 2
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[-1.0, -2.0, -3.0]]).numpy()
    dim = 0
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.LogSoftmax"] = logsoftmax_inputs()

import torch, copy

def multi_margin_loss_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target = torch.tensor([3]).numpy()
    weight = torch.tensor([1.0, 1.0, 1.0, 1.0]).numpy()
    input_dict = {
        "p": 1,
        "margin": 1.0,
        "weight": weight,
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
    weight = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict = {
        "p": 2,
        "margin": 1.5,
        "weight": weight,
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8],
                         [0.5, 0.6, 0.7, 0.9]]).numpy()
    target = torch.tensor([3, 1]).numpy()
    weight = torch.tensor([1.0, 1.0, 1.0, 1.0]).numpy()
    input_dict = {
        "p": 1,
        "margin": 2.0,
        "weight": weight,
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target = torch.tensor([3]).numpy()
    weight = torch.tensor([1.0, 1.0, 1.0, 1.0]).numpy()
    input_dict = {
        "p": 1,
        "margin": 1.0,
        "weight": weight,
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8],
                         [0.5, 0.6, 0.7, 0.9]]).numpy()
    target = torch.tensor([3, 1]).numpy()
    weight = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict = {
        "p": 2,
        "margin": 1.5,
        "weight": weight,
        "size_average": True,
        "reduce": False,
        "reduction": "sum",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target = torch.tensor([3]).numpy()
    weight = torch.tensor([1.0, 1.0, 1.0, 1.0]).numpy()
    input_dict = {
        "p": 1,
        "margin": 2.0,
        "weight": weight,
        "size_average": False,
        "reduce": True,
        "reduction": "none",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8],
                         [0.5, 0.6, 0.7, 0.9]]).numpy()
    target = torch.tensor([3, 1]).numpy()
    weight = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict = {
        "p": 2,
        "margin": 1.5,
        "weight": weight,
        "size_average": True,
        "reduce": False,
        "reduction": "mean",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8],
                         [0.5, 0.6, 0.7, 0.9]]).numpy()
    target = torch.tensor([3, 1]).numpy()
    weight = torch.tensor([1.0, 1.0, 1.0, 1.0]).numpy()
    input_dict = {
        "p": 1,
        "margin": 1.0,
        "weight": weight,
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target = torch.tensor([3]).numpy()
    weight = torch.tensor([1.0, 1.0, 1.0, 1.0]).numpy()
    input_dict = {
        "p": 2,
        "margin": 2.0,
        "weight": weight,
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target = torch.tensor([3]).numpy()
    weight = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict = {
        "p": 1,
        "margin": 1.5,
        "weight": weight,
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
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
    input = torch.tensor([[1.0, -2.0], [3.0, -4.0]]).numpy()
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
    
    # Input 5, valid
    input = torch.tensor([1.0]).numpy()
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
    
    # Input 6, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
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
    input = torch.ones((1, 2, 3)).numpy()
    lower = 0.125
    upper = 0.3333333333333333
    inplace = True
    
    input_dict = {
        "input": input,
        "lower": lower,
        "upper": upper,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([0.0]).numpy()
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
    
    # Input 9, valid
    input = torch.tensor([1.0, -2.0, 3.0, -4.0]).numpy()
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
    
    # Input 10, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
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

def replication_pad_3d_inputs():
    list_of_inputs = []
    
    # Input 1: 5D tensor with padding tuple (3, 3, 6, 6, 1, 1)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (3, 3, 6, 6, 1, 1)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 5D tensor with padding tuple (1, 1, 1, 1, 1, 1)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (1, 1, 1, 1, 1, 1)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 5D tensor with padding tuple (0, 0, 0, 0, 0, 0)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (0, 0, 0, 0, 0, 0)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 5D tensor with padding tuple (2, 2, 3, 3, 4, 4)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (2, 2, 3, 3, 4, 4)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 5D tensor with padding tuple (1, 1, 2, 2, 3, 3)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (1, 1, 2, 2, 3, 3)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 5D tensor with padding tuple (5, 5, 10, 10, 15, 15)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (5, 5, 10, 10, 15, 15)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 5D tensor with padding tuple (1, 2, 3, 4, 5, 6)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (1, 2, 3, 4, 5, 6)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 5D tensor with padding tuple (0, 1, 2, 3, 4, 5)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (0, 1, 2, 3, 4, 5)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 5D tensor with padding tuple (3, 3, 0, 0, 0, 0)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (3, 3, 0, 0, 0, 0)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 5D tensor with padding tuple (1, 2, 3, 4, 5, 6)
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (1, 2, 3, 4, 5, 6)
    
    input_dict = {
        "padding": padding,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad3d_2"] = replication_pad_3d_inputs()

import torch, copy

def silu_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()   # tensor
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
    input = torch.zeros((1, 4)).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(5).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.5, 2.7, 3.9]).numpy()
    input_dict = {
        "input": input,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-0.5, -1.7, -2.9]).numpy()
    input_dict = {
        "input": input,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.ones((3, 2)).numpy()
    input_dict = {
        "input": input,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(4, 3).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.SiLU"] = silu_inputs()

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
    input = torch.randn(5, 4, 3).numpy()
    input_dict = {
        "input": input,
        "dim": 2
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
    input = torch.tensor([-1.0, 0.0, 1.0]).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((3, 2, 1)).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(4, 5).numpy()
    input_dict = {
        "input": input,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[1.0, 2.0],
                         [3.0, 4.0],
                         [5.0, 6.0]]).numpy()
    input_dict = {
        "input": input,
        "dim": 1
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
    input = torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]]).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], 
                          [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    input_dict = {
        "input": input,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], 
                          [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    input_dict = {
        "input": input,
        "dim": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([-1.0, -2.0, -3.0, -4.0]).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[1.0, 2.0, 3.0], 
                         [4.0, 5.0, 6.0], 
                         [7.0, 8.0, 9.0]]).numpy()
    input_dict = {
        "input": input,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[0.1, 0.2, 0.3], 
                         [0.4, 0.5, 0.6]]).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1.0, 2.0], 
                         [3.0, 4.0], 
                         [5.0, 6.0]]).numpy()
    input_dict = {
        "input": input,
        "dim": 1
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
        "lambd": lambd,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    lambd = 0.5
    input_dict = {
        "lambd": lambd,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0.0, 0.0, 0.0]).numpy()
    lambd = 0.5
    input_dict = {
        "lambd": lambd,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((2, 3)).numpy()
    lambd = 0.5
    input_dict = {
        "lambd": lambd,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.randn(5).numpy()
    lambd = 0.7
    input_dict = {
        "lambd": lambd,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.5, 2.5, 3.5]).numpy()
    lambd = 0.3
    input_dict = {
        "lambd": lambd,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([-1.5, -2.5, -3.5]).numpy()
    lambd = 0.3
    input_dict = {
        "lambd": lambd,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.ones((1, 4)).numpy()
    lambd = 0.0
    input_dict = {
        "lambd": lambd,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.1, 0.2, 0.3]).numpy()
    lambd = 1.0
    input_dict = {
        "lambd": lambd,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(2, 2).numpy()
    lambd = 0.8
    input_dict = {
        "lambd": lambd,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.Softshrink"] = softshrink_inputs()

import torch, copy
import numpy as np

def binary_cross_entropy_with_logits_inputs():
    list_of_inputs = []
    
    # Input 1 - basic case
    input = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - different reduction type
    input = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    size_average = False
    reduce = True
    reduction = 'sum'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - with negative values
    input = np.array([-0.1, 0.2, -0.3], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - multi-dimensional tensor
    input = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    target = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    weight = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - different size_average
    input = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    size_average = False
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - no reduce
    input = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    size_average = True
    reduce = False
    reduction = 'none'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - different reduction type
    input = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    size_average = False
    reduce = False
    reduction = 'sum'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - with weights
    input = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([1.5, 2.0, 1.0], dtype=np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - large values
    input = np.array([10.5, 20.3, 15.7], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - mixed values
    input = np.array([0.5, 0.2, 0.8], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    weight = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.binary_cross_entropy_with_logits_1"] = binary_cross_entropy_with_logits_inputs()

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
    alpha = 0.5
    inplace = True
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()   # tensor
    alpha = 2.0 # float
    inplace = False # boolean
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.zeros((1, 4)).numpy()
    alpha = 1.5
    inplace = False
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.0]).numpy()
    alpha = 1.0
    inplace = True
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    alpha = 0.75
    inplace = False
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([-1.0, 0.0, 1.0]).numpy()
    alpha = 2.5
    inplace = True
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0]).numpy()
    alpha = 1.0
    inplace = False
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.5, 1.5, 2.5]).numpy()
    alpha = 1.2
    inplace = False
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1.0, -2.0]).numpy()
    alpha = 3.0
    inplace = True
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.celu"] = celu_inputs()

import torch, copy

def dropout_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    p = 0.5
    training = True
    inplace = False
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    p = 0.3
    training = False
    inplace = True
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.zeros((1, 4, 5)).numpy()
    p = 0.8
    training = True
    inplace = False
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    p = 0.1
    training = False
    inplace = False
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.ones((3, 2)).numpy()
    p = 0.0
    training = True
    inplace = True
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.zeros((4, 3, 2)).numpy()
    p = 0.9
    training = True
    inplace = False
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[-1.0, -2.0], [3.0, 4.0]]).numpy()
    p = 0.6
    training = True
    inplace = False
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.ones((1, 1)).numpy()
    p = 0.2
    training = False
    inplace = True
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.zeros((2, 2, 2, 2)).numpy()
    p = 0.4
    training = True
    inplace = False
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([5.0, 6.0, 7.0]).numpy()
    p = 0.7
    training = True
    inplace = False
    
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.dropout"] = dropout_inputs()

import torch, copy

def hardshrink_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, -2.0, 3.0]).numpy()
    lambd = 0.5
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([-1.0, 2.0, -3.0]).numpy()
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
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    lambd = 0.7
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    lambd = 0.7
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((3, 2)).numpy()
    lambd = 0.5
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0, -2.0, 3.0]).numpy()
    lambd = 0.2
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[-1.0, 2.0], [3.0, -4.0]]).numpy()
    lambd = 0.2
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
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
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    inplace = False
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    inplace = False
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0.0, 1.0, 2.0]).numpy()
    inplace = False
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-4.0, -5.0, -6.0]).numpy()
    inplace = False
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([4.0, 5.0, 6.0]).numpy()
    inplace = False
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((2, 3)).numpy()
    inplace = False
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((1, 4, 5)).numpy()
    inplace = False
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[-3.0, 0.0], [3.0, 6.0]]).numpy()
    inplace = False
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[-1.5, 1.5], [0.0, 2.0]]).numpy()
    inplace = False
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]).numpy()
    inplace = False
    
    input_dict = {
        "input": input,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.hardswish_1"] = hardswish_inputs()

import torch, copy

def margin_ranking_loss_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input2 = torch.tensor([0.1, 0.2, 0.3]).numpy()
    target = torch.tensor([1.0, 1.0, 1.0]).numpy()
    margin = 0.5
    reduction = 'mean'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input1 = torch.ones((2, 3)).numpy()
    input2 = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]]).numpy()
    target = torch.tensor([[0.1, 0.2, 0.3],
                          [0.4, 0.5, 0.6]]).numpy()
    margin = 0.0
    reduction = 'none'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input2 = torch.tensor([0.1, 0.2, 0.3]).numpy()
    target = torch.tensor([-1.0, -1.0, -1.0]).numpy()
    margin = 0.0
    reduction = 'sum'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input1 = torch.ones((3, 3)).numpy()
    input2 = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]]).numpy()
    target = torch.tensor([[0.1, 0.2, 0.3],
                          [0.4, 0.5, 0.6],
                          [0.7, 0.8, 0.9]]).numpy()
    margin = 0.0
    reduction = 'sum'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input1 = torch.tensor([1.0]).numpy()
    input2 = torch.tensor([0.1]).numpy()
    target = torch.tensor([1.0]).numpy()
    margin = 0.1
    reduction = 'mean'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input1 = torch.tensor([1.0, 2.0]).numpy()
    input2 = torch.tensor([0.1, 0.2]).numpy()
    target = torch.tensor([1.0, 1.0]).numpy()
    margin = -0.5
    reduction = 'none'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input1 = torch.ones((2, 2)).numpy()
    input2 = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0]]).numpy()
    target = torch.tensor([[0.1, 0.2],
                          [0.3, 0.4]]).numpy()
    margin = 0.5
    reduction = 'sum'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input1 = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input2 = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    target = torch.tensor([1.0, 1.0, 1.0, 1.0]).numpy()
    margin = 0.0
    reduction = 'mean'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input1 = torch.tensor([1.0, 2.0]).numpy()
    input2 = torch.tensor([0.1, 0.2]).numpy()
    target = torch.tensor([1.0, 1.0]).numpy()
    margin = 0.5
    reduction = 'mean'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input1 = torch.tensor([-1.0, -2.0]).numpy()
    input2 = torch.tensor([1.0, 2.0]).numpy()
    target = torch.tensor([1.0, 1.0]).numpy()
    margin = 0.5
    reduction = 'none'
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.margin_ranking_loss"] = margin_ranking_loss_inputs()

import torch, copy

def max_pool1d_inputs():
    list_of_inputs = []
    
    # Input 1 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]).numpy()
    kernel_size = 2
    stride = 1
    padding = 0
    dilation = 1
    ceil_mode = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]]).numpy()
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    ceil_mode = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]]).numpy()
    kernel_size = 4
    stride = 1
    padding = 0
    dilation = 2
    ceil_mode = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]).numpy()
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 1
    ceil_mode = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]).numpy()
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    ceil_mode = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]]).numpy()
    kernel_size = 5
    stride = 1
    padding = 2
    dilation = 1
    ceil_mode = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]]).numpy()
    kernel_size = 2
    stride = 1
    padding = 0
    dilation = 1
    ceil_mode = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]]).numpy()
    kernel_size = 4
    stride = 2
    padding = 1
    dilation = 1
    ceil_mode = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]]).numpy()
    kernel_size = 5
    stride = 1
    padding = 2
    dilation = 1
    ceil_mode = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]]]).numpy()
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    ceil_mode = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.max_pool1d"] = max_pool1d_inputs()

import torch, copy

def pdist_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()
    p = 2.0
    input_dict = {
        "input": input,
        "p": p
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    p = 1.0
    input_dict = {
        "input": input,
        "p": p
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).numpy()
    p = 0.5
    input_dict = {
        "input": input,
        "p": p
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0], [2.0], [3.0]]).numpy()
    p = 2.0
    input_dict = {
        "input": input,
        "p": p
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]).numpy()
    p = 3.0
    input_dict = {
        "input": input,
        "p": p
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()
    p = 0.0
    input_dict = {
        "input": input,
        "p": p
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()
    p = 2.0
    input_dict = {
        "input": input,
        "p": p
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()
    p = 1.5
    input_dict = {
        "input": input,
        "p": p
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()
    p = 2.0
    input_dict = {
        "input": input,
        "p": p
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()
    p = 2.0
    input_dict = {
        "input": input,
        "p": p
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.pdist"] = pdist_inputs()

import torch, copy

def prelu_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, -2.0, 3.0]).numpy()
    weight = torch.tensor(0.5).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[1.0, -2.0], [3.0, -4.0]]).numpy()
    weight = torch.tensor([0.1]).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((2, 3, 4)).numpy()
    weight = torch.tensor([0.5, 0.2, 0.1]).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    weight = torch.tensor(-0.5).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.ones((1, 5)).numpy()
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0, -2.0, 3.0, -4.0]).numpy()
    weight = torch.tensor([0.5]).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((3, 4)).numpy()
    weight = torch.tensor([0.1]).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]]).numpy()
    weight = torch.tensor([0.2]).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.ones((1, 2, 3)).numpy()
    weight = torch.tensor([0.5]).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1.0, -2.0, -3.0, -4.0]).numpy()
    weight = torch.tensor([0.3]).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.prelu"] = prelu_inputs()

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
    input = torch.ones((2, 3)).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([-1.0, -2.0, 3.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([0.0, 1.0, 6.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([7.0, 8.0, 9.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((1, 4)).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[-1.0, 0.0], [1.0, 2.0]]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.ones((3, 2)).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
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
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()   # tensor
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
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()   # tensor
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]]).numpy()   # tensor
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0]).numpy()   # tensor
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([-1.0]).numpy()   # tensor
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.zeros((2, 3, 4)).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.0]).numpy()   # tensor
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-0.5]).numpy()   # tensor
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.selu"] = selu_inputs()

import torch, copy

def constant__inputs():
    list_of_inputs = []
    
    # Input 1, valid
    tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    val = 5.0
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    tensor = torch.ones((2, 3)).numpy()
    val = -1.0
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    tensor = torch.zeros((4, 5, 6)).numpy()
    val = 0.0
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    tensor = torch.ones(3).numpy()
    val = 10.5
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    val = -2.0
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    tensor = torch.ones((1, 1)).numpy()
    val = 0.5
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    val = 1.0
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    tensor = torch.zeros((2, 2, 2)).numpy()
    val = 3.14
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    tensor = torch.ones((1, 2, 3, 4)).numpy()
    val = -5.0
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    tensor = torch.tensor([1.0]).numpy()
    val = 100.0
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.init.constant__1"] = constant__inputs()

import torch, copy

def constant__inputs():
    list_of_inputs = []
    
    # Input 1, valid
    tensor = torch.tensor([1, 2, 3]).numpy()
    val = 5
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    tensor = torch.ones((2, 3)).numpy()
    val = -1
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    val = 0
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    tensor = torch.zeros((1, 2, 3)).numpy()
    val = 99
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    tensor = torch.tensor([1.0]).numpy()
    val = 7
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [0, 1, 2]]]).numpy()
    val = -5
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    tensor = torch.tensor([1, 2, 3, 4, 5]).numpy()
    val = 10
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    tensor = torch.tensor([[1, 2], [3, 4], [5, 6]]).numpy()
    val = 100
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    tensor = torch.tensor([1, 2, 3]).numpy()
    val = 0
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    tensor = torch.ones((3, 4)).numpy()
    val = -10
    
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.init.constant__2"] = constant__inputs()

import torch, copy

def not_equal_inputs():
    list_of_inputs = []
    
    # Input 1 - valid
    input = torch.tensor([1, 2, 3]).numpy()
    other = torch.tensor([1, 2, 4]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - valid
    input = torch.tensor([0.5, 1.5]).numpy()
    other = torch.tensor([0.5, 1.0]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - valid
    input = torch.ones((2, 3)).numpy()
    other = torch.zeros((2, 3)).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - valid
    input = torch.tensor([1, 2, 3, 4, 5]).numpy()
    other = torch.tensor([1, 2, 3, 4, 5]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - valid
    input = torch.tensor([-1, -2, -3]).numpy()
    other = torch.tensor([1, 2, 3]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - valid
    input = torch.tensor([0]).numpy()
    other = torch.tensor([1]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - valid
    input = torch.ones((3, 4)).numpy()
    other = torch.zeros((3, 4)).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - valid
    input = torch.tensor([1, 2, 3]).numpy()
    other = torch.tensor([1, 2, 3]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - valid
    input = torch.tensor([10, 20]).numpy()
    other = torch.tensor([10, 20]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - valid
    input = torch.tensor([100, 200, 300]).numpy()
    other = torch.tensor([100, 200, 400]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.not_equal"] = not_equal_inputs()

import torch, copy

def numel_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1, 2, 3, 4, 5]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.zeros((4, 4)).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((1, 2, 3, 4, 5)).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(1, 2, 3, 4, 5).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.zeros((3, 3, 3)).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((2, 2, 2, 2, 2)).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[[1, 2, 3], [4, 5, 6]]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.zeros((1, 1, 1, 1)).numpy()
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
    input = torch.randn(1, 4, 6, 8).numpy()
    dims = (3, 2, 1, 0)
    input_dict = {
        "input": input,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn(7, 9).numpy()
    dims = (1, 0)
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
    dims = (2, 1, 0)
    input_dict = {
        "input": input,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(3, 4, 5, 6).numpy()
    dims = (3, 2, 1, 0)
    input_dict = {
        "input": input,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(1, 2, 3, 4).numpy()
    dims = (3, 2, 1, 0)
    input_dict = {
        "input": input,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(5, 4, 3, 2, 1).numpy()
    dims = (4, 3, 2, 1, 0)
    input_dict = {
        "input": input,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(1, 1, 1).numpy()
    dims = (2, 1, 0)
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
    input = torch.randn(5).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([0.0, -1.0, 2.0]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.zeros((3, 4)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([-0.5, -1.5, -2.5]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(10).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.ones((1, 2, 3)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([0.1, 0.2, 0.3]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.positive"] = positive_inputs()

import torch, copy

def reshape_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.arange(4.).numpy()
    shape = (2, 2)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[0, 1], [2, 3]]).numpy()
    shape = (-1,)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((3, 4)).numpy()
    shape = (2, 6)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.zeros((1, 5, 6)).numpy()
    shape = (30,)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.arange(12.).numpy()
    shape = (-1, 3)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((2, 2, 2)).numpy()
    shape = (4, -1)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.arange(6.).numpy()
    shape = (2, 3)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.zeros((4, 5)).numpy()
    shape = (-1, 2)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.ones((3, 3, 3)).numpy()
    shape = (1, 27)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.arange(8.).numpy()
    shape = (-1, 2, 2)
    
    input_dict = {
        "input": input,
        "shape": shape
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.reshape"] = reshape_inputs()

import torch, copy

def sin_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    out = torch.zeros((3,)).numpy()    # tensor

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    out = torch.zeros((2, 3)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn(4).numpy()
    out = torch.zeros((4,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    out = torch.zeros((3,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.zeros((1, 2, 3)).numpy()
    out = torch.zeros((1, 2, 3)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.0]).numpy()
    out = torch.zeros((1,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(5).numpy()
    out = torch.zeros((5,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.57, 3.14, 2.0]).numpy()  # tensor
    out = torch.zeros((3,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.ones((3, 4)).numpy()
    out = torch.zeros((3, 4)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(6).numpy()
    out = torch.zeros((6,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.sin"] = sin_inputs()

import torch, copy

def erfc_inputs():
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
    input = torch.tensor([0.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.ones((1, 1)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.5, 2.5]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((3, 2)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-1.5, -2.5]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.0, 1.0, 2.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.ones((4, 5)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.special.erfc"] = erfc_inputs()

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
    input = torch.tensor([0.5, 1.5, 2.5]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones(5).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.1, 0.2, 0.3]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - negative values
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - mixed dimensions
    input = torch.ones((3, 4)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - single element tensor
    input = torch.tensor([10.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - float values
    input = torch.tensor([1.0, 2.5, 3.7]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - negative float values
    input = torch.tensor([-1.0, -2.5, -3.7]).numpy()
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
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.zeros((3, 4)).numpy()
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
    input = torch.tensor([1.5, 2.5, 3.5]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([0.1, 0.2, 0.3]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.ones((4, 5)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.special.i1e"] = i1e_inputs()

import torch, copy

def polygamma_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    n = 0
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    n = 1
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0.5, 1.5, 2.5]).numpy()
    n = 2
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((1, 4)).numpy()
    n = 3
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.1, 0.2]).numpy()
    n = 4
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((3, 2)).numpy()
    n = 5
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.0]).numpy()
    n = 6
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.ones((2, 2, 2)).numpy()
    n = 7
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.0, 1.0, 2.0]).numpy()
    n = 8
    input_dict = {
        "n": n,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.ones((4, 3)).numpy()
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
    input = torch.tensor([0.0, 1.0, 2.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1.5, -2.7, 0.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((3, 4)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.zeros((2, 2, 2)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([3.14]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.0, -1.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((1, 5)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([-5.0, 0.0, 1.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([2.5, -3.1, 0.0]).numpy()
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
    other = torch.tensor([0.1, 0.2, 0.3]).numpy() # tensor
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    other = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0.0]).numpy()
    other = torch.tensor([1.0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-1.0, -2.0]).numpy()
    other = torch.tensor([0.5, 0.7]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.zeros((3, 4)).numpy()
    other = torch.ones((3, 4)).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0]).numpy()
    other = torch.tensor([0.0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([2.0, 3.0, 4.0]).numpy()
    other = torch.tensor([1.0, 2.0, 3.0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-1.0]).numpy()
    other = torch.tensor([-2.0]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.ones((1, 2, 3)).numpy()
    other = torch.zeros((1, 2, 3)).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([0.5, 1.5]).numpy()
    other = torch.tensor([0.2, 0.8]).numpy()
    
    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.special.xlog1py"] = xlog1py_inputs()

import torch, copy

def take_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[4, 3, 5], [6, 7, 8]]).numpy()
    index = torch.tensor([0, 2, 5]).numpy()
    
    input_dict = {
        "input": input,
        "index": index
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1, 2, 3, 4, 5]).numpy()
    index = torch.tensor([0, 4]).numpy()
    
    input_dict = {
        "input": input,
        "index": index
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    index = torch.tensor([0, 1, 2, 3]).numpy()
    
    input_dict = {
        "input": input,
        "index": index
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).numpy()
    index = torch.tensor([0, 2, 4]).numpy()
    
    input_dict = {
        "input": input,
        "index": index
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[1, 2], [3, 4], [5, 6]]).numpy()
    index = torch.tensor([0, 1, 2, 3, 4]).numpy()
    
    input_dict = {
        "input": input,
        "index": index
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    index = torch.tensor([2, 4]).numpy()
    
    input_dict = {
        "input": input,
        "index": index
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[1, 2], [3, 4]]).numpy()
    index = torch.tensor([0, 1]).numpy()
    
    input_dict = {
        "input": input,
        "index": index
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[1, 2], [3, 4]]).numpy()
    index = torch.tensor([3]).numpy()
    
    input_dict = {
        "input": input,
        "index": index
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[1, 2], [3, 4], [5, 6]]).numpy()
    index = torch.tensor([0, 1, 2]).numpy()
    
    input_dict = {
        "input": input,
        "index": index
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1, 2], [3, 4]]).numpy()
    index = torch.tensor([0, 2]).numpy()
    
    input_dict = {
        "input": input,
        "index": index
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.take_1"] = take_inputs()

import torch, copy

def tril_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]]).numpy()
    diagonal = 0
    out = torch.zeros(input.shape).numpy()

    input_dict = {
        "input": input,
        "diagonal": diagonal,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0]]).numpy()
    diagonal = 1
    out = torch.zeros(input.shape).numpy()

    input_dict = {
        "input": input,
        "diagonal": diagonal,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]]).numpy()
    diagonal = -1
    out = torch.zeros(input.shape).numpy()

    input_dict = {
        "input": input,
        "diagonal": diagonal,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                        [6.0, 7.0, 8.0, 9.0],
                        [11.0, 12.0, 13.0, 14.0],
                        [16.0, 17.0, 18.0, 19.0]]).numpy()
    diagonal = 2
    out = torch.zeros(input.shape).numpy()

    input_dict = {
        "input": input,
        "diagonal": diagonal,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0]]).numpy()
    diagonal = 0
    out = torch.zeros(input.shape).numpy()

    input_dict = {
        "input": input,
        "diagonal": diagonal,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0]]).numpy()
    diagonal = -2
    out = torch.zeros(input.shape).numpy()

    input_dict = {
        "input": input,
        "diagonal": diagonal,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0, 16.0]]).numpy()
    diagonal = 0
    out = torch.zeros(input.shape).numpy()

    input_dict = {
        "input": input,
        "diagonal": diagonal,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0]]).numpy()
    diagonal = 1
    out = torch.zeros(input.shape).numpy()

    input_dict = {
        "input": input,
        "diagonal": diagonal,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                        [6.0, 7.0, 8.0, 9.0, 10.0],
                        [11.0, 12.0, 13.0, 14.0, 15.0]]).numpy()
    diagonal = -1
    out = torch.zeros(input.shape).numpy()

    input_dict = {
        "input": input,
        "diagonal": diagonal,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]]).numpy()
    diagonal = 0
    out = torch.zeros(input.shape).numpy()

    input_dict = {
        "input": input,
        "diagonal": diagonal,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.tril_1"] = tril_inputs()

