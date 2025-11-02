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
    input = torch.tensor([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]]).numpy()
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
    dim = 1
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
    input = torch.tensor([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]]).numpy()
    dim = 1
    keepdim = True
    out = torch.zeros(1).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((2, 3, 4)).numpy()
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
    input = torch.randn((5, 3)).numpy()
    dim = 1
    keepdim = False
    out = torch.zeros(5).numpy()
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[-1.0, -2.0], [3.0, 4.0]]).numpy()
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
    
    # Input 9, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()
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
    input = torch.randn((2, 2, 2)).numpy()
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
    
    return list_of_inputs

generated_inputs["torch.amin_1"] = amin_inputs()

import torch, copy

def any_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([0, 1, 2]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([[False, True], [True, False]]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0, 0, 0]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1, 0, 0]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[True, True], [False, False]]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([-1, 0, 1]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([-1, -2, -3]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[0, 1], [2, 3]]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[[False, True], [True, False]], [[True, False], [False, True]]]).numpy()
    out = torch.tensor([], dtype=torch.bool).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.any_1"] = any_inputs()

import torch, copy

def are_deterministic_algorithms_enabled_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1, 2, 3]).numpy()
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
    input = torch.tensor([[1, 2], [3, 4]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([-1, -2, -3]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([2.5, 3.7]).numpy()
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
    input = torch.tensor([[0.0785, 1.5267, -0.8521, 0.4065],
                        [0.1598, 0.0788, -0.0745, -1.2700],
                        [1.2208, 1.0722, -0.7064, 1.2564],
                        [0.0669, -0.2318, -0.8229, -0.9280]]).numpy()
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
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
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
    input = torch.tensor([[-1.0, 2.0, -3.0, 4.0]]).numpy()
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
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dim = 1
    descending = True
    stable = False
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
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
    
    # Input 6, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    dim = -1
    descending = False
    stable = False
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[-1.0, -2.0, -3.0, 4.0]]).numpy()
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
    
    # Input 8, valid
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]).numpy()
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
    
    # Input 9, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()
    dim = -1
    descending = True
    stable = False
    
    input_dict = {
        "input": input,
        "dim": dim,
        "descending": descending,
        "stable": stable
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
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
    
    return list_of_inputs

generated_inputs["torch.argsort"] = argsort_inputs()

import torch, copy

def bitwise_left_shift_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()         # tensor
    other = torch.tensor([0, 1, 2], dtype=torch.int32).numpy()        # tensor
    out = torch.zeros((3,), dtype=torch.int32).numpy()               # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([10, 20, 30], dtype=torch.int32).numpy()      # tensor
    other = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()        # tensor
    out = torch.zeros((3,), dtype=torch.int32).numpy()               # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((2, 3), dtype=torch.int32).numpy()              # tensor
    other = torch.zeros((2, 3), dtype=torch.int32).numpy()            # tensor
    out = torch.zeros((2, 3), dtype=torch.int32).numpy()              # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([100, 200], dtype=torch.int32).numpy()        # tensor
    other = torch.tensor([2, 3], dtype=torch.int32).numpy()          # tensor
    out = torch.zeros((2,), dtype=torch.int32).numpy()               # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0, 1, 2, 3], dtype=torch.int32).numpy()      # tensor
    other = torch.tensor([1, 2, 3, 4], dtype=torch.int32).numpy()     # tensor
    out = torch.zeros((4,), dtype=torch.int32).numpy()               # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([-1, -2, -3], dtype=torch.int32).numpy()     # tensor
    other = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()        # tensor
    out = torch.zeros((3,), dtype=torch.int32).numpy()               # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1, 2], dtype=torch.int32).numpy()           # tensor
    other = torch.tensor([0, 1], dtype=torch.int32).numpy()          # tensor
    out = torch.zeros((2,), dtype=torch.int32).numpy()               # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([10, 20], dtype=torch.int32).numpy()          # tensor
    other = torch.tensor([3, 4], dtype=torch.int32).numpy()          # tensor
    out = torch.zeros((2,), dtype=torch.int32).numpy()               # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.zeros((3, 4), dtype=torch.int32).numpy()             # tensor
    other = torch.ones((3, 4), dtype=torch.int32).numpy()             # tensor
    out = torch.zeros((3, 4), dtype=torch.int32).numpy()              # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([5, 6, 7], dtype=torch.int32).numpy()         # tensor
    other = torch.tensor([2, 3, 4], dtype=torch.int32).numpy()        # tensor
    out = torch.zeros((3,), dtype=torch.int32).numpy()               # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.bitwise_left_shift"] = bitwise_left_shift_inputs()

import torch, copy

def complex_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    real = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()   # tensor
    imag = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32).numpy()   # tensor
    out = torch.tensor([1.0, 2.0, 3.0], dtype=torch.complex64).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    real = torch.tensor([1.0], dtype=torch.float64).numpy()   # tensor
    imag = torch.tensor([2.0], dtype=torch.float64).numpy()   # tensor
    out = torch.tensor([1.0], dtype=torch.complex128).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    real = torch.tensor([1.0, 2.0], dtype=torch.float32).numpy()   # tensor
    imag = torch.tensor([3.0, 4.0], dtype=torch.float32).numpy()   # tensor
    out = torch.tensor([1.0, 2.0], dtype=torch.complex64).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    real = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64).numpy()   # tensor
    imag = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64).numpy()   # tensor
    out = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.complex128).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    real = torch.tensor([-1.0, -2.0], dtype=torch.float32).numpy()   # tensor
    imag = torch.tensor([-3.0, -4.0], dtype=torch.float32).numpy()   # tensor
    out = torch.tensor([-1.0, -2.0], dtype=torch.complex64).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    real = torch.tensor([0.0], dtype=torch.float32).numpy()   # tensor
    imag = torch.tensor([0.0], dtype=torch.float32).numpy()   # tensor
    out = torch.tensor([0.0], dtype=torch.complex64).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    real = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).numpy()   # tensor
    imag = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64).numpy()   # tensor
    out = torch.tensor([1.0, 2.0, 3.0], dtype=torch.complex128).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    real = torch.tensor([1.0, 2.0], dtype=torch.float64).numpy()   # tensor
    imag = torch.tensor([3.0, 4.0], dtype=torch.float64).numpy()   # tensor
    out = torch.tensor([1.0, 2.0], dtype=torch.complex128).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    real = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).numpy()   # tensor
    imag = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32).numpy()   # tensor
    out = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.complex64).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    real = torch.tensor([1.0, 2.0], dtype=torch.float32).numpy()   # tensor
    imag = torch.tensor([3.0, 4.0], dtype=torch.float32).numpy()   # tensor
    out = torch.tensor([1.0, 2.0], dtype=torch.complex64).numpy()  # tensor
    
    input_dict = {
        "real": real,
        "imag": imag,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.complex"] = complex_inputs()

import torch, copy

def conj_physical_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1+2j, 3+4j, 5+6j], dtype=torch.complex64).numpy()  # tensor
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()    # tensor
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3), dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1-2j, 3-4j, 5-6j], dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.zeros((3, 4), dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0+2j, 3.0+4j], dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((1, 2, 3), dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0-2j, 3.0-4j, 5.0-6j], dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[1+2j, 3+4j, 5+6j], [7+8j, 9+10j, 11+12j]], dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0+2j], dtype=torch.complex64).numpy()
    out = torch.empty(input.shape, dtype=torch.complex64).numpy()
    
    input_dict = {
        "input": input,
        "out": out
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
    input = torch.ones((2, 3)).numpy()
    dim = 1   # integer
    dtype = None   # dtype
    out = torch.empty(2, 3).numpy()    # tensor

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
    out = torch.empty(5).numpy()    # tensor

    input_dict = {
        "input": input,
        "dim": dim,
        "dtype": dtype,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(3, 4).numpy()
    dim = 0   # integer
    dtype = None   # dtype
    out = torch.empty(3, 4).numpy()    # tensor

    input_dict = {
        "input": input,
        "dim": dim,
        "dtype": dtype,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1.0, -2.0, 3.0]).numpy()
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
    
    # Input 6, valid
    input = torch.ones((3, 4, 5)).numpy()
    dim = 2   # integer
    dtype = None   # dtype
    out = torch.empty(3, 4, 5).numpy()    # tensor

    input_dict = {
        "input": input,
        "dim": dim,
        "dtype": dtype,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(2, 3, 4).numpy()
    dim = 1   # integer
    dtype = None   # dtype
    out = torch.empty(2, 3, 4).numpy()    # tensor

    input_dict = {
        "input": input,
        "dim": dim,
        "dtype": dtype,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([0.5, 0.2, 0.1]).numpy()
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
    
    # Input 9, valid
    input = torch.randn(10).numpy()
    dim = 0   # integer
    dtype = None   # dtype
    out = torch.empty(10).numpy()    # tensor

    input_dict = {
        "input": input,
        "dim": dim,
        "dtype": dtype,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    dim = 0   # integer
    dtype = None   # dtype
    out = torch.empty(4).numpy()    # tensor

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
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "tensor": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input_tensor = torch.ones((2, 3)).numpy()
    input_dict = {
        "tensor": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {
        "tensor": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input_tensor = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]).numpy()
    input_dict = {
        "tensor": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input_tensor = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "tensor": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input_tensor = torch.tensor([0.0]).numpy()
    input_dict = {
        "tensor": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input_tensor = torch.tensor([1.5, 2.5, 3.5]).numpy()
    input_dict = {
        "tensor": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input_tensor = torch.ones((1, 1)).numpy()
    input_dict = {
        "tensor": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input_tensor = torch.tensor([[[[1.0, 2.0]]]]).numpy()
    input_dict = {
        "tensor": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input_tensor = torch.zeros((3, 4)).numpy()
    input_dict = {
        "tensor": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.dequantize_1"] = dequantize_inputs()

import torch, copy

def empty_like_inputs():
    list_of_inputs = []
    
    # Input 1
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    dtype = torch.float32
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input = torch.ones((2, 3)).numpy()
    dtype = torch.float64
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input = torch.zeros((1, 2, 3)).numpy()
    dtype = torch.int32
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input = torch.tensor([[1, 2], [3, 4]]).numpy()
    dtype = torch.int64
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dtype = torch.float32
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input = torch.tensor([[-1.0, -2.0, -3.0]]).numpy()
    dtype = torch.float64
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input = torch.tensor([0.0]).numpy()
    dtype = torch.float32
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input = torch.tensor([1.0, 2.0]).numpy()
    dtype = torch.int32
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    dtype = torch.float64
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dtype = torch.int64
    requires_grad = False
    
    input_dict = {
        "input": input,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.empty_like"] = empty_like_inputs()

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
    input = torch.ones((3, 4)).numpy()
    other = torch.ones((3, 4)).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.zeros((2, 3, 4)).numpy()
    other = torch.zeros((2, 3, 4)).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1, 2, 3]).numpy()
    other = torch.tensor([1, 2, 3]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([0.0, 0.0, 0.0]).numpy()
    other = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([-1, -2, -3]).numpy()
    other = torch.tensor([-1, -2, -3]).numpy()
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1, 2, 3]).numpy()
    other = torch.tensor([1, 2, 3]).numpy()
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
    input = torch.tensor([-1.5, -2.7, -3.9]).numpy()
    out = torch.zeros((3,)).numpy()

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
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    out = torch.zeros((2, 2)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.zeros((2, 2, 2)).numpy()
    out = torch.zeros((2, 2, 2)).numpy()

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
    input = torch.tensor([-1.0]).numpy()
    out = torch.zeros((1,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([2.5, -3.7, 1.9]).numpy()
    out = torch.zeros((3,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out = torch.zeros((3,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    out = torch.zeros((3,)).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.fix"] = fix_inputs()

import torch, copy

def float_power_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64).numpy()
    exponent = 2.5
    out = torch.zeros((3,), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((3, 4), dtype=torch.float64).numpy()
    exponent = 0.5
    out = torch.zeros((3, 4), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64).numpy()
    exponent = -1.5
    out = torch.zeros((4,), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((2, 3, 4), dtype=torch.float64).numpy()
    exponent = 3.0
    out = torch.zeros((2, 3, 4), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1.5], dtype=torch.float64).numpy()
    exponent = 2.0
    out = torch.zeros((1,), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((5,), dtype=torch.float64).numpy()
    exponent = 1.0
    out = torch.zeros((5,), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([2.0, 3.0], dtype=torch.float64).numpy()
    exponent = -2.0
    out = torch.zeros((2,), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.ones((1, 2, 3), dtype=torch.float64).numpy()
    exponent = 0.75
    out = torch.zeros((1, 2, 3), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).numpy()
    exponent = 1.5
    out = torch.zeros((3,), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.ones((4,), dtype=torch.float64).numpy()
    exponent = -1.0
    out = torch.zeros((4,), dtype=torch.float64).numpy()

    input_dict = {
        "input": input,
        "exponent": exponent,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.float_power"] = float_power_inputs()

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
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out = torch.tensor([1.0, 2.0, 3.0]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-1.5, -2.7]).numpy()
    out = torch.tensor([-2.0, -3.0]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1.9, 2.8]).numpy()
    out = torch.tensor([1.0, 2.0]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([-0.1, -0.9]).numpy()
    out = torch.tensor([-1.0, -1.0]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([0.0]).numpy()
    out = torch.tensor([0.0]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    out = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([2.5, 3.7]).numpy()
    out = torch.tensor([2.0, 3.0]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1.9, -2.8]).numpy()
    out = torch.tensor([-2.0, -3.0]).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.floor"] = floor_inputs()

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
    input = torch.tensor([-1-2j, -3-4j, -5-6j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[0+0j, 1+1j], [2+2j, 3+3j]], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.5+0.5j, 1.5+1.5j, 2.5+2.5j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([0.0+0.0j, 1.0+1.0j], dtype=torch.complex64).numpy()
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
    input = torch.tensor([-1j, -2j, -3j], dtype=torch.complex64).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([0+1j, 0+2j, 0+3j], dtype=torch.complex64).numpy()
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
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([0.1, 0.2, 0.3]).numpy()
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
    input = torch.tensor([[1.0, 2.0],
                        [3.0, 4.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[0.0, 1.0],
                        [2.0, 3.0]]).numpy()
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
    input = torch.tensor([0.5]).numpy()
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
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
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
    input = torch.zeros((1, 2, 3), requires_grad=True).detach().numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.zeros((1, 2, 3), requires_grad=False).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - negative values
    input = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True).detach().numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - negative values
    input = torch.tensor([-1.0, -2.0, -3.0], requires_grad=False).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - mixed dimensions
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True).detach().numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - mixed dimensions
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=False).numpy()
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
    input = torch.tensor([[1.0, 2.0],
                         [3.0, 4.0]]).numpy()
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
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                          [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    input_dict = {
        "obj": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - negative values
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "obj": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - scalar tensor
    input = torch.tensor(5.0).numpy()
    input_dict = {
        "obj": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - empty tensor
    input = torch.empty(0).numpy()
    input_dict = {
        "obj": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - single element tensor
    input = torch.tensor([1.0]).numpy()
    input_dict = {
        "obj": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - mixed dimensions
    input = torch.tensor([[[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]]]).numpy()
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
    input = torch.tensor([1+0j, 2+0j, 3+0j]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.0, 0.0, 0.0]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((2, 3)).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.zeros((2, 3)).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1+1j, 2+1j, 3+1j]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1+0j, 2+0j, 3+0j]).numpy()   # tensor
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
    other = torch.tensor([3, 4, 5]).numpy()     # tensor
    out = torch.tensor([15, 20, 15]).numpy()    # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1, 2, 3]).numpy()   # tensor
    other = torch.tensor([4, 5, 6]).numpy()   # tensor
    out = torch.tensor([4, 10, 6]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0, 1, 2]).numpy()   # tensor
    other = torch.tensor([0, 2, 3]).numpy()   # tensor
    out = torch.tensor([0, 2, 6]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([10, 20]).numpy()   # tensor
    other = torch.tensor([5, 10]).numpy()   # tensor
    out = torch.tensor([10, 20]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([7, 11, 13]).numpy()   # tensor
    other = torch.tensor([2, 3, 5]).numpy()   # tensor
    out = torch.tensor([14, 33, 65]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([100]).numpy()   # tensor
    other = torch.tensor([50]).numpy()   # tensor
    out = torch.tensor([100]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([-5, -10, -15]).numpy()   # tensor
    other = torch.tensor([3, 4, 5]).numpy()   # tensor
    out = torch.tensor([15, 20, 15]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([2, 4, 6]).numpy()   # tensor
    other = torch.tensor([-3, -6, -9]).numpy()   # tensor
    out = torch.tensor([6, 12, 18]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1, 2, 3]).numpy()   # tensor
    other = torch.tensor([0, 1, 2]).numpy()   # tensor
    out = torch.tensor([0, 2, 6]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([3, 5, 7]).numpy()   # tensor
    other = torch.tensor([2, 4, 6]).numpy()   # tensor
    out = torch.tensor([6, 20, 42]).numpy()   # tensor
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.lcm"] = lcm_inputs()

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
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[-1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[-1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((3, 4)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[-1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[-1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(2, 3).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[-1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]).numpy()
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
    out = (torch.tensor([0.0]).numpy(), torch.tensor([0]).numpy())
    
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
    out = (torch.tensor([0.0]).numpy(), torch.tensor([0]).numpy())
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
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
    
    # Input 4, valid
    input = torch.tensor([[0.0, -1.0], [1.0, 2.0], [-2.0, 3.0]]).numpy()
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
    
    # Input 5, valid
    input = torch.tensor([[-1.0, 0.0], [1.0, 2.0], [-2.0, 3.0]]).numpy()
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
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
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
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()
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
    
    # Input 8, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
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
    input = torch.tensor([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0], [-5.0, -6.0, -7.0]]).numpy()
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
    
    # Input 10, valid
    input = torch.tensor([[0.0, 1.0], [2.0, 3.0]]).numpy()
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

def nansum_inputs():
    list_of_inputs = []
    
    # Input 1 - basic tensor with NaN
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
    
    # Input 2 - 2D tensor with NaN
    input = torch.tensor([[1, 2], [3., float("nan")]]).numpy()
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
    
    # Input 3 - 2D tensor with NaN and dim=0
    input = torch.tensor([[1, 2], [3., float("nan")]]).numpy()
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
    
    # Input 4 - 2D tensor with NaN and dim=1
    input = torch.tensor([[1, 2], [3., float("nan")]]).numpy()
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
    
    # Input 5 - 2D tensor with NaN and keepdim=True
    input = torch.tensor([[1, 2], [3., float("nan")]]).numpy()
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
    
    # Input 6 - 2D tensor with NaN and keepdim=True
    input = torch.tensor([[1, 2], [3., float("nan")]]).numpy()
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
    
    # Input 7 - 3D tensor with NaN
    input = torch.tensor([[[1, 2], [3., float("nan")]], [[4, 5], [6., float("nan")]]]).numpy()
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
    
    # Input 8 - 3D tensor with NaN and dim=0
    input = torch.tensor([[[1, 2], [3., float("nan")]], [[4, 5], [6., float("nan")]]]).numpy()
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
    
    # Input 9 - 3D tensor with NaN and dim=1
    input = torch.tensor([[[1, 2], [3., float("nan")]], [[4, 5], [6., float("nan")]]]).numpy()
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
    
    # Input 10 - 3D tensor with NaN and dim=2
    input = torch.tensor([[[1, 2], [3., float("nan")]], [[4, 5], [6., float("nan")]]]).numpy()
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

def avgpool2d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic case with integer kernel_size, stride, padding
    input = torch.randn(1, 1, 4, 4).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Different kernel_size and stride values
    input = torch.randn(2, 3, 6, 6).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - With ceil_mode true
    input = torch.randn(1, 2, 5, 5).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - With count_include_pad false
    input = torch.randn(1, 1, 4, 4).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": None,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - With divisor_override specified
    input = torch.randn(1, 1, 4, 4).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 3,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - Non-square kernel and stride
    input = torch.randn(1, 1, 5, 5).numpy()
    input_dict = {
        "kernel_size": (2, 3),
        "stride": (1, 2),
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - With padding and different kernel size
    input = torch.randn(1, 1, 6, 6).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Large kernel_size and stride values
    input = torch.randn(1, 1, 8, 8).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 3,
        "padding": 2,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Different input dimensions (3D)
    input = torch.randn(1, 5, 4).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - With negative padding (valid case)
    input = torch.randn(1, 1, 6, 6).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.AvgPool2d_1"] = avgpool2d_inputs()

import torch, copy

def avgpool2d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic case with square kernel and stride
    input = torch.randn(20, 16, 50, 32).numpy()
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (0, 0)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Non-square kernel and stride
    input = torch.randn(10, 8, 64, 64).numpy()
    kernel_size = (3, 2)
    stride = (2, 1)
    padding = (1, 1)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - With padding
    input = torch.randn(5, 4, 20, 20).numpy()
    kernel_size = (4, 4)
    stride = (1, 1)
    padding = (2, 2)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - With ceil_mode
    input = torch.randn(15, 32, 30, 30).numpy()
    kernel_size = (5, 5)
    stride = (3, 3)
    padding = (0, 0)
    ceil_mode = True
    count_include_pad = True
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - With divisor_override
    input = torch.randn(10, 8, 64, 64).numpy()
    kernel_size = (2, 2)
    stride = (1, 1)
    padding = (0, 0)
    ceil_mode = False
    count_include_pad = True
    divisor_override = 4
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - Negative values in input tensor
    input = torch.randn(3, 2, 10, 10).numpy()
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Different dimensions (4D)
    input = torch.randn(1, 1, 100, 100).numpy()
    kernel_size = (7, 7)
    stride = (4, 4)
    padding = (3, 3)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Large kernel size
    input = torch.randn(2, 16, 200, 200).numpy()
    kernel_size = (10, 10)
    stride = (5, 5)
    padding = (2, 2)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - With count_include_pad set to False
    input = torch.randn(5, 8, 10, 10).numpy()
    kernel_size = (2, 2)
    stride = (1, 1)
    padding = (0, 0)
    ceil_mode = False
    count_include_pad = False
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - With positive padding (valid)
    input = torch.randn(5, 8, 10, 10).numpy()
    kernel_size = (2, 2)
    stride = (1, 1)
    padding = (1, 1)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.AvgPool2d_2"] = avgpool2d_inputs()

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
    input = torch.randn(1, 1, 32, 32).numpy()
    input_dict = {
        "input": input,
        "p": 0.1,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.randn(2, 16, 32, 32).numpy()
    input_dict = {
        "input": input,
        "p": 0.3,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(15, 4, 8, 8).numpy()
    input_dict = {
        "input": input,
        "p": 0.6,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(32, 1, 16, 16).numpy()
    input_dict = {
        "input": input,
        "p": 0.4,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(25, 8, 32, 32).numpy()
    input_dict = {
        "input": input,
        "p": 0.8,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(12, 32, 8, 8).numpy()
    input_dict = {
        "input": input,
        "p": 0.5,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(1, 4, 16, 16).numpy()
    input_dict = {
        "input": input,
        "p": 0.2,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.FeatureAlphaDropout"] = featurealphadropout_inputs()

import torch, copy

def huberloss_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    target = torch.tensor([0.5, 1.5, 2.5]).numpy()
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
    input = torch.tensor([1.0]).numpy()
    target = torch.tensor([2.0]).numpy()
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
    input = torch.tensor([-1.0, -2.0]).numpy()
    target = torch.tensor([0.5, 1.5]).numpy()
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
    input = torch.tensor([0.0, 1.0, 2.0, 3.0]).numpy()
    target = torch.tensor([0.0, 0.0, 0.0, 0.0]).numpy()
    reduction = 'sum'
    delta = 0.5
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    target = torch.tensor([5.0, 6.0, 7.0, 8.0]).numpy()
    reduction = 'none'
    delta = 3.0
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((3, 4)).numpy()
    target = torch.ones((3, 4)).numpy()
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
    input = torch.tensor([-1.0, 0.0, 1.0]).numpy()
    target = torch.tensor([-2.0, 0.0, 2.0]).numpy()
    reduction = 'sum'
    delta = 2.5
    input_dict = {
        "input": input,
        "target": target,
        "reduction": reduction,
        "delta": delta
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.5, 1.5]).numpy()
    target = torch.tensor([0.2, 0.8]).numpy()
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
    input = torch.tensor([10.0]).numpy()
    target = torch.tensor([5.0]).numpy()
    reduction = 'mean'
    delta = 2.0
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
    input = torch.ones((1, 1)).numpy()
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
    input = torch.zeros((3, 2)).numpy()
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
    input = torch.ones((5, 1)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[-1.0, -2.0], [3.0, 4.0]]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.LogSigmoid"] = logsigmoid_inputs()

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
    input = torch.ones((2, 3, 4)).numpy()
    dim = 2
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-1.0, 0.0, 1.0]).numpy()
    dim = 0
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[1.0, -2.0], [3.0, -4.0]]).numpy()
    dim = 1
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dim = 1
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]).numpy()
    dim = 1
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0]).numpy()
    dim = 0
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]]).numpy()
    dim = 1
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    dim = 1
    
    input_dict = {
        "input": input,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.LogSoftmax"] = logsoftmax_inputs()

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
    
    # Input 2 - Non-square window
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
    
    # Input 3 - With padding
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
    
    # Input 7 - Large kernel size
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Mixed dimensions (e.g., different height and width)
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": (4, 3),
        "stride": (3, 2),
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Large stride
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 4,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - Different kernel size with different stride
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": (5, 4),
        "stride": (4, 3),
        "padding": 0,
        "dilation": 1,
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
    input = torch.randn(1, 1, 4, 4).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Non-square kernel and stride
    input = torch.randn(1, 1, 5, 5).numpy()
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
    input = torch.randn(2, 3, 6, 6).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - With dilation
    input = torch.randn(1, 1, 5, 5).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - With return_indices=True
    input = torch.randn(1, 1, 4, 4).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - With ceil_mode=True
    input = torch.randn(1, 1, 5, 5).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Different batch size
    input = torch.randn(3, 2, 4, 4).numpy()
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
    
    # Input 8 - With negative values in input tensor
    input = torch.randn(1, 1, 4, 4).numpy()
    input[0, 0, 0, 0] = -1.0
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Large kernel size with large stride
    input = torch.randn(1, 1, 10, 10).numpy()
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
    
    # Input 10 - Multiple channels with padding
    input = torch.randn(2, 4, 7, 7).numpy()
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
    
    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_2"] = maxpool2d_inputs()

import torch, copy

def maxpool2d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic case with square kernel and stride
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Non-square window with stride
    input = torch.randn(10, 8, 60, 40).numpy()
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
    
    # Input 3 - With padding (valid)
    input = torch.randn(5, 4, 100, 80).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - With dilation
    input = torch.randn(3, 2, 50, 30).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - With return_indices
    input = torch.randn(1, 1, 20, 15).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - With ceil mode
    input = torch.randn(1, 2, 50, 30).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Different kernel size and stride
    input = torch.randn(2, 4, 100, 80).numpy()
    input_dict = {
        "kernel_size": (5, 4),
        "stride": (3, 2),
        "padding": (2, 2),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Large padding values but valid (max padding should be less than or equal to half of kernel size)
    input = torch.randn(1, 3, 200, 150).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": (1, 1),
        "padding": (2, 2),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Negative values in input tensor
    input = torch.randn(2, 3, 50, 40).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - Different dimensions (2D tensor)
    input = torch.randn(3, 5, 60, 40).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_3"] = maxpool2d_inputs()

import torch, copy

def maxpool2d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic case with square kernel and stride
    input = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Non-square window
    input = torch.randn(10, 8, 40, 30).numpy()
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
    input = torch.randn(5, 4, 20, 15).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": (1, 1),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - With dilation
    input = torch.randn(5, 4, 20, 15).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": (0, 0),
        "dilation": (2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - With return_indices
    input = torch.randn(5, 4, 20, 15).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - With ceil_mode
    input = torch.randn(5, 4, 20, 15).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Negative values in input
    input = torch.randn(5, 4, 20, 15).numpy()
    input[ input < 0] = -1.0
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Different dimensions
    input = torch.randn(2, 3, 50, 40).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Large stride
    input = torch.randn(5, 4, 20, 15).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 5,
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - Large kernel size
    input = torch.randn(5, 4, 20, 15).numpy()
    input_dict = {
        "kernel_size": 7,
        "stride": 1,
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_4"] = maxpool2d_inputs()

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
    input = torch.tensor([[0.5, 0.3, 0.1, 0.9]]).numpy()
    target = torch.tensor([[0, 1, -1, 2]]).numpy()
    input_dict = {
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.9, 0.7, 0.5, 0.3]]).numpy()
    target = torch.tensor([[3, 0, -1, 1], [2, 1, -1, 0]]).numpy()
    input_dict = {
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target = torch.tensor([[3, 0, -1, 1]]).numpy()
    input_dict = {
        "size_average": False,
        "reduce": False,
        "reduction": "sum",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.9, 0.7, 0.5, 0.3]]).numpy()
    target = torch.tensor([[3, 0, -1, 1], [2, 1, -1, 0]]).numpy()
    input_dict = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target = torch.tensor([[3, 0, -1, 1]]).numpy()
    input_dict = {
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[0.5, 0.3, 0.1, 0.9]]).numpy()
    target = torch.tensor([[0, 1, -1, 2]]).numpy()
    input_dict = {
        "size_average": True,
        "reduce": False,
        "reduction": "sum",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.9, 0.7, 0.5, 0.3]]).numpy()
    target = torch.tensor([[3, 0, -1, 1], [2, 1, -1, 0]]).numpy()
    input_dict = {
        "size_average": False,
        "reduce": False,
        "reduction": "none",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
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
    
    # Input 10, valid
    input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.9, 0.7, 0.5, 0.3]]).numpy()
    target = torch.tensor([[3, 0, -1, 1], [2, 1, -1, 0]]).numpy()
    input_dict = {
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MultiLabelMarginLoss"] = multilabelmarginloss_inputs()

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
    input = torch.randn(5).numpy()
    lower = 0.2
    upper = 0.5
    inplace = False
    
    input_dict = {
        "input": input,
        "lower": lower,
        "upper": upper,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
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
    
    # Input 6, valid
    input = torch.tensor([1.0, 2.0]).numpy()
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
    
    # Input 7, valid
    input = torch.tensor([-1.0, -2.0]).numpy()
    lower = 0.2
    upper = 0.5
    inplace = False
    
    input_dict = {
        "input": input,
        "lower": lower,
        "upper": upper,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.ones((3, 4)).numpy()
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
    
    # Input 9, valid
    input = torch.randn(10).numpy()
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
    
    # Input 10, valid
    input = torch.tensor([1.0, -2.0, 3.0]).numpy()
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
    
    return list_of_inputs

generated_inputs["torch.nn.RReLU"] = rrelu_inputs()

import torch, copy

def replication_pad3d_inputs():
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
    input = torch.randn(1, 1, 2, 2, 2).numpy()
    padding = 1
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.randn(2, 3, 5, 10, 15).numpy()
    padding = 2
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(4, 2, 3, 6, 8).numpy()
    padding = 0
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.randn(1, 4, 7, 3, 5).numpy()
    padding = 5
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(3, 5, 4, 8, 10).numpy()
    padding = 1
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(2, 3, 5, 6, 8).numpy()
    padding = 3
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(1, 2, 3, 4, 5).numpy()
    padding = 4
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(5, 1, 2, 3, 4).numpy()
    padding = 2
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(6, 3, 5, 7, 8).numpy()
    padding = 1
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11, valid
    input = torch.randn(7, 4, 6, 9, 10).numpy()
    padding = 3
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12, valid
    input = torch.randn(8, 5, 4, 6, 7).numpy()
    padding = 0
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad3d_1"] = replication_pad3d_inputs()

import torch, copy

def replication_pad3d_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.randn(16, 3, 8, 320, 480).numpy()
    padding = (3, 3, 6, 6, 1, 1)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((1, 2, 4, 8, 16)).numpy()
    padding = (1, 1, 2, 2, 3, 3)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.zeros((2, 3, 5, 10, 15)).numpy()
    padding = (2, 2, 2, 2, 2, 2)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(1, 1, 3, 6, 9).numpy()
    padding = (0, 0, 0, 0, 0, 0)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.ones((4, 2, 7, 14, 28)).numpy()
    padding = (3, 3, 3, 3, 3, 3)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(8, 4, 10, 20, 30).numpy()
    padding = (1, 2, 3, 4, 5, 6)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.zeros((1, 1, 2, 4, 8)).numpy()
    padding = (0, 0, 0, 0, 0, 0)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.ones((3, 5, 6, 12, 24)).numpy()
    padding = (2, 2, 2, 2, 2, 2)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(5, 3, 10, 20, 40).numpy()
    padding = (4, 4, 4, 4, 4, 4)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.zeros((2, 2, 5, 10, 15)).numpy()
    padding = (1, 1, 1, 1, 1, 1)
    
    input_dict = {
        "input": input,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad3d_2"] = replication_pad3d_inputs()

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
    input = torch.randn(5).numpy()
    input_dict = {
        "input": input,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([[1.0, 2.0],
                         [3.0, 4.0]]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(4, 3, 2).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.5, 2.5, 3.5]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([-0.5, -1.5, -2.5]).numpy()
    input_dict = {
        "input": input,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.ones((1, 1)).numpy()
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
    input = torch.randn(4, 5, 6).numpy()
    input_dict = {
        "input": input,
        "dim": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {
        "input": input,
        "dim": 0
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
    input = torch.randn(3, 4).numpy()
    input_dict = {
        "input": input,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]).numpy()
    input_dict = {
        "input": input,
        "dim": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.ones((1, 2, 3)).numpy()
    input_dict = {
        "input": input,
        "dim": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
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
    input = torch.randn(4, 5).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
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
    input = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": input,
        "dim": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.ones((5, 6)).numpy()
    input_dict = {
        "input": input,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(3, 4, 5).numpy()
    input_dict = {
        "input": input,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1.0, 2.0, -3.0]).numpy()
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
    lambd = 0.1
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(5).numpy()
    lambd = 0.8
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.0, 0.0, 0.0]).numpy()
    lambd = 0.3
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([1.5, -1.5, 2.5]).numpy()
    lambd = 0.7
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    lambd = 0.0
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    lambd = 0.0
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.5, 1.5, 2.5]).numpy()
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

generated_inputs["torch.nn.Softshrink"] = softshrink_inputs()

import torch, copy

def celu_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    alpha = 1.0
    inplace = False
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    alpha = 1.5
    inplace = True
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    alpha = 2.0
    inplace = False
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((1, 1, 2)).numpy()
    alpha = 0.5
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
    inplace = False
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([2.5, -1.5]).numpy()
    alpha = 3.0
    inplace = True
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((5,)).numpy()
    alpha = 0.1
    inplace = False
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-3.0, -2.0, -1.0]).numpy()
    alpha = 1.0
    inplace = True
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.ones((3, 4, 5)).numpy()
    alpha = 2.0
    inplace = False
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0, 2.0]).numpy()
    alpha = 0.5
    inplace = False
    
    input_dict = {
        "input": input,
        "alpha": alpha,
        "inplace": inplace
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.celu"] = celu_inputs()

import torch, copy

def gumbel_softmax_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    logits = torch.tensor([0.1, 0.2, 0.3]).numpy()  # tensor
    tau = 0.5  # float
    hard = True  # boolean
    dim = 0  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    logits = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]]).numpy()  # tensor
    tau = 1.0  # float
    hard = False  # boolean
    dim = 1  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()  # tensor
    tau = 0.1  # float
    hard = True  # boolean
    dim = 2  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    logits = torch.tensor([-0.1, -0.2, -0.3]).numpy()  # tensor
    tau = 0.5  # float
    hard = False  # boolean
    dim = 0  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()  # tensor
    tau = 0.7  # float
    hard = True  # boolean
    dim = 0  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()  # tensor
    tau = 0.3  # float
    hard = False  # boolean
    dim = 1  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    logits = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]).numpy()  # tensor
    tau = 0.8  # float
    hard = True  # boolean
    dim = 2  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    logits = torch.tensor([-1.0, -2.0, -3.0]).numpy()  # tensor
    tau = 0.2  # float
    hard = True  # boolean
    dim = 0  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()  # tensor
    tau = 0.9  # float
    hard = False  # boolean
    dim = 1  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    logits = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]).numpy()  # tensor
    tau = 0.4  # float
    hard = True  # boolean
    dim = 2  # integer
    
    input_dict = {
        "logits": logits,
        "tau": tau,
        "hard": hard,
        "dim": dim
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.gumbel_softmax_1"] = gumbel_softmax_inputs()

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
    input = torch.tensor([[1.0, -2.0, 3.0],
                         [4.0, -5.0, 6.0]]).numpy()
    lambd = 0.5
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1.0, -2.0, 3.0]).numpy()
    lambd = 0.2
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    lambd = 0.5
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.0, 0.0, 0.0]).numpy()
    lambd = 0.5
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.0, -2.0, 3.0]).numpy()
    lambd = 0.8
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[-1.0, -2.0],
                         [3.0, -4.0]]).numpy()
    lambd = 0.5
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0, -2.0, 3.0]).numpy()
    lambd = 0.1
    
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[-1.0, -2.0, -3.0],
                         [4.0, 5.0, 6.0]]).numpy()
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
    input = torch.tensor([0.0, 1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[-1.0, 0.0], [1.0, 2.0]]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]).numpy()
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
    input = torch.ones((3, 4)).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([-4.0, -5.0, -6.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([-3.0, 3.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.5, 2.5, 3.5]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.hardswish"] = hardswish_inputs()

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
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).numpy()   # tensor
    p = 0.0  # float
    
    input_dict = {
        "input": input,
        "p": p
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()   # tensor
    p = float('inf')  # float
    
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
    input = torch.tensor([[1.0], [2.0], [3.0]]).numpy()   # tensor
    p = 2.0  # float
    
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
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).numpy()   # tensor
    p = 3.0  # float
    
    input_dict = {
        "input": input,
        "p": p
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]).numpy()   # tensor
    p = 0.5  # float
    
    input_dict = {
        "input": input,
        "p": p
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()   # tensor
    p = 1.0  # float
    
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
    input = torch.tensor([1.0, -2.0, 3.0]).numpy()   # tensor
    weight = torch.tensor([0.1]).numpy()            # tensor
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    weight = torch.tensor([0.5]).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    input = torch.ones((2, 3, 4)).numpy()
    weight = torch.tensor([0.1]).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    input = torch.tensor([[1.0, -2.0], [3.0, -4.0]]).numpy()
    weight = torch.tensor([0.5]).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    weight = torch.tensor([0.1]).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    input = torch.ones((1, 5)).numpy()
    weight = torch.tensor([0.2]).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    input = torch.ones((3, 4, 5)).numpy()
    weight = torch.tensor([0.0]).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    weight = torch.tensor([0.5]).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    input = torch.ones((2, 3, 4, 5)).numpy()
    weight = torch.tensor([0.1]).numpy()
    
    input_dict = {
        "input": input,
        "weight": weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    input = torch.tensor([-1.0, 2.0]).numpy()
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
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    
    input_dict = {
        "input": input,
        "inplace": False
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([7.0, 8.0, 9.0]).numpy()
    
    input_dict = {
        "input": input,
        "inplace": False
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([0.0, 1.0, 2.0]).numpy()
    
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
    input = torch.tensor([0.5, 1.5, 2.5]).numpy()
    
    input_dict = {
        "input": input,
        "inplace": False
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([10.0, 20.0]).numpy()
    
    input_dict = {
        "input": input,
        "inplace": False
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    
    input_dict = {
        "input": input,
        "inplace": False
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-5.0, -4.0, -3.0]).numpy()
    
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
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
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
    input = torch.zeros((2, 2, 2)).numpy()
    input_dict = {
        "input": input,
        "inplace": False
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
    input = torch.tensor([-1.0, -2.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((3, 4)).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([0.0, 1.0, -1.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([-1.0, 0.0, 1.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([2.0, -2.0, 3.0]).numpy()
    input_dict = {
        "input": input,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.selu"] = selu_inputs()

import torch, copy

def numel_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1, 2, 3, 4, 5]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.zeros((4, 4)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((1, 2, 3, 4, 5)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn((2, 2, 2)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1]).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.zeros((3, 4, 5)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.ones((1, 1, 1, 1)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn((1, 2, 3, 4, 5, 6, 7)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.zeros((2, 3, 4, 5, 6)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.ones((1, 2, 3, 4, 5, 6, 7, 8)).numpy()
    input_dict = {
        "input": input
    }
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
    input = torch.randn(7, 2).numpy()
    dims = (1, 0)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.randn(3, 3, 3, 3).numpy()
    dims = (3, 2, 1, 0)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.randn(2, 5, 7, 3, 6).numpy()
    dims = (4, 3, 2, 1, 0)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.randn(1, 1).numpy()
    dims = (1, 0)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn(4, 5, 6).numpy()
    dims = (2, 1, 0)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.randn(1, 2, 3, 4, 5).numpy()
    dims = (4, 3, 2, 1, 0)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.randn(6, 4, 2).numpy()
    dims = (2, 1, 0)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(1, 2, 3, 4, 5, 6).numpy()
    dims = (5, 4, 3, 2, 1, 0)
    
    input_dict = {
        "input": input,
        "dims": dims
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.permute"] = permute_inputs()

import torch, copy

def set_num_interop_threads_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = 1   # integer
    input_dict = {
        "num_threads": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.set_num_interop_threads"] = set_num_interop_threads_inputs()

import torch, copy

def erfc_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out = torch.empty(input.shape).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    out = torch.empty(input.shape).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    out = torch.empty(input.shape).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    input = torch.tensor([0.0]).numpy()
    out = torch.empty(input.shape).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    input = torch.ones((1, 1, 1)).numpy()
    out = torch.empty(input.shape).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    input = torch.tensor([1.5, 2.5, 3.5]).numpy()
    out = torch.empty(input.shape).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    input = torch.tensor([-1.5, -2.5, -3.5]).numpy()
    out = torch.empty(input.shape).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    input = torch.ones((3, 2)).numpy()
    out = torch.empty(input.shape).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    input = torch.tensor([0.1, 0.2]).numpy()
    out = torch.empty(input.shape).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    input = torch.ones((4, 5)).numpy()
    out = torch.empty(input.shape).numpy()

    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.special.erfc"] = erfc_inputs()

import torch, copy

def i0e_inputs():
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
    input = torch.tensor([0.1, 0.2, 0.3]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    input = torch.ones((1, 5)).numpy()
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
    input = torch.zeros((3, 4)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    input = torch.ones((2, 2, 2)).numpy()
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    input = torch.tensor([10.0, 20.0]).numpy()   # tensor
    input_dict = {
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    input = torch.tensor([0.5, 1.5, 2.5]).numpy()   # tensor
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
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.zeros((2, 2, 2)).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.0]).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.0]).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1e-5, 1e-4, 1e-3]).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {"x": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.special.i1e"] = i1e_inputs()

import torch, copy

def polygamma_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    n = 0  # integer
    
    input_dict = {
        "n": n,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    n = 1  # integer
    
    input_dict = {
        "n": n,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([0.5, 1.5, 2.5]).numpy()
    n = 2  # integer
    
    input_dict = {
        "n": n,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    n = 3  # integer
    
    input_dict = {
        "n": n,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1.0]).numpy()
    n = 4  # integer
    
    input_dict = {
        "n": n,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((3, 2)).numpy()
    n = 5  # integer
    
    input_dict = {
        "n": n,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.0, 2.0]).numpy()
    n = 6  # integer
    
    input_dict = {
        "n": n,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    n = 7  # integer
    
    input_dict = {
        "n": n,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.1, 0.2]).numpy()
    n = 8  # integer
    
    input_dict = {
        "n": n,
        "input": input
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    n = 9  # integer
    
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
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([0.5, 1.5, 2.5]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.ones((3, 4)).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.zeros((2, 2, 2)).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.0, 0.5, 1.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.ones((1, 5)).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([0.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([3.14, 6.28, 9.42]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.special.sinc"] = sinc_inputs()

import torch, copy

def xlog1py_inputs():
    list_of_inputs = []
    
    # Input 1
    x = torch.tensor([0.0, 1.0, 2.0]).numpy()
    y = torch.tensor([1.0, 1.0, 1.0]).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    x = torch.tensor([0.5, 1.5, 2.5]).numpy()
    y = torch.tensor([0.5, 1.5, 2.5]).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    x = torch.ones((3, 2)).numpy()
    y = torch.ones((3, 2)).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    x = torch.tensor([0.0]).numpy()
    y = torch.tensor([1.0]).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    x = torch.tensor([-1.0, -2.0]).numpy()
    y = torch.tensor([1.0, 1.0]).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    x = torch.tensor([0.0, 1.0, 2.0, 3.0]).numpy()
    y = torch.tensor([1.0, 1.0, 1.0, 1.0]).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    x = torch.tensor([0.0, 1.0, 2.0]).numpy()
    y = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    x = torch.tensor([1.0, 2.0, 3.0]).numpy()
    y = torch.tensor([0.0, 1.0, 2.0]).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    x = torch.ones((2, 3)).numpy()
    y = torch.ones((2, 3)).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    x = torch.tensor([0.0, 1.0]).numpy()
    y = torch.tensor([0.5, 1.5]).numpy()
    input_dict = {
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.special.xlog1py"] = xlog1py_inputs()

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
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other = torch.tensor([1.0, 1.0, 2.0]).numpy()
    out = torch.tensor([True, True, True]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[0, -1], [-2, -3]]).numpy()
    other = torch.tensor([[0, -2], [-3, -4]]).numpy()
    out = torch.tensor([True, True, True, True]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.ones((3, 2)).numpy()
    other = torch.zeros((3, 2)).numpy()
    out = torch.ones((3, 2)).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1, 2, 3]).numpy()
    other = torch.tensor([3, 2, 1]).numpy()
    out = torch.tensor([False, True, True]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([[10, 20], [30, 40]]).numpy()
    other = torch.tensor([[5, 10], [15, 20]]).numpy()
    out = torch.tensor([True, True, True, True]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[0.5, 1.5], [2.5, 3.5]]).numpy()
    other = torch.tensor([[0.5, 1.0], [2.0, 3.0]]).numpy()
    out = torch.tensor([True, True, True, True]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    other = torch.tensor([[1, 1, 1], [1, 1, 1]]).numpy()
    out = torch.tensor([True, True, True, True, True, True]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([[-1, -2], [-3, -4]]).numpy()
    other = torch.tensor([[1, 1], [1, 1]]).numpy()
    out = torch.tensor([False, False, False, False]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    other = torch.tensor([[[1, 1], [4, 4]], [[1, 1], [8, 8]]]).numpy()
    out = torch.tensor([True, True, False, True]).numpy()
    
    input_dict = {
        "input": input,
        "other": other,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.ge"] = ge_inputs()

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
    input = torch.tensor([-0.5, 0.0, 0.5])
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
    input = torch.tensor([0.1, 0.2])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([0.9, 1.1])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([-0.9, -0.1])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([2.0, 3.0])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.5, 2.5])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-0.2, 0.8])
    out = torch.empty_like(input)
    
    input_dict = {
        "input": input.numpy(),
        "out": out.numpy()
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.log1p"] = log1p_inputs()

import torch, copy

def lu_solve_inputs():
    list_of_inputs = []
    
    # Input 1
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    b = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).numpy()
    LU_pivots = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([1, 2], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    LU_pivots = torch.tensor([2, 1], dtype=torch.int32).numpy()
    
    input_dict = {
        "b": b,
        "LU_data": LU_data,
        "LU_pivots": LU_pivots
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
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

def nansum_inputs():
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
    input = torch.tensor([[[1.0, float('nan')], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
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
    input = torch.tensor([-1.0, 2.0, float('nan'), -4.0]).numpy()
    dtype = None
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.tensor([-1.0, float('nan'), float('nan'), 4.0]).numpy()
    dtype = torch.float32
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([[1.0, float('nan')], [float('nan'), 4.0]]).numpy()
    dtype = torch.float64
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([[[1.0, float('nan')], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    dtype = torch.float32
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([-1.0, -2.0, float('nan'), -4.0]).numpy()
    dtype = None
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-1.0, float('nan'), float('nan'), -4.0]).numpy()
    dtype = torch.float64
    input_dict = {
        "input": input,
        "dtype": dtype
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nansum_1"] = nansum_inputs()

import torch, copy

def smoothl1loss_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    target = torch.tensor([0.1, 0.2, 0.3]).numpy() # tensor
    input_dict = {
        "input": input,
        "target": target,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    target = torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": False,
        "reduce": True,
        "reduction": 'sum',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = torch.tensor([[-1.0, 2.0, -3.0]]).numpy()
    target = torch.tensor([0.1, 0.2, 0.3]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": True,
        "reduce": False,
        "reduction": 'none',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = torch.tensor([0.0, 0.0, 0.0]).numpy()
    target = torch.tensor([0.5, 0.5, 0.5]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": False,
        "reduce": False,
        "reduction": 'none',
        "beta": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    target = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "beta": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = torch.ones((1, 2, 3)).numpy()
    target = torch.tensor([[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]]]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": True,
        "reduce": True,
        "reduction": 'sum',
        "beta": 1.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.tensor([1.0]).numpy()
    target = torch.tensor([0.5]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": False,
        "reduce": True,
        "reduction": 'mean',
        "beta": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0, 2.0]).numpy()
    target = torch.tensor([0.1, 0.2]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": True,
        "reduce": False,
        "reduction": 'sum',
        "beta": 3.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    target = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": False,
        "reduce": True,
        "reduction": 'none',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.tensor([-2.0, -1.0]).numpy()
    target = torch.tensor([-0.5, 0.5]).numpy()
    input_dict = {
        "input": input,
        "target": target,
        "size_average": True,
        "reduce": False,
        "reduction": 'mean',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.SmoothL1Loss"] = smoothl1loss_inputs()

import torch, copy
import numpy as np

def binary_cross_entropy_with_logits_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input = np.array([0.5, -0.3, 0.8]).astype(np.float32)
    target = np.array([1.0, 0.0, 1.0]).astype(np.float32)
    weight = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    size_average = True
    reduce = True
    reduction = 'mean'
    pos_weight = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = np.array([[0.1, -0.5], [0.3, -0.2]]).astype(np.float32)
    target = np.array([[1.0, 0.0], [1.0, 0.0]]).astype(np.float32)
    weight = np.array([[1.0, 1.0], [1.0, 1.0]]).astype(np.float32)
    size_average = False
    reduce = True
    reduction = 'sum'
    pos_weight = np.array([1.0, 1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input = np.array([0.1]).astype(np.float32)
    target = np.array([1.0]).astype(np.float32)
    weight = np.array([1.0]).astype(np.float32)
    size_average = True
    reduce = False
    reduction = 'none'
    pos_weight = np.array([1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input = np.array([-0.1, 0.5, -0.2, 0.7]).astype(np.float32)
    target = np.array([0.0, 1.0, 0.0, 1.0]).astype(np.float32)
    weight = np.array([0.5, 0.5, 0.5, 0.5]).astype(np.float32)
    size_average = False
    reduce = True
    reduction = 'mean'
    pos_weight = np.array([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input = np.array([0.0]).astype(np.float32)
    target = np.array([0.0]).astype(np.float32)
    weight = np.array([1.0]).astype(np.float32)
    size_average = True
    reduce = False
    reduction = 'none'
    pos_weight = np.array([1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input = np.array([0.5, -0.5, 0.8]).astype(np.float32)
    target = np.array([1.0, 0.0, 1.0]).astype(np.float32)
    weight = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    size_average = False
    reduce = False
    reduction = 'none'
    pos_weight = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = np.array([[0.1, -0.5], [0.3, -0.2]]).astype(np.float32)
    target = np.array([[1.0, 0.0], [1.0, 0.0]]).astype(np.float32)
    weight = np.array([[1.0, 1.0], [1.0, 1.0]]).astype(np.float32)
    size_average = True
    reduce = False
    reduction = 'sum'
    pos_weight = np.array([1.0, 1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = np.array([0.1, -0.5, 0.3]).astype(np.float32)
    target = np.array([1.0, 0.0, 1.0]).astype(np.float32)
    weight = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    size_average = False
    reduce = True
    reduction = 'mean'
    pos_weight = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = np.array([0.1]).astype(np.float32)
    target = np.array([1.0]).astype(np.float32)
    weight = np.array([1.0]).astype(np.float32)
    size_average = True
    reduce = True
    reduction = 'sum'
    pos_weight = np.array([1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = np.array([0.5, -0.5, 0.8]).astype(np.float32)
    target = np.array([1.0, 0.0, 1.0]).astype(np.float32)
    weight = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    size_average = True
    reduce = True
    reduction = 'none'
    pos_weight = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction,
        "pos_weight": pos_weight
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.binary_cross_entropy_with_logits"] = binary_cross_entropy_with_logits_inputs()

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
    input = torch.tensor([-0.5461, 0.1347, -2.7266, -0.2746]).numpy()
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
    input = torch.tensor([-1.57, 0.0, 1.57]).numpy()
    out = torch.zeros(3).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input = torch.randn((3, 4)).numpy()
    out = torch.zeros(12).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input = torch.tensor([1.0]).numpy()
    out = torch.zeros(1).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input = torch.ones((1, 2, 3)).numpy()
    out = torch.zeros(6).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input = torch.randn(5).numpy()
    out = torch.zeros(5).numpy()
    
    input_dict = {
        "input": input,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.sin"] = sin_inputs()

