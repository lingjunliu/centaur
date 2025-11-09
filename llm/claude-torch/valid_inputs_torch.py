generated_inputs = {}
import torch
import numpy as np
import copy

def any_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([False, True])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[False, False], [False, False]])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0, 0, 0])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0, 1, 2])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0.0, 0.0, 0.0])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0.0, 1.5, -2.3])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[True, False], [False, True]], [[False, False], [True, True]]])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-1, -2, -3])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([True])
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.zeros((10, 10), dtype=bool)
    out = np.array(False)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0, 1, 2], dtype=np.uint8)
    out = np.array(0, dtype=np.uint8)
    input_dict = {
        "input": input_tensor,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.any_1"] = any_inputs()

import torch
import copy

def are_deterministic_algorithms_enabled_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.are_deterministic_algorithms_enabled"] = are_deterministic_algorithms_enabled_inputs()

import torch
import numpy as np
import copy

def argsort_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([3.0, 1.0, 4.0, 2.0]),
        "dim": -1,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[0.5, 2.0], [1.5, 0.5], [0.1, 3.0]]),
        "dim": 0,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[3.0, 1.0, 4.0], [2.0, 5.0, 1.0]]),
        "dim": 1,
        "descending": True,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[1.0, 3.0], [2.0, 0.0]], [[4.0, 1.0], [3.0, 2.0]]]),
        "dim": 2,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([-1.0, -3.0, 2.0, 0.0, -2.0]),
        "dim": -1,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1.0, 2.0, 1.0, 3.0, 2.0]),
        "dim": 0,
        "descending": False,
        "stable": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[5.0, 2.0, 8.0, 2.0], [1.0, 3.0, 1.0, 4.0]]),
        "dim": 1,
        "descending": True,
        "stable": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, 4.0], [3.0, 2.0], [5.0, 0.0]]),
        "dim": -2,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([42.0]),
        "dim": 0,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[[1.0, 2.0], [3.0, 0.0]]]]),
        "dim": 3,
        "descending": True,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.argsort"] = argsort_inputs()

import torch
import numpy as np
import copy

def asin_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([0.0, 0.5, -0.5, 0.707])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[0.1, 0.2], [0.3, 0.4]])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array(0.5)
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[-0.1, -0.2], [-0.3, -0.4]], [[0.1, 0.2], [0.3, 0.4]]])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-1.0, -0.9, 0.9, 1.0])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0.0, 0.0, 0.0])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0.1, 0.2, 0.3])
    out_tensor = np.empty(3)
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[0.5, -0.5], [0.3, -0.3]])
    out_tensor = np.empty((2, 2))
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.linspace(-0.99, 0.99, 100)
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0.01, 0.001, 0.0001])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[[0.1, 0.2], [0.3, 0.4]]]])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.asin"] = asin_inputs()

import torch
import numpy as np
import copy

def conj_physical_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 1D complex tensor
    input_tensor = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D complex tensor
    input_tensor = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Real tensor (float32)
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Real tensor (float64)
    input_tensor = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Complex tensor with negative values
    input_tensor = np.array([-1+2j, -3-4j, 5-6j], dtype=np.complex64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D complex tensor
    input_tensor = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex128)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single element complex tensor
    input_tensor = np.array([3+4j], dtype=np.complex64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Complex tensor with zeros
    input_tensor = np.array([0+0j, 1+0j, 0+1j], dtype=np.complex128)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Integer tensor
    input_tensor = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D complex tensor
    input_tensor = np.array([[[[1+1j]]]], dtype=np.complex64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Large 2D complex tensor
    input_tensor = np.random.randn(5, 5).astype(np.float32) + 1j * np.random.randn(5, 5).astype(np.float32)
    input_tensor = input_tensor.astype(np.complex64)
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.conj_physical"] = conj_physical_inputs()

import torch
import copy
import numpy as np

def diag_embed_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "offset": 0,
        "dim1": -2,
        "dim2": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "offset": 1,
        "dim1": -2,
        "dim2": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[5.0, 6.0], [7.0, 8.0]]),
        "offset": -1,
        "dim1": -2,
        "dim2": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        "offset": 0,
        "dim1": -2,
        "dim2": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([10.0, 20.0, 30.0, 40.0]),
        "offset": 0,
        "dim1": 0,
        "dim2": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1, 2, 3, 4, 5]),
        "offset": 2,
        "dim1": -2,
        "dim2": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([7.0, 8.0, 9.0]),
        "offset": 3,
        "dim1": -2,
        "dim2": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1.5, 2.5, 3.5]),
        "offset": -3,
        "dim1": -2,
        "dim2": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        "offset": 0,
        "dim1": 1,
        "dim2": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([42.0]),
        "offset": 0,
        "dim1": -2,
        "dim2": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.diag_embed"] = diag_embed_inputs()

import torch
import numpy as np
import copy

def equal_inputs():
    list_of_inputs = []
    
    input_tensor = torch.tensor([1, 2, 3]).numpy()
    other_tensor = torch.tensor([1, 2, 3]).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([1, 2, 3]).numpy()
    other_tensor = torch.tensor([1, 2, 4]).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([[1, 2], [3, 4]]).numpy()
    other_tensor = torch.tensor([[1, 2], [3, 4]]).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([[1, 2, 3]]).numpy()
    other_tensor = torch.tensor([[1], [2], [3]]).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([1.5, 2.5, 3.5]).numpy()
    other_tensor = torch.tensor([1.5, 2.5, 3.5]).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([-1, -2, -3]).numpy()
    other_tensor = torch.tensor([-1, -2, -3]).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.zeros((2, 3, 4)).numpy()
    other_tensor = torch.zeros((2, 3, 4)).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor(5).numpy()
    other_tensor = torch.tensor(5).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([]).numpy()
    other_tensor = torch.tensor([]).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.ones((10, 10)).numpy()
    other_tensor = torch.ones((10, 10)).numpy()
    input_dict = {
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.equal"] = equal_inputs()

import torch
import numpy as np
import copy

def fix_inputs():
    list_of_inputs = []
    
    # Input 1: 1D positive floats
    input_dict = {
        "input": np.array([1.5, 2.7, 3.2, 4.9]),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 1D negative floats
    input_dict = {
        "input": np.array([-1.5, -2.7, -3.2, -4.9]),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D mixed positive and negative floats
    input_dict = {
        "input": np.array([-1.5, 2.7, -3.2, 4.9]),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D array
    input_dict = {
        "input": np.array([[1.6, 2.3, 3.8], [4.1, 5.9, 6.2]]),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D array with negative values
    input_dict = {
        "input": np.array([[-1.6, -2.3, -3.8], [-4.1, -5.9, -6.2]]),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D array
    input_dict = {
        "input": np.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]]),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: scalar
    input_dict = {
        "input": np.array(3.14159),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: with out parameter
    input_dict = {
        "input": np.array([1.5, 2.7, 3.2, 4.9]),
        "out": np.empty(4),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D with out parameter
    input_dict = {
        "input": np.array([[1.6, 2.3], [3.8, 4.1]]),
        "out": np.empty((2, 2)),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: zero values
    input_dict = {
        "input": np.array([0.0, 0.5, -0.5, 0.0]),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: large 1D array
    input_dict = {
        "input": np.linspace(-10.5, 10.5, 100),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: single element array
    input_dict = {
        "input": np.array([5.678]),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.fix"] = fix_inputs()

import torch
import copy
import numpy as np

def flipud_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor
    input = torch.arange(4).view(2, 2).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 1D tensor
    input = torch.arange(5).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor
    input = torch.arange(24).view(2, 3, 4).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor
    input = torch.arange(16).view(2, 2, 2, 2).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor with negative values
    input = torch.tensor([[-1.0, -2.0], [3.0, 4.0]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensor with float values
    input = torch.tensor([[1.5, 2.5], [3.5, 4.5]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Large 2D tensor
    input = torch.arange(100).view(10, 10).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 1D tensor with single element
    input = torch.tensor([42]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D tensor with single row
    input = torch.tensor([[1, 2, 3]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D tensor with single column
    input = torch.tensor([[1], [2], [3]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: 3D tensor with negative values
    input = torch.tensor([[[-1, -2], [-3, -4]], [[5, 6], [7, 8]]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: 2D tensor with mixed positive/negative/zero
    input = torch.tensor([[0, -1, 2], [-3, 4, -5]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.flipud"] = flipud_inputs()

import torch
import numpy as np
import copy

def floor_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([-0.8166, 1.5308, -0.2530, -0.2091])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.5, -2.3, 3.7], [-4.1, 5.9, -6.2]])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array(3.14159)
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-100.9, -200.1, -300.5])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0.001, -0.001, 0.999, -0.999])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1.5, 2.5, 3.5])
    out_tensor = np.empty(3)
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[[1.1, 2.2], [3.3, 4.4]]]])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0.0, 0.0, 0.0])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([999.99, 1000.01, 5000.5])
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.floor"] = floor_inputs()

import torch
import copy
import numpy as np

def floor_divide_inputs():
    list_of_inputs = []
    
    input = torch.tensor([10.0, 20.0, 30.0]).numpy()
    other = torch.tensor([3.0, 4.0, 5.0]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[15.0, 25.0], [35.0, 45.0]]).numpy()
    other = torch.tensor([[3.0, 5.0], [7.0, 9.0]]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([-10.0, -20.0, 30.0]).numpy()
    other = torch.tensor([3.0, 4.0, -5.0]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([10, 20, 30]).numpy()
    other = torch.tensor([3, 4, 5]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[[12.0, 15.0], [18.0, 21.0]], [[24.0, 27.0], [30.0, 33.0]]]).numpy()
    other = torch.tensor([[[3.0, 5.0], [6.0, 7.0]], [[8.0, 9.0], [10.0, 11.0]]]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([10.0, 20.0, 30.0, 40.0]).numpy()
    other = torch.tensor([2.0]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.ones((3, 4)).numpy() * 10
    other = torch.tensor([2.0, 3.0, 4.0, 5.0]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([1000.0, 2000.0, 3000.0]).numpy()
    other = torch.tensor([7.0, 13.0, 17.0]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([0.5, 1.5, 2.5, 3.5]).numpy()
    other = torch.tensor([0.3, 0.4, 0.5, 0.6]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([100, 200, 300]).numpy()
    other = torch.tensor([11, 13, 17]).numpy()
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.floor_divide_1"] = floor_divide_inputs()

import torch
import numpy as np
import copy

def floor_divide_inputs():
    list_of_inputs = []
    
    input = torch.tensor([10.0, 20.0, 30.0]).numpy()
    other = 3
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[12.0, 15.0], [18.0, 21.0]]).numpy()
    other = 5
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([-10.0, -20.0, -30.0]).numpy()
    other = 3
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[[8.0, 16.0], [24.0, 32.0]], [[40.0, 48.0], [56.0, 64.0]]]).numpy()
    other = 4
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([10.0, -10.0, 20.0, -20.0]).numpy()
    other = 7
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([100.0]).numpy()
    other = 9
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(5, 5).numpy() * 100
    other = 11
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([0.0, 5.0, 10.0, 0.0]).numpy()
    other = 2
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([10.0, 20.0, 30.0]).numpy()
    other = -3
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.ones(2, 2, 2, 2).numpy() * 15
    other = 4
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([100, 200, 300], dtype=torch.float32).numpy()
    other = 13
    input_dict = {"input": input, "other": other}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.floor_divide_2"] = floor_divide_inputs()

import torch
import numpy as np
import copy

def floor_divide_inputs():
    list_of_inputs = []
    
    input = torch.tensor([10.0, 20.0, 30.0]).numpy()
    other = 3.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[12.5, 25.0], [37.5, 50.0]]).numpy()
    other = 5.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([-10.0, -20.0, -30.0]).numpy()
    other = 3.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([15.0, -15.0, 22.0, -22.0]).numpy()
    other = 4.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[[8.0, 16.0], [24.0, 32.0]], [[40.0, 48.0], [56.0, 64.0]]]).numpy()
    other = 8.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([100.0, 200.0, 300.0]).numpy()
    other = -10.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor(42.0).numpy()
    other = 7.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([10.0, 20.0, 30.0]).numpy()
    other = 2.5
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([1000.0, 2000.0, 3000.0, 4000.0]).numpy()
    other = 100.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([0.5, 1.5, 2.5, 3.5]).numpy()
    other = 0.5
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([-50.0, -100.0, -150.0]).numpy()
    other = -5.0
    input_dict = {
        "input": input,
        "other": other
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.floor_divide_3"] = floor_divide_inputs()

import torch
import numpy as np
import copy

def ge_inputs():
    list_of_inputs = []
    
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other = 2.0
    out = torch.empty(3, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    other = 2.5
    out = torch.empty((2, 2), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([-1.0, 0.0, 1.0]).numpy()
    other = 0.0
    out = torch.empty(3, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([5.0, -3.0, 0.0, -10.0]).numpy()
    other = -5.0
    out = torch.empty(4, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    other = 4.5
    out = torch.empty((2, 2, 2), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([100.0, 200.0, 300.0]).numpy()
    other = 150.0
    out = torch.empty(3, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([0.001, 0.002, 0.003]).numpy()
    other = 0.0015
    out = torch.empty(3, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([5.0]).numpy()
    other = 5.0
    out = torch.empty(1, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.ones((2, 2, 2, 2)).numpy()
    other = 0.5
    out = torch.empty((2, 2, 2, 2), dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    input = torch.tensor([-5.0, -10.0, -15.0, -20.0]).numpy()
    other = -12.0
    out = torch.empty(4, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "other": other, "out": out}))
    
    return list_of_inputs

generated_inputs["torch.ge_2"] = ge_inputs()

import torch
import numpy as np
import copy

def gradient_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([4., 1., 1., 16.]),
        "spacing": [1.0],
        "dim": [0],
        "edge_order": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1., 2., 4., 8.], [10., 20., 40., 80.]]),
        "spacing": [3., 2.],
        "dim": [0, 1],
        "edge_order": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1., 2., 4., 8.], [10., 20., 40., 80.], [5., 10., 20., 40.]]),
        "spacing": [2.0, 3.0],
        "dim": [0, 1],
        "edge_order": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1., 4., 9., 16., 25.]),
        "spacing": [1.],
        "dim": [0],
        "edge_order": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(3, 4, 5),
        "spacing": [1., 2., 0.5],
        "dim": [0, 1, 2],
        "edge_order": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1., 2., 4., 8.], [10., 20., 40., 80.]]),
        "spacing": [1.],
        "dim": [1],
        "edge_order": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.linspace(0, 10, 20),
        "spacing": [0.5],
        "dim": [0],
        "edge_order": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[5., 10., 15., 20.], [20., 25., 30., 35.], [40., 45., 50., 55.]]),
        "spacing": [-1., 2.],
        "dim": [0, 1],
        "edge_order": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(3, 4, 5, 6),
        "spacing": [1., 1., 1., 1.],
        "dim": [0, 1, 2, 3],
        "edge_order": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1., 3., 7., 12.], [2., 5., 11., 18.], [4., 8., 15., 24.]]),
        "spacing": [1.5, 2.5],
        "dim": [0, 1],
        "edge_order": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.gradient_5"] = gradient_inputs()

import torch
import numpy as np
import copy

def imag_inputs():
    list_of_inputs = []
    
    input_val = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([[[1+0j, 2+1j]], [[3+2j, 4+3j]]], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([-1-2j, -3-4j, -5+6j], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([1+0j, 2+0j, 3+0j], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([0+1j, 0+2j, 0+3j], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([5+7j], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([1.5+2.5j, 3.7+4.2j], dtype=np.complex128)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([[1+2j, 3+4j, 5+6j], [7+8j, 9+10j, 11+12j]], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([[[[1+1j]]]], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = np.array([1-2j, -3+4j, 5-6j, -7-8j], dtype=np.complex64)
    input_dict = {"input": input_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.imag"] = imag_inputs()

import torch
import numpy as np
import copy

def is_floating_point_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1, 2, 3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1, 2, 3], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([True, False, True], dtype=np.bool_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1, 2, 3], dtype=np.uint8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array(5.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array(5, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.ones((2, 3, 4, 5), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.is_floating_point"] = is_floating_point_inputs()

import torch
import copy

def is_grad_enabled_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.is_grad_enabled"] = is_grad_enabled_inputs()

import torch
import numpy as np
import copy

def is_nonzero_inputs():
    list_of_inputs = []
    
    # Input 1: scalar tensor with non-zero value
    input = torch.tensor(5).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: scalar tensor with zero value
    input = torch.tensor(0).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: scalar tensor with negative value
    input = torch.tensor(-3).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: scalar float tensor with non-zero value
    input = torch.tensor(1.5).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: scalar float tensor with zero value
    input = torch.tensor(0.0).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: scalar float tensor with small positive value
    input = torch.tensor(0.001).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: scalar float tensor with negative value
    input = torch.tensor(-2.5).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: scalar tensor with large positive value
    input = torch.tensor(1000000).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: scalar bool tensor True
    input = torch.tensor(True).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: scalar bool tensor False
    input = torch.tensor(False).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: scalar tensor with value 1
    input = torch.tensor(1).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: scalar tensor with value -1
    input = torch.tensor(-1).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.is_nonzero"] = is_nonzero_inputs()

import torch
import numpy as np
import copy

def is_storage_inputs():
    list_of_inputs = []
    
    # Input 1: 1D tensor
    obj = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D tensor
    obj = torch.ones((2, 3)).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor
    obj = torch.randn(2, 3, 4).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: scalar tensor
    obj = torch.tensor(5.0).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: negative values
    obj = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: integer tensor
    obj = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: empty tensor
    obj = torch.tensor([]).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor
    obj = torch.zeros((2, 3, 4, 5)).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: boolean tensor
    obj = torch.tensor([True, False, True]).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: large 2D tensor
    obj = torch.randn(100, 100).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: mixed positive and negative values
    obj = torch.tensor([[1.0, -2.0], [3.0, -4.0]]).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: zeros tensor
    obj = torch.zeros(10).numpy()
    input_dict = {"obj": obj}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.is_storage"] = is_storage_inputs()

import torch
import numpy as np
import copy

def isreal_inputs():
    list_of_inputs = []
    
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([1+1j, 2+2j, 3+3j]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([1+0j, 2+0j, 3+0j]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([1, 1+1j, 2+0j]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[1+1j, 2+2j], [3+0j, 4+0j]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([1, 2, 3, 4, 5]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([-1+1j, -2-2j, -3+0j]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor(5.0).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor(5.0+0j).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([0+0j, 0+0j]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.isreal"] = isreal_inputs()

import torch
import numpy as np
import copy

def svdvals_inputs():
    list_of_inputs = []
    
    A = torch.randn(5, 3).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(4, 4).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(3, 7).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(2, 5, 3).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(6, 4, dtype=torch.float64).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(5, 5, dtype=torch.cfloat).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(4, 6, dtype=torch.cdouble).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(1, 5).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(5, 1).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(3, 2, 8, 6).numpy()
    input_dict = {
        "A": A,
        "driver": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.linalg.svdvals"] = svdvals_inputs()

import torch
import numpy as np
import copy

def matrix_power_inputs():
    list_of_inputs = []
    
    input = np.eye(2)
    n = 2
    input_dict = {
        "input": input,
        "n": n
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = np.array([[1.0, 2.0, 0.0],
                      [0.0, 1.0, 1.0],
                      [0.0, 0.0, 1.0]])
    n = 3
    input_dict = {
        "input": input,
        "n": n
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = np.array([[2.0, 3.0],
                      [1.0, 4.0]])
    n = 0
    input_dict = {
        "input": input,
        "n": n
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = np.array([[5.0, -2.0],
                      [1.0, 3.0]])
    n = 1
    input_dict = {
        "input": input,
        "n": n
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = np.eye(4)
    n = 2
    input_dict = {
        "input": input,
        "n": n
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = np.array([[-1.0, 2.0],
                      [3.0, -4.0]])
    n = 3
    input_dict = {
        "input": input,
        "n": n
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = np.array([[1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0],
                      [0.0, 0.0, 3.0]])
    n = 4
    input_dict = {
        "input": input,
        "n": n
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = np.array([[4.0, 2.0],
                      [1.0, 3.0]])
    n = -1
    input_dict = {
        "input": input,
        "n": n
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = np.array([[2.0, 0.0],
                      [0.0, 3.0]])
    n = -2
    input_dict = {
        "input": input,
        "n": n
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = np.eye(5) * 2.0
    n = 2
    input_dict = {
        "input": input,
        "n": n
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.matrix_power"] = matrix_power_inputs()

import torch
import numpy as np
import copy

def median_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 1D tensor
    input_tensor = torch.tensor([1.5219, -1.5212, 0.2202]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D tensor
    input_tensor = torch.randn(4, 5).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Single element tensor
    input_tensor = torch.tensor([5.0]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Tensor with negative values
    input_tensor = torch.tensor([-5.0, -2.0, -10.0, -1.0]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Tensor with mixed positive and negative values
    input_tensor = torch.tensor([[0.2505, -0.3982, -0.9948, 0.3518, -1.3131],
                                  [0.3180, -0.6993, 1.0436, 0.0438, 0.2270]]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Tensor with even number of elements
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Tensor with odd number of elements
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D tensor
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Tensor with zeros
    input_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Large 1D tensor
    input_tensor = torch.randn(100).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Small 2D tensor
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.median_1"] = median_inputs()

import torch
import numpy as np
import copy

def minimum_inputs():
    list_of_inputs = []
    
    input_val = torch.tensor([1, 2, -1]).numpy()
    other = torch.tensor([3, 0, 4]).numpy()
    out = torch.tensor([]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.tensor([[1.5, 2.5], [3.5, 4.5]]).numpy()
    other = torch.tensor([[2.0, 1.0], [3.0, 5.0]]).numpy()
    out = torch.tensor([[]]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.tensor([-5.0, -3.0, -1.0]).numpy()
    other = torch.tensor([-2.0, -4.0, -6.0]).numpy()
    out = torch.tensor([]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).numpy()
    other = torch.tensor([[[2, 1], [4, 3]], [[6, 5], [8, 7]]]).numpy()
    out = torch.tensor([[[]]]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other = torch.tensor([2.5]).numpy()
    out = torch.tensor([]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.randn(10, 10).numpy()
    other = torch.randn(10, 10).numpy()
    out = torch.tensor([[]]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.zeros(5).numpy()
    other = torch.ones(5).numpy()
    out = torch.tensor([]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.tensor([1.1, 2.2, 3.3, 4.4]).numpy()
    other = torch.tensor([1.05, 2.25, 3.2, 4.5]).numpy()
    out = torch.tensor([]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    other = torch.tensor([[0, 1, 2]]).numpy()
    out = torch.tensor([[]]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    input_val = torch.tensor([10, 20, 30]).numpy()
    other = torch.tensor([15, 18, 25]).numpy()
    out = torch.tensor([]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input_val, "other": other, "out": out}))
    
    return list_of_inputs


generated_inputs["torch.minimum"] = minimum_inputs()

import torch
import numpy as np
import copy

def msort_inputs():
    list_of_inputs = []
    
    input = torch.tensor([[-0.1321, 0.4370, -1.2631, -1.1289],
                          [-2.0527, -1.1250, 0.2275, 0.3077],
                          [-0.0881, -0.1259, -0.5495, 1.0284]]).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([5.0, 2.0, 8.0, 1.0, 9.0]).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(4, 3, 2).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[-5.0, -2.0], [-9.0, -1.0], [-3.0, -7.0]]).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[3.0, 7.0, 1.0], [9.0, 2.0, 5.0]]).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[0.0, -1.0, 2.0], [1.0, 0.0, -2.0], [-1.0, 2.0, 0.0]]).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([42.0]).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[10, 5, 8], [3, 15, 1], [7, 2, 12]]).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 3, 4, 5).numpy()
    out = None
    input_dict = {
        "input": input,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.msort"] = msort_inputs()

import torch
import numpy as np
import copy

def nansum_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 1D tensor with NaN
    input_tensor = np.array([1.0, 2.0, float('nan'), 4.0])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D tensor with NaN
    input_tensor = np.array([[1.0, 2.0], [3.0, float('nan')]])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: All NaN values
    input_tensor = np.array([float('nan'), float('nan'), float('nan')])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: No NaN values
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D tensor with NaN
    input_tensor = np.array([[[1.0, float('nan')], [3.0, 4.0]], [[5.0, 6.0], [float('nan'), 8.0]]])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Single element with NaN
    input_tensor = np.array([float('nan')])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single element without NaN
    input_tensor = np.array([42.0])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Negative values with NaN
    input_tensor = np.array([-1.0, -2.0, float('nan'), -4.0])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Large 2D tensor with multiple NaNs
    input_tensor = np.array([[1.0, 2.0, 3.0], [float('nan'), 5.0, float('nan')], [7.0, 8.0, 9.0]])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Mix of positive, negative, zero, and NaN
    input_tensor = np.array([10.0, -5.0, 0.0, float('nan'), 3.14, -2.71])
    input_dict = {
        "input": input_tensor,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nansum_1"] = nansum_inputs()

import numpy as np
import torch
import copy

def nansum_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([1.0, 2.0, np.nan, 4.0]),
        "dim": (),
        "keepdim": False,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, 2.0], [3.0, np.nan]]),
        "dim": (0,),
        "keepdim": False,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]]),
        "dim": (1,),
        "keepdim": True,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[1.0, 2.0], [3.0, np.nan]], [[5.0, 6.0], [np.nan, 8.0]]]),
        "dim": (0, 1),
        "keepdim": False,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([np.nan, np.nan, np.nan]),
        "dim": (),
        "keepdim": False,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "dim": (0,),
        "keepdim": True,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[[1.0, -2.0], [np.nan, 4.0]], [[-5.0, 6.0], [7.0, np.nan]]]]),
        "dim": (2,),
        "keepdim": False,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([np.nan]),
        "dim": (),
        "keepdim": False,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.5, 2.5, np.nan, -3.5], [np.nan, 5.5, 6.5, 7.5], [8.5, np.nan, 9.5, 10.5]]),
        "dim": (1,),
        "keepdim": True,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[-1.0, -2.0, np.nan], [-4.0, np.nan, -6.0]]),
        "dim": (0,),
        "keepdim": False,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nansum_3"] = nansum_inputs()

import torch
import numpy as np
import copy

def logsigmoid_inputs():
    list_of_inputs = []
    
    # Input 1: 1D tensor with positive values
    input_dict = {
        "input": torch.randn(5).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 1D tensor with negative values
    input_dict = {
        "input": torch.tensor([-1.0, -2.0, -3.0, -4.0]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 2D tensor
    input_dict = {
        "input": torch.randn(3, 4).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor
    input_dict = {
        "input": torch.randn(2, 3, 4).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 4D tensor
    input_dict = {
        "input": torch.randn(2, 3, 4, 5).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Single scalar value
    input_dict = {
        "input": torch.tensor([5.0]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Tensor with zeros
    input_dict = {
        "input": torch.zeros(3, 3).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Tensor with mixed positive and negative values
    input_dict = {
        "input": torch.tensor([[-1.5, 2.3], [0.5, -3.2]]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Large values
    input_dict = {
        "input": torch.tensor([10.0, 20.0, 30.0]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Very small values
    input_dict = {
        "input": torch.tensor([0.01, 0.001, 0.0001]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: 5D tensor
    input_dict = {
        "input": torch.randn(1, 2, 2, 2, 2).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Single element tensor
    input_dict = {
        "input": torch.tensor([-0.5]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.LogSigmoid"] = logsigmoid_inputs()

import torch
import numpy as np
import copy

def logsoftmax_inputs():
    list_of_inputs = []
    
    input_tensor = torch.randn(2, 3).numpy()
    input_dict = {
        "dim": 1,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(4, 5).numpy()
    input_dict = {
        "dim": 0,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(10).numpy()
    input_dict = {
        "dim": 0,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "dim": 2,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(5, 6, 7).numpy()
    input_dict = {
        "dim": 1,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {
        "dim": 3,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(3, 4).numpy()
    input_dict = {
        "dim": -1,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 3, 5).numpy()
    input_dict = {
        "dim": -2,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(100, 50).numpy()
    input_dict = {
        "dim": 1,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 2, 2, 2, 3).numpy()
    input_dict = {
        "dim": 4,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 10).numpy()
    input_dict = {
        "dim": 1,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(10, 1).numpy()
    input_dict = {
        "dim": 0,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.LogSoftmax"] = logsoftmax_inputs()

import numpy as np
import copy

def reflectionpad2d_inputs():
    list_of_inputs = []
    
    padding = 1
    input_tensor = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 2
    input_tensor = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 1
    input_tensor = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 1
    input_tensor = np.random.randn(1, 3, 4, 4).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 2
    input_tensor = np.random.randn(2, 2, 5, 5).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 1
    input_tensor = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 1
    input_tensor = np.random.randn(1, 1, 4, 6).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 3
    input_tensor = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 1
    input_tensor = np.random.randn(1, 1, 10, 10).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 2
    input_tensor = np.random.randn(4, 3, 6, 6).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad2d_1"] = reflectionpad2d_inputs()

import torch
import numpy as np
import copy

def reflectionpad2d_inputs():
    list_of_inputs = []
    
    padding = (2, 2, 2, 2)
    input_tensor = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (1, 1, 2, 0)
    input_tensor = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (1, 1, 1, 1)
    input_tensor = torch.randn(2, 3, 4, 4).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (1, 2, 1, 1)
    input_tensor = torch.randn(2, 5, 5).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (0, 2, 1, 0)
    input_tensor = torch.randn(1, 1, 4, 4).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (2, 2, 3, 3)
    input_tensor = torch.randn(1, 3, 8, 8).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (1, 0, 0, 0)
    input_tensor = torch.randn(1, 1, 5, 5).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (1, 1, 1, 1)
    input_tensor = torch.randn(1, 1, 2, 2).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (1, 1, 1, 1)
    input_tensor = torch.randn(1, 10, 6, 6).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (2, 1, 1, 2)
    input_tensor = torch.randn(1, 1, 3, 7).numpy()
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad2d_2"] = reflectionpad2d_inputs()

import torch
import numpy as np
import copy

def replicationpad3d_inputs():
    list_of_inputs = []
    
    padding = 3
    input_tensor = np.random.randn(16, 3, 8, 320, 480).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 1
    input_tensor = np.random.randn(3, 10, 20, 30).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 0
    input_tensor = np.random.randn(2, 5, 4, 6, 8).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 10
    input_tensor = np.random.randn(1, 1, 5, 5, 5).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 2
    input_tensor = np.random.randn(8, 16, 12, 24, 36).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 5
    input_tensor = np.random.randn(1, 15, 15, 15).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 1
    input_tensor = np.random.randn(4, 32, 3, 3, 3).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 2
    input_tensor = np.random.randn(32, 8, 16, 32, 64).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 4
    input_tensor = np.random.randn(64, 7, 14, 14).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 3
    input_tensor = np.random.randn(2, 3, 5, 10, 15).astype(np.float32)
    input_dict = {
        "padding": padding,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad3d_1"] = replicationpad3d_inputs()

import torch
import numpy as np
import copy

def replicationpad3d_inputs():
    list_of_inputs = []
    
    padding = (3, 3, 3, 3, 3, 3)
    input_tensor = torch.randn(16, 3, 8, 320, 480).numpy()
    input_dict = {"padding": padding, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (3, 3, 6, 6, 1, 1)
    input_tensor = torch.randn(16, 3, 8, 320, 480).numpy()
    input_dict = {"padding": padding, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (1, 2, 3, 4, 5, 6)
    input_tensor = torch.randn(2, 10, 10, 10).numpy()
    input_dict = {"padding": padding, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (0, 0, 2, 2, 0, 0)
    input_tensor = torch.randn(1, 1, 5, 5, 5).numpy()
    input_dict = {"padding": padding, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (1, 1, 1, 1, 1, 1)
    input_tensor = torch.randn(1, 1, 2, 2, 2).numpy()
    input_dict = {"padding": padding, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (10, 10, 10, 10, 10, 10)
    input_tensor = torch.randn(2, 3, 5, 5, 5).numpy()
    input_dict = {"padding": padding, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (0, 0, 0, 0, 0, 0)
    input_tensor = torch.randn(4, 8, 16, 32, 64).numpy()
    input_dict = {"padding": padding, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (2, 3, 1, 4, 0, 1)
    input_tensor = torch.randn(5, 4, 8, 16).numpy()
    input_dict = {"padding": padding, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (1, 1, 2, 2, 3, 3)
    input_tensor = torch.randn(1, 1, 4, 4, 4).numpy()
    input_dict = {"padding": padding, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (5, 5, 3, 3, 2, 2)
    input_tensor = torch.randn(8, 16, 10, 20, 30).numpy()
    input_dict = {"padding": padding, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad3d_2"] = replicationpad3d_inputs()

import torch
import numpy as np
import copy

def silu_inputs():
    list_of_inputs = []
    
    input_dict = {
        "inplace": False,
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inplace": True,
        "input": np.array([0.5, -0.5, 1.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inplace": False,
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inplace": True,
        "input": np.array([[-1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inplace": False,
        "input": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inplace": True,
        "input": np.random.randn(2, 3, 4).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inplace": False,
        "input": np.random.randn(2, 3, 4, 5).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inplace": False,
        "input": np.array([5.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inplace": False,
        "input": np.array([0.0, 0.0, 0.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inplace": False,
        "input": np.array([-10.0, -5.0, -1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inplace": True,
        "input": np.array([10.0, 20.0, 30.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "inplace": False,
        "input": np.random.randn(1, 2, 3, 4, 5).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.SiLU"] = silu_inputs()

import torch
import numpy as np
import copy

def softmax_inputs():
    list_of_inputs = []
    
    input_tensor = np.random.randn(2, 3).astype(np.float32)
    input_dict = {
        "dim": 1,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(4, 5).astype(np.float32)
    input_dict = {
        "dim": 0,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
    input_dict = {
        "dim": 0,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "dim": 2,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(3, 5, 7).astype(np.float32)
    input_dict = {
        "dim": 1,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "dim": 3,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]]).astype(np.float32)
    input_dict = {
        "dim": 1,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(3, 6).astype(np.float32)
    input_dict = {
        "dim": -1,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 4, 3).astype(np.float32)
    input_dict = {
        "dim": -2,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(1, 2, 3, 2, 4).astype(np.float32)
    input_dict = {
        "dim": 4,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(10, 100).astype(np.float32)
    input_dict = {
        "dim": 1,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.Softmax"] = softmax_inputs()

import torch
import numpy as np
import copy

def softmin_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dim = 1
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    dim = 0
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dim = 0
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32)
    dim = 2
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(4, 5, 6).astype(np.float32)
    dim = 1
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 3, 4, 5).astype(np.float32)
    dim = 3
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
    dim = 1
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    dim = 0
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(3, 2, 4).astype(np.float32)
    dim = 0
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    dim = 1
    input_dict = {"dim": dim, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.Softmin"] = softmin_inputs()

import torch
import numpy as np
import copy

def softshrink_inputs():
    list_of_inputs = []
    
    lambd = 0.5
    input_tensor = np.array([1.0, -2.0, 0.3, -0.3, 0.0])
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 1.0
    input_tensor = np.array([[2.5, -3.0, 0.5], [1.5, -1.5, 0.0]])
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 0.0
    input_tensor = np.array([0.1, -0.1, 1.0, -1.0])
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 2.0
    input_tensor = np.array([[[5.0, -5.0], [1.0, -1.0]], [[3.0, -3.0], [0.5, -0.5]]])
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 0.25
    input_tensor = np.array([1.5])
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 1.5
    input_tensor = np.array([10.0, -10.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0])
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 0.1
    input_tensor = np.random.randn(2, 2, 2, 2)
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 3.0
    input_tensor = np.array([3.0, -3.0, 3.1, -3.1, 2.9, -2.9])
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 0.5
    input_tensor = np.random.randn(10, 10)
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    lambd = 0.75
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    input_dict = {
        "lambd": lambd,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.Softshrink"] = softshrink_inputs()

import torch
import numpy as np
import copy

def celu_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([1.0, 2.0, -1.0, -2.0]),
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, -1.0], [2.0, -2.0]]),
        "alpha": 2.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[1.0, -1.0], [0.5, -0.5]], [[0.0, 1.0], [-0.5, 0.5]]]),
        "alpha": 0.5,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0.1, 0.5, 1.0, 2.0, 5.0]),
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([-0.1, -0.5, -1.0, -2.0, -5.0]),
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0.0, 1.0, -1.0, 0.0, 2.0]),
        "alpha": 1.5,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1.0, -1.0, 2.0, -2.0]),
        "alpha": 10.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0.001, -0.001, 0.01, -0.01]),
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(2, 3, 4, 5).astype(np.float32),
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1.5]),
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(10, 10).astype(np.float32),
        "alpha": 0.1,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.celu"] = celu_inputs()

import torch
import numpy as np
import copy

def dropout_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "p": 0.5,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(3, 4).astype(np.float32),
        "p": 0.8,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.ones((2, 3, 4), dtype=np.float32),
        "p": 0.1,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(2, 3, 8, 8).astype(np.float32),
        "p": 0.3,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "p": 0.5,
        "training": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(5, 5).astype(np.float32),
        "p": 0.0,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "p": 1.0,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(1, 2, 3, 4, 5).astype(np.float32),
        "p": 0.4,
        "training": True,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([-1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32),
        "p": 0.6,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.random.randn(10, 10).astype(np.float64),
        "p": 0.25,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.dropout"] = dropout_inputs()

import torch
import numpy as np
import copy

def hardshrink_inputs():
    list_of_inputs = []
    
    input = torch.tensor([0.2, 0.8, -0.3, -0.7, 1.5]).numpy()
    lambd = 0.5
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[1.0, -0.5, 0.3], [2.0, -1.5, 0.0]]).numpy()
    lambd = 0.8
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 3, 4).numpy()
    lambd = 0.1
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([0.1, 0.2, -0.1, -0.2, 0.0]).numpy()
    lambd = 0.5
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([2.0, -3.0, 5.0, -4.0]).numpy()
    lambd = 0.5
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 3, 8, 8).numpy()
    lambd = 1.0
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor(0.3).numpy()
    lambd = 0.5
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([1.0, -1.0, 0.5, -0.5]).numpy()
    lambd = 0.0
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([5.0, -5.0, 3.0, -3.0, 2.0]).numpy()
    lambd = 4.0
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([-1.0, -2.0, -3.0, -0.3, -5.0]).numpy()
    lambd = 0.5
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.ones((5, 5)).numpy()
    lambd = 0.3
    input_dict = {
        "input": input,
        "lambd": lambd
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.hardshrink"] = hardshrink_inputs()

import torch
import numpy as np
import copy

def hardswish_inputs():
    list_of_inputs = []
    
    input_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([-5.0, -3.0, -1.0, 0.0], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([[-4.0, -2.0], [1.0, 3.5]], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([[[0.5, 1.5], [2.5, 3.5]], [[4.0, -1.0], [-2.0, -3.0]]], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([-3.5, -3.0, -2.5, 2.5, 3.0, 3.5], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.zeros((3, 3), dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([2.5], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([-10.0, -20.0, -100.0], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_arr = np.array([10.0, 20.0, 100.0], dtype=np.float32)
    input_dict = {
        "input": input_arr,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.hardswish"] = hardswish_inputs()

import torch
import numpy as np
import copy

def prelu_inputs():
    list_of_inputs = []
    
    input = torch.tensor([1.0, -2.0, 3.0, -4.0]).numpy()
    weight = torch.tensor(0.25).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[1.0, -1.0], [2.0, -2.0]]).numpy()
    weight = torch.tensor(0.1).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 3, 4).numpy()
    weight = torch.tensor([0.1, 0.2, 0.3]).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 5, 3, 3).numpy()
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    weight = torch.tensor(0.5).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]).numpy()
    weight = torch.tensor(0.3).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(1, 4, 2, 2, 2).numpy()
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor(-5.0).numpy()
    weight = torch.tensor(0.15).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[1.0, -1.0], [-2.0, 2.0]]).numpy()
    weight = torch.tensor(-0.1).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(10, 8, 20).numpy()
    weight = torch.tensor([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.prelu"] = prelu_inputs()

import torch
import numpy as np
import copy

def relu6_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-1.0, -2.0, -3.0, 0.0, 1.0, 2.0])
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    inplace = True
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-5.0, -10.0, -15.0, -20.0])
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([5.5, 5.9, 6.0, 6.1, 6.5, 7.0])
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 3, 4, 4).astype(np.float32)
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([10.0, 20.0, 50.0, 100.0])
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.zeros((3, 3))
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([3.5])
    inplace = False
    input_dict = {"input": input_tensor, "inplace": inplace}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.relu6"] = relu6_inputs()

import torch
import numpy as np
import copy

def selu_inputs():
    list_of_inputs = []
    
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([[-1.0, -0.5, 0.0], [0.5, 1.0, 1.5]]).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 3, 5, 5).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor(2.5).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.zeros(3, 3).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(10, 10).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([0.001, -0.001, 0.0001, -0.0001]).numpy()
    input_dict = {
        "input": input_tensor,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.selu"] = selu_inputs()

import torch
import numpy as np
import copy

def constant_inputs():
    list_of_inputs = []
    
    tensor = torch.zeros(5).numpy()
    val = 3.5
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.zeros((3, 4)).numpy()
    val = -2.0
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.ones((2, 3, 4)).numpy()
    val = 0.0
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.randn((2, 2, 2, 2)).numpy()
    val = 0.001
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.zeros(1).numpy()
    val = 100.0
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.zeros((10, 10)).numpy()
    val = -50.5
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.empty(8).numpy()
    val = 1.234567
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.zeros((1, 2, 3, 4, 5)).numpy()
    val = -0.0001
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.zeros((5, 5)).numpy()
    val = 1.0
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.zeros((4, 3, 2)).numpy()
    val = -1.0
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.zeros(1000).numpy()
    val = 0.5
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = torch.zeros((7, 3)).numpy()
    val = 42.0
    input_dict = {
        "tensor": tensor,
        "val": val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.init.constant_"] = constant_inputs()

import torch
import numpy as np
import copy

def numel_inputs():
    list_of_inputs = []
    
    # Input 1: 1D tensor
    input_dict = {
        "input": torch.randn(5).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D tensor
    input_dict = {
        "input": torch.zeros(4, 4).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 5D tensor
    input_dict = {
        "input": torch.randn(1, 2, 3, 4, 5).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor
    input_dict = {
        "input": torch.ones(2, 3, 4).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: scalar (0D tensor)
    input_dict = {
        "input": torch.tensor(42.0).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 1D tensor with single element
    input_dict = {
        "input": torch.tensor([1.0]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 4D tensor
    input_dict = {
        "input": torch.randn(2, 3, 4, 5).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: tensor with negative values
    input_dict = {
        "input": torch.tensor([-1.0, -2.0, -3.0, -4.0]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D tensor with different shape
    input_dict = {
        "input": torch.randn(10, 20).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 6D tensor
    input_dict = {
        "input": torch.randn(1, 2, 2, 2, 2, 2).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: 3D tensor with larger dimensions
    input_dict = {
        "input": torch.zeros(5, 6, 7).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: 1D tensor with many elements
    input_dict = {
        "input": torch.arange(100).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.numel"] = numel_inputs()

import torch
import numpy as np
import copy

def permute_inputs():
    list_of_inputs = []
    
    input_tensor = torch.randn(2, 3, 5).numpy()
    dims = (2, 0, 1)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(4, 6).numpy()
    dims = (1, 0)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    dims = (3, 1, 0, 2)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(10).numpy()
    dims = (0,)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(3, 4, 5).numpy()
    dims = (0, 1, 2)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 3, 4, 5, 6).numpy()
    dims = (4, 2, 0, 3, 1)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(10, 20).numpy()
    dims = (1, 0)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.ones(3, 3, 3).numpy()
    dims = (1, 2, 0)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 2, 3, 4).numpy()
    dims = (0, 2, 3, 1)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.zeros(5, 10, 15).numpy()
    dims = (2, 1, 0)
    input_dict = {
        "input": input_tensor,
        "dims": dims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.permute"] = permute_inputs()

import torch
import numpy as np
import copy

def positive_inputs():
    list_of_inputs = []
    
    # Input 1: 1D tensor with positive and negative values
    input = torch.randn(5).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D tensor
    input = torch.randn(3, 4).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor
    input = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: scalar tensor
    input = torch.tensor(5.0).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: tensor with all negative values
    input = torch.tensor([-1.0, -2.0, -3.0, -4.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: tensor with all positive values
    input = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: tensor with zeros
    input = torch.zeros(4, 5).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: tensor with ones
    input = torch.ones(3, 3).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D tensor
    input = torch.randn(2, 2, 2, 2).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: large 1D tensor
    input = torch.randn(100).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: integer tensor
    input = torch.tensor([1, -2, 3, -4, 5]).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: float64 tensor
    input = torch.tensor([1.5, -2.5, 3.5], dtype=torch.float64).numpy()
    input_dict = {"input": input}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.positive"] = positive_inputs()

import torch
import copy
import numpy as np

def reshape_inputs():
    list_of_inputs = []
    
    input = torch.arange(4.).numpy()
    shape = (2, 2)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([[0, 1], [2, 3]]).numpy()
    shape = (-1,)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.arange(24).reshape(2, 3, 4).numpy()
    shape = (6, 4)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.arange(12.).numpy()
    shape = (2, 2, 3)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.ones((4, 5)).numpy()
    shape = (5, 4)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.arange(60).numpy()
    shape = (3, -1, 5)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 3, 4, 5).numpy()
    shape = (6, 20)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([5.0]).numpy()
    shape = (1, 1)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(3, 4, 5).numpy()
    shape = (-1,)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.arange(48).numpy()
    shape = (2, 2, 3, 4)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.tensor([-1, -2, -3, -4, -5, -6]).numpy()
    shape = (2, 3)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(100).numpy()
    shape = (10, 10)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.reshape"] = reshape_inputs()

import torch
import copy
import numpy as np

def round_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([4.7, -2.3, 9.1, -7.7]),
        "decimals": 0,
        "out": np.array([0.0, 0.0, 0.0, 0.0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([-0.5, 0.5, 1.5, 2.5]),
        "decimals": 0,
        "out": np.array([0.0, 0.0, 0.0, 0.0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0.1234567]),
        "decimals": 3,
        "out": np.array([0.0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([1200.1234567]),
        "decimals": -3,
        "out": np.array([0.0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[1.234, 2.567], [3.891, 4.123]]),
        "decimals": 1,
        "out": np.array([[0.0, 0.0], [0.0, 0.0]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[12345.678, 98765.432]]]),
        "decimals": -2,
        "out": np.array([[[0.0, 0.0]]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([-3.14159, -2.71828, -1.41421]),
        "decimals": 2,
        "out": np.array([0.0, 0.0, 0.0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[10.5, 20.5, 30.5], [40.5, 50.5, 60.5]]),
        "decimals": 0,
        "out": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([3.141592653589793]),
        "decimals": 5,
        "out": np.array([0.0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([100.456, 200.789, 300.123]),
        "decimals": -1,
        "out": np.array([0.0, 0.0, 0.0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.round"] = round_inputs()

import torch
import copy
import numpy as np

def set_num_interop_threads_inputs():
    list_of_inputs = []
    
    # Input 1: typical small positive number
    input_dict = {
        "num": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.set_num_interop_threads"] = set_num_interop_threads_inputs()

import torch
import numpy as np
import copy

def sin_inputs():
    list_of_inputs = []
    
    input_tensor = torch.tensor([0.0, 1.5708, 3.1416, 4.7124]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([-0.5461, -1.2, -2.7266, -0.2746]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([1.5708]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.zeros(5).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 2, 2, 2).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([10.0, 20.0, 30.0, 40.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([0.001, 0.002, 0.003]).numpy()
    input_dict = {
        "input": input_tensor,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out_tensor = torch.empty(3).numpy()
    input_dict = {
        "input": input_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.sin"] = sin_inputs()

import torch
import numpy as np
import copy

def special_erfc_inputs():
    list_of_inputs = []
    
    input_dict = {
        "input": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "out": np.empty(3, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([-2.0, -1.0, -0.5], dtype=np.float32),
        "out": np.empty(3, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32),
        "out": np.empty((2, 2), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32),
        "out": np.empty((2, 2, 2), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array(1.5, dtype=np.float32),
        "out": np.empty((), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "out": np.empty(4, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([5.0, 10.0, 15.0], dtype=np.float32),
        "out": np.empty(3, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0.25, 0.75, 1.25], dtype=np.float64),
        "out": np.empty(3, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0.001, 0.01, 0.1], dtype=np.float32),
        "out": np.empty(3, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([[[[0.5, 1.0], [1.5, 2.0]]]], dtype=np.float32),
        "out": np.empty((1, 1, 2, 2), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.special.erfc"] = special_erfc_inputs()

import torch
import copy
import numpy as np

def i0e_inputs():
    list_of_inputs = []
    
    input_val = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {
        "input": input_val,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {
        "input": input_val,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    out = torch.empty(2, 2).numpy()
    input_dict = {
        "input": input_val,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor(5.0).numpy()
    out = torch.empty(()).numpy()
    input_dict = {
        "input": input_val,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.ones((2, 3, 4)).numpy()
    out = torch.empty(2, 3, 4).numpy()
    input_dict = {
        "input": input_val,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([0.0, 1.0, 2.0]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {
        "input": input_val,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([-5.0, 0.0, 5.0]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {
        "input": input_val,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([10.0, 20.0, 30.0]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {
        "input": input_val,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([0.001, 0.01, 0.1]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {
        "input": input_val,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]]).numpy()
    out = torch.empty(2, 3).numpy()
    input_dict = {
        "input": input_val,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([0.5, 1.5, 2.5]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {
        "input": input_val,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs


generated_inputs["torch.special.i0e"] = i0e_inputs()

import torch
import numpy as np
import copy

def i1e_inputs():
    list_of_inputs = []
    
    input_val = torch.tensor([1.0, 2.0, 3.0]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([-5.0, 0.0, 5.0]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    out = torch.empty(2, 2).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]]).numpy()
    out = torch.empty(2, 2, 2).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor(2.5).numpy()
    out = torch.empty(()).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([10.0, 20.0, 30.0]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([0.001, 0.01, 0.1]).numpy()
    out = torch.empty(3).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]]).numpy()
    out = torch.empty(2, 3).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.tensor([0.0]).numpy()
    out = torch.empty(1).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_val = torch.ones((2, 2, 2, 2)).numpy()
    out = torch.empty(2, 2, 2, 2).numpy()
    input_dict = {"input": input_val, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.special.i1e"] = i1e_inputs()

import torch
import numpy as np
import copy

def sinc_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([0.0, 1.0, 2.0, 3.0])
    out = np.empty(4)
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    out = np.empty(7)
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.array([[0.5, 1.5], [2.5, 3.5]])
    out = np.empty((2, 2))
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    out = np.empty((2, 2, 2))
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.array(1.5)
    out = np.empty(())
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.array([10.0, 20.0, 30.0, 40.0])
    out = np.empty(4)
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.array([0.01, 0.1, 0.5, 0.9])
    out = np.empty(4)
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.array([[-1.0, -0.5], [0.5, 1.0]])
    out = np.empty((2, 2))
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.linspace(-5.0, 5.0, 20)
    out = np.empty(20)
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    input_tensor = np.ones((2, 2, 2, 2)) * 0.5
    out = np.empty((2, 2, 2, 2))
    list_of_inputs.append(copy.deepcopy({"input": input_tensor, "out": out}))
    
    return list_of_inputs

generated_inputs["torch.special.sinc"] = sinc_inputs()

import torch
import copy
import numpy as np

def take_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([[4, 3, 5], [6, 7, 8]])
    index = np.array([0, 2, 5], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([10, 20, 30, 40, 50])
    index = np.array([2], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    index = np.array([0, 3, 7], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.arange(100)
    index = np.array([5, 10, 50, 99, 0], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
    index = np.array([[0, 1], [2, 5]], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([42])
    index = np.array([0], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.ones((2, 2, 2, 2))
    index = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    index = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[-5, -10, -15], [-20, -25, -30]])
    index = np.array([1, 4], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.arange(24).reshape(4, 6)
    index = np.array([23, 20, 15, 10, 5, 0], dtype=np.int64)
    input_dict = {"input": input_tensor, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.take"] = take_inputs()

import torch
import copy
import numpy as np

def tril_inputs():
    list_of_inputs = []
    
    input_tensor = torch.randn(3, 3).numpy()
    diagonal = 0
    out = torch.zeros(3, 3).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(4, 6).numpy()
    diagonal = 1
    out = torch.zeros(4, 6).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(4, 6).numpy()
    diagonal = -1
    out = torch.zeros(4, 6).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(5, 5).numpy()
    diagonal = 2
    out = torch.zeros(5, 5).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 2).numpy()
    diagonal = -1
    out = torch.zeros(2, 2).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(6, 4).numpy()
    diagonal = 0
    out = torch.zeros(6, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(3, 3).numpy()
    diagonal = 5
    out = torch.zeros(3, 3).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(4, 4).numpy()
    diagonal = -3
    out = torch.zeros(4, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 1).numpy()
    diagonal = 0
    out = torch.zeros(1, 1).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(7, 3).numpy()
    diagonal = -2
    out = torch.zeros(7, 3).numpy()
    input_dict = {
        "input": input_tensor,
        "diagonal": diagonal,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.tril"] = tril_inputs()

import torch
import numpy as np
import copy

def unique_consecutive_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([1, 1, 2, 2, 3, 1, 1, 2])
    input_dict = {
        "input": input_tensor,
        "return_inverse": False,
        "return_counts": False,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([5, 5, 5, 3, 3, 1])
    input_dict = {
        "input": input_tensor,
        "return_inverse": True,
        "return_counts": False,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([10, 10, 20, 20, 20, 30])
    input_dict = {
        "input": input_tensor,
        "return_inverse": False,
        "return_counts": True,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([7, 7, 8, 8, 9, 9, 9])
    input_dict = {
        "input": input_tensor,
        "return_inverse": True,
        "return_counts": True,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1, 2], [1, 2], [3, 4], [3, 4]])
    input_dict = {
        "input": input_tensor,
        "return_inverse": False,
        "return_counts": False,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1, 1, 2, 2], [3, 3, 4, 4]])
    input_dict = {
        "input": input_tensor,
        "return_inverse": False,
        "return_counts": False,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[5, 6], [5, 6], [7, 8]])
    input_dict = {
        "input": input_tensor,
        "return_inverse": True,
        "return_counts": False,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[2, 2, 3], [4, 4, 5]])
    input_dict = {
        "input": input_tensor,
        "return_inverse": False,
        "return_counts": True,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1, 2]], [[1, 2]], [[3, 4]]])
    input_dict = {
        "input": input_tensor,
        "return_inverse": False,
        "return_counts": False,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1.5, 1.5, 2.5, 2.5, 3.5])
    input_dict = {
        "input": input_tensor,
        "return_inverse": True,
        "return_counts": True,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.unique_consecutive"] = unique_consecutive_inputs()

