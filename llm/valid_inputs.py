import torch, copy
import numpy as np
import torch.nn as nn
import torch.jit
def adaptive_avg_pool2d_inputs():
    list_of_inputs = []
    # Input 1: Basic float input with a single integer output size
    input1 = torch.randn(1, 3, 32, 32).numpy()
    output_size1 = 16
    input_dict1 = {"input": input1, "output_size": output_size1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Different input size, tuple output size
    input2 = torch.randn(2, 4, 64, 64).numpy()
    output_size2 = (8, 8)
    input_dict2 = {"input": input2, "output_size": output_size2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Single channel input
    input3 = torch.randn(1, 1, 128, 128).numpy()
    output_size3 = (32, 32)
    input_dict3 = {"input": input3, "output_size": output_size3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Large input size, single int
    input6 = torch.randn(4, 16, 256, 256).numpy()
    output_size6 = 64
    input_dict6 = {"input": input6, "output_size": output_size6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 5: 3D input
    input7 = torch.randn(3, 32, 32).numpy()
    output_size7 = (16, 16)
    input_dict7 = {"input": input7, "output_size": output_size7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def adaptive_max_pool2d_inputs():
    list_of_inputs = []
    input1 = np.random.randn(2, 3, 20, 20).astype(np.float32)
    output_size1 = (5, 7)
    input_dict1 = {"input": input1, "output_size": output_size1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.random.randn(1, 1, 32, 32).astype(np.float64)
    output_size2 = (10, 10)
    input_dict2 = {"input": input2, "output_size": output_size2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.random.randn(4, 5, 16, 16).astype(np.float16)
    output_size3 = (8, 8)
    input_dict3 = {"input": input3, "output_size": output_size3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.randn(3, 7, 24, 24).astype(np.float32)
    output_size4 = (12, 6)
    input_dict4 = {"input": input4, "output_size": output_size4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.random.randn(1, 2, 28, 28).astype(np.float64)
    output_size5 = (14, 7)
    input_dict5 = {"input": input5, "output_size": output_size5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.random.randn(2, 4, 10, 10).astype(np.float32)
    output_size6 = 7
    input_dict6 = {"input": input6, "output_size": output_size6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.random.randn(1, 3, 15, 15).astype(np.float64)
    output_size7 = 5
    input_dict7 = {"input": input7, "output_size": output_size7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def addcdiv_inputs():
    list_of_inputs = []
    input_tensor = np.random.randn(3, 4).astype(np.float32)
    tensor1 = np.random.randn(3, 4).astype(np.float32)
    tensor2 = np.random.randn(3, 4).astype(np.float32)
    value = 2.0
    input_dict = {
        "input": input_tensor,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = np.random.randn(2, 3, 4).astype(np.float64)
    tensor1 = np.random.randn(2, 3, 4).astype(np.float64)
    tensor2 = np.random.randn(2, 3, 4).astype(np.float64)
    value = -1.0
    input_dict = {
        "input": input_tensor,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    tensor1 = np.array([4.0, 5.0, 6.0]).astype(np.float32)
    tensor2 = np.array([7.0, 8.0, 9.0]).astype(np.float32)
    value = 1.5
    input_dict = {
        "input": input_tensor,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = np.random.randn(1, 5, 5).astype(np.float32)
    tensor1 = np.random.randn(1, 5, 5).astype(np.float32)
    tensor2 = np.random.randn(1, 5, 5).astype(np.float32)
    value = -0.5
    input_dict = {
        "input": input_tensor,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = np.random.randn(2, 2, 2, 2).astype(np.float32)
    tensor1 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    tensor2 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    value = 1.0
    input_dict = {
        "input": input_tensor,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def addmm_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensors
    input1 = np.random.randn(3, 5).astype(np.float32)
    mat1_1 = np.random.randn(3, 4).astype(np.float32)
    mat2_1 = np.random.randn(4, 5).astype(np.float32)
    input_dict1 = {"input": input1, "mat1": mat1_1, "mat2": mat2_1, "beta": 1.0, "alpha": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Int tensors with different beta and alpha
    input2 = np.random.randint(-5, 5, size=(2, 3)).astype(np.int32)
    mat1_2 = np.random.randint(-5, 5, size=(2, 4)).astype(np.int32)
    mat2_2 = np.random.randint(-5, 5, size=(4, 3)).astype(np.int32)
    input_dict2 = {"input": input2, "mat1": mat1_2, "mat2": mat2_2, "beta": 0.5, "alpha": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Negative values and zero beta
    input3 = np.random.randn(4, 2).astype(np.float64) * -1
    mat1_3 = np.random.randn(4, 3).astype(np.float64) * -1
    mat2_3 = np.random.randn(3, 2).astype(np.float64) * -1
    input_dict3 = {"input": input3, "mat1": mat1_3, "mat2": mat2_3, "beta": 0.0, "alpha": 1.5}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Larger matrices
    input4 = np.random.randn(10, 10).astype(np.float32)
    mat1_4 = np.random.randn(10, 5).astype(np.float32)
    mat2_4 = np.random.randn(5, 10).astype(np.float32)
    input_dict4 = {"input": input4, "mat1": mat1_4, "mat2": mat2_4, "beta": 0.8, "alpha": 0.7}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Complex tensors
    input5 = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    mat1_5 = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(np.complex64)
    mat2_5 = (np.random.randn(3, 2) + 1j * np.random.randn(3, 2)).astype(np.complex64)
    input_dict5 = {"input": input5, "mat1": mat1_5, "mat2": mat2_5, "beta": 1.0, "alpha": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Different shapes
    input6 = np.random.randn(5, 7).astype(np.float32)
    mat1_6 = np.random.randn(5, 2).astype(np.float32)
    mat2_6 = np.random.randn(2, 7).astype(np.float32)
    input_dict6 = {"input": input6, "mat1": mat1_6, "mat2": mat2_6, "beta": 0.2, "alpha": 0.9}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def addmv_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensors
    input = torch.randn(3).numpy()
    mat = torch.randn(3, 2).numpy()
    vec = torch.randn(2).numpy()
    input_dict = {'input': input, 'mat': mat, 'vec': vec, 'beta': 1.0, 'alpha': 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Different shapes, beta and alpha
    input = torch.randn(5).numpy()
    mat = torch.randn(5, 4).numpy()
    vec = torch.randn(4).numpy()
    input_dict = {'input': input, 'mat': mat, 'vec': vec, 'beta': 0.5, 'alpha': 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Negative values
    input = torch.randn(4).numpy() * -1
    mat = torch.randn(4, 3).numpy() * -1
    vec = torch.randn(3).numpy() * -1
    input_dict = {'input': input, 'mat': mat, 'vec': vec, 'beta': 1.0, 'alpha': 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4:  Zero values
    input = torch.zeros(2).numpy()
    mat = torch.zeros(2, 5).numpy()
    vec = torch.zeros(5).numpy()
    input_dict = {'input': input, 'mat': mat, 'vec': vec, 'beta': 1.0, 'alpha': 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5:  Double type
    input = torch.randn(6, dtype=torch.float64).numpy()
    mat = torch.randn(6, 2, dtype=torch.float64).numpy()
    vec = torch.randn(2, dtype=torch.float64).numpy()
    input_dict = {'input': input, 'mat': mat, 'vec': vec, 'beta': 1.0, 'alpha': 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Beta and alpha are zeros.
    input = torch.randn(3).numpy()
    mat = torch.randn(3, 2).numpy()
    vec = torch.randn(2).numpy()
    input_dict = {'input': input, 'mat': mat, 'vec': vec, 'beta': 0.0, 'alpha': 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def addr_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 3).numpy()
    vec11 = torch.randn(3).numpy()
    vec21 = torch.randn(3).numpy()
    beta1 = 1.0
    alpha1 = 1.0
    input_dict1 = {
        "input": input1,
        "vec1": vec11,
        "vec2": vec21,
        "beta": beta1,
        "alpha": alpha1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.zeros(2, 4).numpy()
    vec12 = torch.ones(2).numpy()
    vec22 = torch.ones(4).numpy()
    beta2 = 0.5
    alpha2 = 2.0
    input_dict2 = {
        "input": input2,
        "vec1": vec12,
        "vec2": vec22,
        "beta": beta2,
        "alpha": alpha2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.ones(4, 5).numpy()
    vec13 = torch.arange(1, 5).float().numpy()
    vec23 = torch.arange(1, 6).float().numpy()
    beta3 = 0.0
    alpha3 = -1.0
    input_dict3 = {
        "input": input3,
        "vec1": vec13,
        "vec2": vec23,
        "beta": beta3,
        "alpha": alpha3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randint(-5, 5, (2, 2), dtype=torch.int32).numpy()
    vec14 = torch.tensor([-1, 2], dtype=torch.int32).numpy()
    vec24 = torch.tensor([3, -4], dtype=torch.int32).numpy()
    beta4 = 1
    alpha4 = 1
    input_dict4 = {
        "input": input4,
        "vec1": vec14,
        "vec2": vec24,
        "beta": beta4,
        "alpha": alpha4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(5, 1).numpy()
    vec15 = torch.randn(5).numpy()
    vec25 = torch.randn(1).numpy()
    beta5 = 0.25
    alpha5 = 0.75
    input_dict5 = {
        "input": input5,
        "vec1": vec15,
        "vec2": vec25,
        "beta": beta5,
        "alpha": alpha5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def allclose_inputs():
    list_of_inputs = []
    # Case 1: Simple float tensors
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input2 = np.array([1.001, 2.002, 3.003], dtype=np.float32)
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 0.01,
        "rtol": 0.001,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Integer tensors
    input1 = np.array([1, 2, 3], dtype=np.int32)
    input2 = np.array([1, 2, 4], dtype=np.int32)
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 1,
        "rtol": 0.0,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Multidimensional tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input2 = np.array([[1.0, 2.0], [3.0, 4.1]], dtype=np.float32)
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 0.0,
        "rtol": 0.01,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: Negative values
    input1 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input2 = np.array([-1.001, -2.002, -3.003], dtype=np.float32)
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 0.01,
        "rtol": 0.001,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Tensors with NaN
    input1 = np.array([1.0, float('nan'), 3.0], dtype=np.float32)
    input2 = np.array([1.0, float('nan'), 3.0], dtype=np.float32)
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 0.0,
        "rtol": 0.0,
        "equal_nan": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def alpha_dropout_inputs():
    list_of_inputs = []
    input1 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "p": 0.5,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.random.randn(1, 5, 5, 5).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "p": 0.2,
        "training": False,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.random.randn(10).astype(np.float16)
    input_dict3 = {
        "input": input3,
        "p": 0.8,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.random.randn(4, 4).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "p": 0.3,
        "training": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.random.randn(2, 2, 2, 2, 2).astype(np.float64)
    input_dict5 = {
        "input": input5,
        "p": 0.1,
        "training": True,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.random.randn(3, 3).astype(np.float32) * -1
    input_dict6 = {
        "input": input6,
        "p": 0.6,
        "training": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.random.randn(1, 2, 3, 4).astype(np.float32)
    input_dict7 = {
        "input": input7,
        "p": 0.0,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def amax_inputs():
    list_of_inputs = []
    # Input 1: 1D float tensor
    input1 = torch.randn(5).numpy()
    dim1 = (0,)
    keepdim1 = False
    input_dict1 = {"input": input1, "dim": dim1, "keepdim": keepdim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D int tensor
    input2 = torch.randint(-10, 10, (3, 4)).numpy()
    dim2 = (0,)
    keepdim2 = True
    input_dict2 = {"input": input2, "dim": dim2, "keepdim": keepdim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 4: 4D float tensor with negative values
    input4 = torch.randn(2, 2, 2, 2).numpy() * -1
    dim4 = (1, 3)
    keepdim4 = True
    input_dict4 = {"input": input4, "dim": dim4, "keepdim": keepdim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 2D int tensor, no dim
    input5 = torch.randint(0, 10, (3, 4)).numpy()
    dim5 = None
    keepdim5 = False
    input_dict5 = {"input": input5, "dim": dim5, "keepdim": keepdim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: 2D float tensor, dim is an int
    input6 = torch.randn(3, 4).numpy()
    dim6 = (1,)
    keepdim6 = False
    input_dict6 = {"input": input6, "dim": dim6, "keepdim": keepdim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: 3D float tensor
    input7 = torch.randn(2, 3, 4).numpy()
    dim7 = (0, 2)
    keepdim7 = True
    input_dict7 = {"input": input7, "dim": dim7, "keepdim": keepdim7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def as_strided_inputs():
    list_of_inputs = []
    # Example 1: Basic 2D float tensor
    input_tensor = torch.randn(5, 7).numpy()
    size = (3, 4)
    stride = (7, 1)
    storage_offset = 0
    input_dict = {"input": input_tensor, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 2: 3D int tensor with offset
    input_tensor = torch.randint(0, 10, (4, 5, 6)).numpy()
    size = (2, 3, 4)
    stride = (30, 6, 1)
    storage_offset = 7
    input_dict = {"input": input_tensor, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 3: 1D tensor
    input_tensor = torch.arange(10).float().numpy()
    size = (5,)
    stride = (2,)
    storage_offset = 1
    input_dict = {"input": input_tensor, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 4: Larger 2D tensor with different strides
    input_tensor = torch.randn(10, 12).numpy()
    size = (5, 5)
    stride = (12, 2)
    storage_offset = 3
    input_dict = {"input": input_tensor, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 5: Bool Tensor
    input_tensor = torch.randint(0, 2, (3, 3), dtype=torch.bool).numpy()
    size = (2, 2)
    stride = (3, 1)
    storage_offset = 0
    input_dict = {"input": input_tensor, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 6: Complex Tensor
    input_tensor = torch.randn(3, 3, dtype=torch.complex64).numpy()
    size = (2, 2)
    stride = (3, 1)
    storage_offset = 0
    input_dict = {"input": input_tensor, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 7: 4D tensor
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    size = (1, 2, 2, 3)
    stride = (60, 20, 5, 1)
    storage_offset = 2
    input_dict = {"input": input_tensor, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def atan2_inputs():
    list_of_inputs = []
    # Test case 1: Basic float tensors
    input1 = np.random.randn(3, 4).astype(np.float32)
    other1 = np.random.randn(3, 4).astype(np.float32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Test case 2: Integer tensors
    input2 = np.random.randint(-5, 5, size=(2, 2)).astype(np.int32)
    other2 = np.random.randint(-5, 5, size=(2, 2)).astype(np.int32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Test case 3: Negative values
    input3 = np.random.randn(5).astype(np.float64) * -1
    other3 = np.random.randn(5).astype(np.float64) * -1
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Test case 4: Different shapes (but broadcastable)
    input4 = np.random.randn(2, 3, 4).astype(np.float32)
    other4 = np.random.randn(4).astype(np.float32)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Test case 5: Scalar values
    input5 = np.random.randn(1).astype(np.float32)[0]
    other5 = np.random.randn(1).astype(np.float32)[0]
    input_dict5 = {"input": np.array(input5), "other": np.array(other5)}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Test case 6: One dimensional arrays
    input6 = np.random.randn(10).astype(np.float32)
    other6 = np.random.randn(10).astype(np.float32)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Test case 7: Zero values
    input7 = np.zeros((3, 3)).astype(np.float32)
    other7 = np.ones((3, 3)).astype(np.float32)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def baddbmm_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensors - Corrected shape
    input1 = np.random.randn(2, 5).astype(np.float32)
    batch1_1 = np.random.randn(2, 2, 3).astype(np.float32)
    batch2_1 = np.random.randn(2, 3, 5).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "batch1": batch1_1,
        "batch2": batch2_1,
        "beta": 1.0,
        "alpha": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Different shapes, beta and alpha - Corrected shape
    input2 = np.random.randn(3, 7).astype(np.float32)
    batch1_2 = np.random.randn(3, 3, 4).astype(np.float32)
    batch2_2 = np.random.randn(3, 4, 7).astype(np.float32)
    input_dict2 = {
        "input": input2,
        "batch1": batch1_2,
        "batch2": batch2_2,
        "beta": 0.5,
        "alpha": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 4: Negative values - Corrected Shape
    input4 = np.random.randn(4, 6).astype(np.float32)
    batch1_4 = np.random.randn(4, 4, 2).astype(np.float32)
    batch2_4 = np.random.randn(4, 2, 6).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "batch1": batch1_4,
        "batch2": batch2_4,
        "beta": -1.0,
        "alpha": -0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Larger batch size - Corrected Shape
    input5 = np.random.randn(6, 8).astype(np.float32)
    batch1_5 = np.random.randn(6, 6, 5).astype(np.float32)
    batch2_5 = np.random.randn(6, 5, 8).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "batch1": batch1_5,
        "batch2": batch2_5,
        "beta": 1.0,
        "alpha": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Different beta and alpha values - Corrected Shape
    input6 = np.random.randn(5, 7).astype(np.float32)
    batch1_6 = np.random.randn(5, 5, 3).astype(np.float32)
    batch2_6 = np.random.randn(5, 3, 7).astype(np.float32)
    input_dict6 = {
        "input": input6,
        "batch1": batch1_6,
        "batch2": batch2_6,
        "beta": 0.7,
        "alpha": 1.3
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def batch_norm_inputs():
    list_of_inputs = []
    # Input 1: Basic 2D input
    input1 = np.random.randn(2, 3).astype(np.float32)
    running_mean1 = np.random.randn(3).astype(np.float32)
    running_var1 = np.random.rand(3).astype(np.float32)
    weight1 = np.random.randn(3).astype(np.float32)
    bias1 = np.random.randn(3).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "running_mean": running_mean1,
        "running_var": running_var1,
        "weight": weight1,
        "bias": bias1,
        "training": False,
        "momentum": 0.1,
        "eps": 1e-5
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 3D input (Batch, Channel, Length)
    input2 = np.random.randn(4, 5, 6).astype(np.float32)
    running_mean2 = np.random.randn(5).astype(np.float32)
    running_var2 = np.random.rand(5).astype(np.float32)
    weight2 = np.random.randn(5).astype(np.float32)
    bias2 = np.random.randn(5).astype(np.float32)
    input_dict2 = {
        "input": input2,
        "running_mean": running_mean2,
        "running_var": running_var2,
        "weight": weight2,
        "bias": bias2,
        "training": True,
        "momentum": 0.2,
        "eps": 1e-8
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 4D input (Batch, Channel, Height, Width)
    input3 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    running_mean3 = np.random.randn(3).astype(np.float32)
    running_var3 = np.random.rand(3).astype(np.float32)
    weight3 = np.random.randn(3).astype(np.float32)
    bias3 = np.random.randn(3).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "running_mean": running_mean3,
        "running_var": running_var3,
        "weight": weight3,
        "bias": bias3,
        "training": False,
        "momentum": 0.15,
        "eps": 1e-6
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Input with negative values and different eps
    input4 = np.random.randn(1, 2, 3, 4).astype(np.float32) * -1
    running_mean4 = np.random.randn(2).astype(np.float32)
    running_var4 = np.random.rand(2).astype(np.float32)
    weight4 = np.random.randn(2).astype(np.float32)
    bias4 = np.random.randn(2).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "running_mean": running_mean4,
        "running_var": running_var4,
        "weight": weight4,
        "bias": bias4,
        "training": True,
        "momentum": 0.3,
        "eps": 1e-3
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Large input size
    input5 = np.random.randn(8, 16, 32, 32).astype(np.float32)
    running_mean5 = np.random.randn(16).astype(np.float32)
    running_var5 = np.random.rand(16).astype(np.float32)
    weight5 = np.random.randn(16).astype(np.float32)
    bias5 = np.random.randn(16).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "running_mean": running_mean5,
        "running_var": running_var5,
        "weight": weight5,
        "bias": bias5,
        "training": False,
        "momentum": 0.05,
        "eps": 1e-7
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def binary_cross_entropy_with_logits_inputs():
    list_of_inputs = []
    input1 = np.random.randn(3, 4).astype(np.float32)
    target1 = np.random.randint(0, 2, size=(3, 4)).astype(np.float32)
    weight1 = np.random.rand(3, 4).astype(np.float32)
    pos_weight1 = np.random.rand(1).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "target": target1,
        "weight": weight1,
        "reduction": 'mean',
        "pos_weight": pos_weight1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.random.randn(5).astype(np.float64)
    target2 = np.random.randint(0, 2, size=(5)).astype(np.float64)
    weight2 = np.random.rand(5).astype(np.float64)
    pos_weight2 = np.random.rand(1).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "target": target2,
        "weight": weight2,
        "reduction": 'sum',
        "pos_weight": pos_weight2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input4 = np.random.randn(1, 5, 5).astype(np.float32)
    target4 = np.random.randint(0, 2, size=(1, 5, 5)).astype(np.float32)
    weight4 = None
    pos_weight4 = None
    input_dict4 = {
        "input": input4,
        "target": target4,
        "weight": weight4,
        "reduction": 'mean',
        "pos_weight": pos_weight4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.random.randn(2, 3).astype(np.float32)
    target5 = np.random.randint(0, 2, size=(2, 3)).astype(np.float32)
    weight5 = None
    pos_weight5 = np.random.rand(1).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "target": target5,
        "weight": weight5,
        "reduction": 'sum',
        "pos_weight": pos_weight5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs
def bincount_inputs():
    list_of_inputs = []
    # Example 1: Basic integer input
    input1 = np.array([1, 2, 2, 3, 3, 3], dtype=np.int64)
    weights1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    minlength1 = 0
    input_dict1 = {"input": input1, "weights": weights1, "minlength": minlength1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Example 2: Input with minlength
    input2 = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    weights2 = np.array([1, 1, 1, 1, 1], dtype=np.float64)
    minlength2 = 7
    input_dict2 = {"input": input2, "weights": weights2, "minlength": minlength2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Example 3: Input with zero weights
    input3 = np.array([0, 1, 0, 1, 2], dtype=np.int64)
    weights3 = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    minlength3 = 0
    input_dict3 = {"input": input3, "weights": weights3, "minlength": minlength3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Example 4: Input with different weights
    input4 = np.array([0, 1, 2, 0, 1, 2, 0], dtype=np.int32)
    weights4 = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=np.float64)
    minlength4 = 0
    input_dict4 = {"input": input4, "weights": weights4, "minlength": minlength4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Example 5: Input with one element
    input5 = np.array([5], dtype=np.int64)
    weights5 = np.array([2.0], dtype=np.float32)
    minlength5 = 10
    input_dict5 = {"input": input5, "weights": weights5, "minlength": minlength5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Example 6: Input with no weights
    input6 = np.array([0, 1, 2, 1, 0], dtype=np.int32)
    weights6 = None
    minlength6 = 0
    input_dict6 = {"input": input6, "weights": weights6, "minlength": minlength6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Example 7: Input with smaller weights
    input7 = np.array([2, 1, 2, 0], dtype=np.int64)
    weights7 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    minlength7 = 5
    input_dict7 = {"input": input7, "weights": weights7, "minlength": minlength7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def bitwise_and_inputs():
    list_of_inputs = []
    input1 = np.array([1, 2, 3, 4], dtype=np.int32)
    other1 = np.array([0, 1, 5, 7], dtype=np.int32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    other2 = np.array([[0, 1], [5, 7]], dtype=np.uint8)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[-1, -2], [-3, -4]], dtype=np.int64)
    other3 = np.array([[0, 1], [5, -7]], dtype=np.int64)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([1, 2, 3, 4], dtype=np.int16)
    other4 = np.array([1], dtype=np.int16)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int8)
    other5 = np.array([[[0, 1], [5, 7]], [[1, 0], [1, 1]]], dtype=np.int8)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([True, False, True], dtype=np.bool_)
    other6 = np.array([False, True, True], dtype=np.bool_)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([1, 2, 3], dtype=np.int32)
    other7 = 2
    input_dict7 = {"input": input7, "other": np.array(other7, dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def bitwise_or_inputs():
    generated_inputs = []
    # Test case 1: Basic integer tensors
    input1 = np.array([1, 2, 3, 4], dtype=np.int32)
    other1 = np.array([4, 3, 2, 1], dtype=np.int32)
    generated_inputs.append({"input": input1, "other": other1})
    # Test case 2: Different shapes, but broadcastable
    input2 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    other2 = np.array([1, 0], dtype=np.int64)
    generated_inputs.append({"input": input2, "other": other2})
    # Test case 3: Scalar value
    input3 = np.array([5, 6, 7, 8], dtype=np.int8)
    other3 = np.array(3, dtype=np.int8)
    generated_inputs.append({"input": input3, "other": other3})
    # Test case 4: Multi-dimensional arrays
    input4 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    other4 = np.array([[[8, 7], [6, 5]], [[4, 3], [2, 1]]], dtype=np.uint8)
    generated_inputs.append({"input": input4, "other": other4})
    
    # Test case 5: Boolean arrays, which should also work (implicitly cast to integers)
    input5 = np.array([True, False, True, False], dtype=bool)
    other5 = np.array([False, True, False, True], dtype=bool)
    generated_inputs.append({"input": input5, "other": other5})
    
    # Test case 6: Negative integers
    input6 = np.array([-1, -2, -3, -4], dtype=np.int32)
    other6 = np.array([4, 3, 2, 1], dtype=np.int32)
    generated_inputs.append({"input": input6, "other": other6})
    # Test case 7: Mixed positive and negative integers
    input7 = np.array([-1, 2, -3, 4], dtype=np.int64)
    other7 = np.array([1, -2, 3, -4], dtype=np.int64)
    generated_inputs.append({"input": input7, "other": other7})
    
    # Test case 8: uint8 array
    input8 = np.array([255, 128, 64, 32], dtype=np.uint8)
    other8 = np.array([1, 2, 4, 8], dtype=np.uint8)
    generated_inputs.append({"input": input8, "other": other8})
    
    return generated_inputs
def bitwise_xor_inputs():
    list_of_inputs = []
    input1 = np.array([1, 2, 3, 4], dtype=np.int32)
    other1 = np.array([5, 6, 7, 8], dtype=np.int32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    other2 = np.array([[5, 6], [7, 8]], dtype=np.int64)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[-1, -2], [-3, -4]], dtype=np.int8)
    other3 = np.array([[5, 6], [7, 8]], dtype=np.int8)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([1, 2, 3, 4], dtype=np.uint8)
    other4 = np.array([5, 6, 7, 8], dtype=np.uint8)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    other5 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int16)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([1, 0, 1, 0], dtype=bool)
    other6 = np.array([0, 1, 0, 1], dtype=bool)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def bmm_inputs():
    list_of_inputs = []
    # Input 1: Basic case with float tensors
    input1 = np.random.randn(10, 3, 4).astype(np.float32)
    mat2_1 = np.random.randn(10, 4, 5).astype(np.float32)
    input_dict1 = {"input": input1, "mat2": mat2_1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Different batch size and dimensions
    input2 = np.random.randn(5, 2, 3).astype(np.float64)
    mat2_2 = np.random.randn(5, 3, 6).astype(np.float64)
    input_dict2 = {"input": input2, "mat2": mat2_2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Using negative values
    input3 = np.random.randn(2, 5, 5).astype(np.float32) * -1
    mat2_3 = np.random.randn(2, 5, 5).astype(np.float32) * -1
    input_dict3 = {"input": input3, "mat2": mat2_3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Different data type (float16)
    input4 = np.random.randn(3, 4, 2).astype(np.float16)
    mat2_4 = np.random.randn(3, 2, 3).astype(np.float16)
    input_dict4 = {"input": input4, "mat2": mat2_4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Larger matrices
    input5 = np.random.randn(1, 128, 256).astype(np.float32)
    mat2_5 = np.random.randn(1, 256, 512).astype(np.float32)
    input_dict5 = {"input": input5, "mat2": mat2_5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Different batch size
    input6 = np.random.randn(32, 8, 16).astype(np.float32)
    mat2_6 = np.random.randn(32, 16, 32).astype(np.float32)
    input_dict6 = {"input": input6, "mat2": mat2_6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Using doubles
    input7 = np.random.randn(4, 10, 10).astype(np.float64)
    mat2_7 = np.random.randn(4, 10, 10).astype(np.float64)
    input_dict7 = {"input": input7, "mat2": mat2_7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def broadcast_shapes_inputs():
    list_of_inputs = []
    # Test case 1: Simple broadcast
    shapes = [(2, 3), (2, 1)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 2: Different dimensions
    shapes = [(5, 4, 3), (3,)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 3: Empty tuple
    shapes = [(5, 4, 3), ()]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 4: More than two shapes
    shapes = [(2, 3, 4), (2, 1, 4), (2, 3, 1)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 5: No broadcasting needed
    shapes = [(2, 3, 4), (2, 3, 4)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 6: Broadcasting with scalar
    shapes = [(5, 4, 3), (1,)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 7: Multiple dimensions requiring broadcasting
    shapes = [(1, 2, 3, 4), (5, 2, 1, 4)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 8: One shape provided
    shapes = [(5,)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def broadcast_to_inputs():
    list_of_inputs = []
    # Case 1: Simple 1D broadcast
    input_tensor = np.array([1, 2, 3])
    shape = [3, 3]
    input_dict = {"input": input_tensor, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Broadcast a scalar to a multi-dimensional tensor
    input_tensor = np.array(5)
    shape = [2, 3, 4]
    input_dict = {"input": input_tensor, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Broadcast a 2D tensor to a 3D tensor
    input_tensor = np.array([[1, 2], [3, 4]])
    shape = [2, 2, 2]
    input_dict = {"input": input_tensor, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: Broadcast a tensor with compatible dimensions
    input_tensor = np.array([[1, 2, 3]])
    shape = [2, 1, 3]
    input_dict = {"input": input_tensor, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Broadcast a tensor with different data type (int)
    input_tensor = np.array([1, 2, 3], dtype=np.int32)
    shape = [3, 3]
    input_dict = {"input": input_tensor, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 6: Broadcast a tensor with different data type (float)
    input_tensor = np.array([1.0, 2.0], dtype=np.float64)
    shape = [3, 2]
    input_dict = {"input": input_tensor, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 7: Broadcast a 3D tensor to a 4D tensor
    input_tensor = np.random.rand(2, 3, 4)
    shape = [5, 2, 3, 4]
    input_dict = {"input": input_tensor, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 8: Broadcast along multiple dimensions
    input_tensor = np.array([[[1], [2]]])
    shape = [2, 1, 2, 3]
    input_dict = {"input": input_tensor, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def cat_inputs():
    list_of_inputs = []
    # Input 1: Basic case with two 2D float tensors
    tensors1 = [torch.randn(2, 3).numpy(), torch.randn(2, 3).numpy()]
    dim1 = 0
    input_dict1 = {"tensors": tensors1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Three 1D int tensors, concatenating along the first dimension
    tensors2 = [torch.randint(0, 10, (5,)).numpy(), torch.randint(0, 10, (5,)).numpy(), torch.randint(0, 10, (5,)).numpy()]
    dim2 = 0
    input_dict2 = {"tensors": tensors2, "dim": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Two 3D tensors with different shapes, concatenating along dim=1
    tensors3 = [torch.randn(2, 3, 4).numpy(), torch.randn(2, 5, 4).numpy()]
    dim3 = 1
    input_dict3 = {"tensors": tensors3, "dim": dim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: One tensor in the list
    tensors4 = [torch.randn(2, 3).numpy()]
    dim4 = 0
    input_dict4 = {"tensors": tensors4, "dim": dim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Two 2D complex tensors
    tensors5 = [torch.randn(2, 3, dtype=torch.complex64).numpy(), torch.randn(2, 3, dtype=torch.complex64).numpy()]
    dim5 = 1
    input_dict5 = {"tensors": tensors5, "dim": dim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Two tensors, negative dim
    tensors6 = [torch.randn(2, 3, 4).numpy(), torch.randn(2, 3, 4).numpy()]
    dim6 = -1
    input_dict6 = {"tensors": tensors6, "dim": dim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: 4D tensors
    tensors7 = [torch.randn(2,3,4,5).numpy(), torch.randn(2,3,4,5).numpy()]
    dim7 = 2
    input_dict7 = {"tensors": tensors7, "dim": dim7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def cdist_inputs():
    list_of_inputs = []
    x1 = np.random.randn(10, 3).astype(np.float32)
    x2 = np.random.randn(5, 3).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = np.random.randn(4, 5).astype(np.float64)
    x2 = np.random.randn(2, 5).astype(np.float64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = np.random.randn(5, 2).astype(np.float32) * -1
    x2 = np.random.randn(3, 2).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = np.random.randn(7, 4).astype(np.float64)
    x2 = np.random.randn(7, 4).astype(np.float64) * -1
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def cholesky_inverse_inputs():
    generated_inputs = []
    # Input 1: Basic float32, lower=False, positive definite
    L = np.array([[2.0, 0.0, 0.0],
                  [0.0, 3.0, 0.0],
                  [0.0, 0.0, 4.0]], dtype=np.float32)
    generated_inputs.append({"L": L, "upper": False})
    # Input 2: float64, upper=True, positive definite
    L = np.array([[2.0, 0.0, 0.0],
                  [0.0, 3.0, 0.0],
                  [0.0, 0.0, 4.0]], dtype=np.float64)
    generated_inputs.append({"L": L, "upper": True})
    # Input 3: Batched input (2 batches), lower=False, positive definite
    L = np.array([[[2.0, 0.0, 0.0],
                   [0.0, 3.0, 0.0],
                   [0.0, 0.0, 4.0]],
                  [[1.0, 0.0, 0.0],
                   [0.0, 4.0, 0.0],
                   [0.0, 0.0, 6.0]]], dtype=np.float32)
    generated_inputs.append({"L": L, "upper": False})
    
    # Input 4: Different size matrix, float32, upper=True, positive definite
    L = np.array([[4.0, 0.0],
                  [0.0, 2.0]], dtype=np.float32)
    generated_inputs.append({"L": L, "upper": True})
    # Input 5: Larger matrix, lower=False, positive definite
    L = np.array([[5.0, 0.0, 0.0, 0.0],
                  [0.0, 6.0, 0.0, 0.0],
                  [0.0, 0.0, 7.0, 0.0],
                  [0.0, 0.0, 0.0, 8.0]], dtype=np.float32)
    generated_inputs.append({"L": L, "upper": False})
    
    return generated_inputs
def chunk_inputs():
    list_of_inputs = []
    input1 = torch.randn(4, 4).numpy()
    chunks1 = 2
    dim1 = 0
    input_dict1 = {
        "input": input1,
        "chunks": chunks1,
        "dim": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(3, 5, 7).numpy()
    chunks2 = 3
    dim2 = 1
    input_dict2 = {
        "input": input2,
        "chunks": chunks2,
        "dim": dim2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randint(0, 10, (2, 6, 4)).numpy()
    chunks3 = 4
    dim3 = 2
    input_dict3 = {
        "input": input3,
        "chunks": chunks3,
        "dim": dim3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(2, 2, 2, 2).numpy()
    chunks4 = 2
    dim4 = 3
    input_dict4 = {
        "input": input4,
        "chunks": chunks4,
        "dim": dim4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(5,).numpy()
    chunks5 = 5
    dim5 = 0
    input_dict5 = {
        "input": input5,
        "chunks": chunks5,
        "dim": dim5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(2, 3, 4, 5).numpy()
    chunks6 = 1
    dim6 = 0
    input_dict6 = {
        "input": input6,
        "chunks": chunks6,
        "dim": dim6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = torch.randn(6, 8).numpy()
    chunks7 = 4
    dim7 = 1
    input_dict7 = {
        "input": input7,
        "chunks": chunks7,
        "dim": dim7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    input8 = torch.randn(2, 4, 6, 8).numpy()
    chunks8 = 2
    dim8 = 2
    input_dict8 = {
        "input": input8,
        "chunks": chunks8,
        "dim": dim8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs
def complex_inputs():
    list_of_inputs = []
    real1 = np.array([1, 2, 3], dtype=np.float32)
    imag1 = np.array([4, 5, 6], dtype=np.float32)
    input_dict1 = {"real": real1, "imag": imag1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    real2 = np.array([[1, 2], [3, 4]], dtype=np.float64)
    imag2 = np.array([[5, 6], [7, 8]], dtype=np.float64)
    input_dict2 = {"real": real2, "imag": imag2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    real3 = np.array([[-1.5, 2.5], [3.5, -4.5]], dtype=np.float32)
    imag3 = np.array([[5.5, -6.5], [-7.5, 8.5]], dtype=np.float32)
    input_dict3 = {"real": real3, "imag": imag3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    real4 = np.array([1, 2, 3], dtype=np.float64)
    imag4 = np.array([4, 5, 6], dtype=np.float64)
    input_dict4 = {"real": real4, "imag": imag4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    real5 = np.array([1], dtype=np.float32)
    imag5 = np.array([0], dtype=np.float32)
    input_dict5 = {"real": real5, "imag": imag5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    real6 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    imag6 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.float64)
    input_dict6 = {"real": real6, "imag": imag6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    real7 = np.array([], dtype=np.float32)
    imag7 = np.array([], dtype=np.float32)
    input_dict7 = {"real": real7, "imag": imag7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def copysign_inputs():
    list_of_inputs = []
    # Test case 1: Basic float tensors, different shapes
    input1 = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    other1 = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Test case 2: Integer tensors
    input2 = np.array([-1, 2, -3, 4], dtype=np.int32)
    other2 = np.array([1, -2, 3, -4], dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Test case 3: Multi-dimensional tensors
    input3 = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float64)
    other3 = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Test case 4: Mixed positive and negative zeros
    input4 = np.array([-0.0, 0.0, -1.0, 1.0], dtype=np.float32)
    other4 = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Test case 5: Different dtypes for input and other (float64 and float32)
    input5 = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float64)
    other5 = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Test case 6: scalar inputs
    input6 = np.array(-5.0)
    other6 = np.array(2.0)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Test case 7: 3D tensors
    input7 = np.random.randn(2, 3, 4).astype(np.float32)
    other7 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def cosine_similarity_inputs():
    list_of_inputs = []
    x1 = np.random.randn(3, 5).astype(np.float32)
    x2 = np.random.randn(3, 5).astype(np.float32)
    dim = 1
    eps = 1e-8
    input_dict = {"x1": x1, "x2": x2, "dim": dim, "eps": eps}
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = np.random.randn(10).astype(np.float64)
    x2 = np.random.randn(10).astype(np.float64)
    dim = 0
    eps = 1e-6
    input_dict = {"x1": x1, "x2": x2, "dim": dim, "eps": eps}
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(2, 3, 4).astype(np.float32)
    dim = 2
    eps = 1e-12
    input_dict = {"x1": x1, "x2": x2, "dim": dim, "eps": eps}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.random.randn(4, 4).astype(np.float32)
    x2 = np.random.randn(4, 4).astype(np.float32)
    dim = 0
    eps = 1e-8
    input_dict = {"x1": x1, "x2": x2, "dim": dim, "eps": eps}
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = np.random.randn(5, 2).astype(np.float32)
    x2 = np.random.randn(5, 2).astype(np.float32)
    dim = 1
    eps = 1e-8
    input_dict = {"x1": x1, "x2": x2, "dim": dim, "eps": eps}
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
    x2 = np.array([[7,8,9],[10,11,12]], dtype=np.float32)
    dim = 1
    eps = 1e-8
    input_dict = {"x1": x1, "x2": x2, "dim": dim, "eps": eps}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def count_nonzero_inputs():
    list_of_inputs = []
    # Input 1: Basic 1D integer tensor with some zeros
    input1 = np.array([0, 1, 2, 0, 3, 0], dtype=np.int64)
    input_dict1 = {"input": input1, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D float tensor with negative values and zeros
    input2 = np.array([[-1.0, 0.0, 2.5], [0.0, -3.2, 0.0]], dtype=np.float32)
    input_dict2 = {"input": input2, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 3D boolean tensor
    input3 = np.array([[[True, False, True], [False, True, False]],
                       [[True, True, False], [False, False, True]]], dtype=np.bool_)
    input_dict3 = {"input": input3, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Tensor with complex numbers
    input4 = np.array([1 + 1j, 0 + 0j, 2 - 1j, 0 + 2j], dtype=np.complex64)
    input_dict4 = {"input": input4, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Empty tensor
    input5 = np.array([], dtype=np.int64)
    input_dict5 = {"input": input5, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs
def cross_entropy_inputs():
    list_of_inputs = []
    # Case 1: Basic case with 2D input and 1D target (long)
    input_tensor = torch.randn(3, 5).numpy()
    target_tensor = torch.randint(0, 5, (3,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Different batch size and number of classes
    input_tensor = torch.randn(5, 10).numpy()
    target_tensor = torch.randint(0, 10, (5,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Input with log probabilities (softmax already applied)
    input_tensor = torch.randn(4, 3).log_softmax(dim=1).numpy()
    target_tensor = torch.randint(0, 3, (4,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: Larger tensors
    input_tensor = torch.randn(10, 20).numpy()
    target_tensor = torch.randint(0, 20, (10,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Negative values in input (valid after softmax/log_softmax)
    input_tensor = torch.randn(3, 4).numpy()
    target_tensor = torch.randint(0, 4, (3,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def cross_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensors
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other1 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Integer tensors
    input2 = np.array([1, 0, 0], dtype=np.int64)
    other2 = np.array([0, 1, 0], dtype=np.int64)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Negative values
    input3 = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    other3 = np.array([4.0, -5.0, 6.0], dtype=np.float32)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Tensors with zeros
    input7 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    other7 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def dist_inputs():
    list_of_inputs = []
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other1 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    p1 = 2.0
    input_dict1 = {"input": input1, "other": other1, "p": p1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    other2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    p2 = 1.0
    input_dict2 = {"input": input2, "other": other2, "p": p2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    other3 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    p3 = 2.5
    input_dict3 = {"input": input3, "other": other3, "p": p3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input5 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    other5 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)
    p5 = 0.5
    input_dict5 = {"input": input5, "other": other5, "p": p5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other6 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    p6 = float('inf')
    input_dict6 = {"input": input6, "other": other6, "p": p6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(2,2)
    other7 = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32).reshape(2,2)
    p7 = 3.0
    input_dict7 = {"input": input7, "other": other7, "p": p7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def dot_inputs():
    list_of_inputs = []
    # Case 1: Basic float tensors
    a = torch.randn(3).numpy()
    b = torch.randn(3).numpy()
    input_dict = {"input": a, "tensor": b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Integer tensors
    a = torch.randint(0, 10, (4,)).numpy()
    b = torch.randint(0, 10, (4,)).numpy()
    input_dict = {"input": a, "tensor": b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Negative values
    a = torch.randint(-10, 0, (5,)).numpy()
    b = torch.randint(-5, 5, (5,)).numpy()
    input_dict = {"input": a, "tensor": b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: Large tensors
    a = torch.randn(1000).numpy()
    b = torch.randn(1000).numpy()
    input_dict = {"input": a, "tensor": b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Small tensors
    a = torch.randn(1).numpy()
    b = torch.randn(1).numpy()
    input_dict = {"input": a, "tensor": b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 7: Float64
    a = torch.randn(3, dtype=torch.float64).numpy()
    b = torch.randn(3, dtype=torch.float64).numpy()
    input_dict = {"input": a, "tensor": b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def einsum_inputs():
    list_of_inputs = []
    # Case 1: Matrix multiplication
    a = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(3, 4).astype(np.float32)
    input_dict = {
        'equation': 'ij,jk->ik',
        'operands': [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Batch matrix multiplication
    a = np.random.rand(5, 2, 3).astype(np.float32)
    b = np.random.rand(5, 3, 4).astype(np.float32)
    input_dict = {
        'equation': 'bij,bjk->bik',
        'operands': [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Trace of a matrix
    a = np.random.rand(4, 4).astype(np.float32)
    input_dict = {
        'equation': 'ii->',
        'operands': [a]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: Sum along an axis
    a = np.random.rand(3, 5).astype(np.float32)
    input_dict = {
        'equation': 'ij->i',
        'operands': [a]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Transpose
    a = np.random.rand(3, 5).astype(np.float32)
    input_dict = {
        'equation': 'ij->ji',
        'operands': [a]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 6: Dot product
    a = np.random.rand(5).astype(np.float32)
    b = np.random.rand(5).astype(np.float32)
    input_dict = {
        'equation': 'i,i->',
        'operands': [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 7: Hadamard product and sum
    a = np.random.rand(3, 3).astype(np.float32)
    b = np.random.rand(3, 3).astype(np.float32)
    input_dict = {
        'equation': 'ij,ij->',
        'operands': [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 8: More complex with 3 tensors
    a = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(3, 4).astype(np.float32)
    c = np.random.rand(4, 2).astype(np.float32)
    input_dict = {
        'equation': 'ij,jk,kl->il',
        'operands': [a, b, c]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 9: Broadcasting Example
    a = np.random.rand(3, 1).astype(np.float32)
    b = np.random.rand(1, 4).astype(np.float32)
    input_dict = {
        'equation': 'ij,jk->ik',
        'operands': [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 10: Integer tensors
    a = np.random.randint(0, 10, size=(2, 3)).astype(np.int32)
    b = np.random.randint(0, 10, size=(3, 4)).astype(np.int32)
    input_dict = {
        'equation': 'ij,jk->ik',
        'operands': [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def embedding_bag_inputs():
    list_of_inputs = []
    # Input 1: Basic example with sum mode
    input_dict = {
        "input": np.array([1, 2, 4, 5, 4, 3, 0], dtype=np.int64),
        "weight": np.random.rand(7, 3).astype(np.float32),
        "offsets": np.array([0, 1, 2, 4, 5, 7], dtype=np.int64),
        "max_norm": None,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
        "padding_idx": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Example with mean mode
    input_dict = {
        "input": np.array([0, 1, 2, 3], dtype=np.int64),
        "weight": np.random.rand(4, 5).astype(np.float32),
        "offsets": np.array([0, 2, 4], dtype=np.int64),
        "max_norm": 1.0,
        "norm_type": 2,
        "scale_grad_by_freq": True,
        "mode": "mean",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
        "padding_idx": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Example with max mode and padding_idx
    input_dict = {
        "input": np.array([0, 1, 2, 0, 3], dtype=np.int64),
        "weight": np.random.rand(4, 4).astype(np.float32),
        "offsets": np.array([0, 3, 5], dtype=np.int64),
        "max_norm": None,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "max",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
        "padding_idx": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Example with include_last_offset=True
    input_dict = {
        "input": np.array([0, 1, 2, 3], dtype=np.int64),
        "weight": np.random.rand(4, 2).astype(np.float32),
        "offsets": np.array([0, 2, 4], dtype=np.int64),
        "max_norm": None,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": True,
        "padding_idx": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Example with smaller embedding dimension
    input_dict = {
        "input": np.array([1, 0, 2, 1], dtype=np.int64),
        "weight": np.random.rand(3, 1).astype(np.float32),
        "offsets": np.array([0, 2, 4], dtype=np.int64),
        "max_norm": None,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
        "padding_idx": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Example with per_sample_weights
    input_dict = {
        "input": np.array([0, 1, 2, 3], dtype=np.int64),
        "weight": np.random.rand(4, 5).astype(np.float32),
        "offsets": np.array([0, 2, 4], dtype=np.int64),
        "max_norm": 1.0,
        "norm_type": 2,
        "scale_grad_by_freq": True,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        "include_last_offset": False,
        "padding_idx": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Example with per_sample_weights and scale_grad_by_freq=False
    input_dict = {
        "input": np.array([0, 1, 2, 3], dtype=np.int64),
        "weight": np.random.rand(4, 5).astype(np.float32),
        "offsets": np.array([0, 2, 4], dtype=np.int64),
        "max_norm": 1.0,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        "include_last_offset": False,
        "padding_idx": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def empty_strided_inputs():
    list_of_inputs = []
    input_dict = {
        "size": (2, 3),
        "stride": (3, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "size": (4, 5, 6),
        "stride": (30, 6, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "size": (1, 1, 1, 1),
        "stride": (1, 1, 1, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "size": (7,),
        "stride": (1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "size": (2, 2, 2, 2, 2),
        "stride": (16, 8, 4, 2, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "size": (10,),
        "stride": (1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def eq_inputs():
    list_of_inputs = []
    # Case 1: Basic float tensors
    input1 = np.array([1.0, 2.0, 3.0])
    input2 = np.array([1.0, 2.0, 4.0])
    list_of_inputs.append({"input": input1, "other": input2})
    # Case 2: Integer tensors
    input1 = np.array([1, 2, 3], dtype=np.int32)
    input2 = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"input": input1, "other": input2})
    # Case 3: Different shapes (broadcastable)
    input1 = np.array([[1, 2], [3, 4]])
    input2 = np.array([1, 2])
    list_of_inputs.append({"input": input1, "other": input2})
    # Case 4: Negative values and different dtypes
    input1 = np.array([-1.0, 0.0, 1.0])
    input2 = np.array([-1, 0, 1], dtype=np.int64)
    list_of_inputs.append({"input": input1, "other": input2})
    
    # Case 5: Multi-dimensional array
    input1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input2 = np.array([[[1, 2], [3, 4]], [[5, 7], [7, 8]]])
    list_of_inputs.append({"input": input1, "other": input2})
    
    # Case 6: Zero-dimensional array
    input1 = np.array(5)
    input2 = np.array(5)
    list_of_inputs.append({"input": input1, "other": input2})
    # Case 7: Complex numbers
    input1 = np.array([1 + 1j, 2 + 2j])
    input2 = np.array([1 + 1j, 3 + 3j])
    list_of_inputs.append({"input": input1, "other": input2})
    # Case 8: Boolean arrays
    input1 = np.array([True, False, True])
    input2 = np.array([True, True, False])
    list_of_inputs.append({"input": input1, "other": input2})
    return list_of_inputs
def flatten_inputs():
    list_of_inputs = []
    # Case 1: Basic 2D float tensor, default start and end dim
    input_tensor = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"input": input_tensor, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: 3D int tensor, specified start and end dim
    input_tensor = np.random.randint(0, 10, size=(2, 3, 4)).astype(np.int32)
    input_dict = {"input": input_tensor, "start_dim": 1, "end_dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: 4D complex tensor, negative start and end dim
    input_tensor = (np.random.randn(2, 3, 2, 2) + 1j * np.random.randn(2, 3, 2, 2)).astype(np.complex64)
    input_dict = {"input": input_tensor, "start_dim": -2, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: 1D tensor, start and end dim are the same
    input_tensor = np.array([1, 2, 3, 4, 5]).astype(np.int64)
    input_dict = {"input": input_tensor, "start_dim": 0, "end_dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: 5D tensor, covering most dimensions
    input_tensor = np.random.randn(1, 2, 3, 4, 5).astype(np.float64)
    input_dict = {"input": input_tensor, "start_dim": 2, "end_dim": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 6: 3D tensor, negative indexing
    input_tensor = np.random.randn(2, 4, 6).astype(np.float32)
    input_dict = {"input": input_tensor, "start_dim": 0, "end_dim": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: bool tensor
    input_tensor = np.array([[True, False], [False, True]]).astype(bool)
    input_dict = {"input": input_tensor, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def flatten_inputs_2():
    list_of_inputs = []
    # Input 1: 2D float tensor, default start_dim and end_dim
    input1 = np.random.randn(2, 3).astype(np.float32)
    input_dict1 = {"input": input1, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 3D int tensor, specified start_dim and end_dim
    input2 = np.random.randint(0, 10, size=(1, 4, 5)).astype(np.int32)
    input_dict2 = {"input": input2, "start_dim": 1, "end_dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 4D float tensor, specified start_dim and end_dim
    input3 = np.random.randn(2, 3, 4, 5).astype(np.float64)
    input_dict3 = {"input": input3, "start_dim": 1, "end_dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 1D float tensor
    input4 = np.random.randn(10).astype(np.float32)
    input_dict4 = {"input": input4, "start_dim": 0, "end_dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: 5D int tensor, flatten all dimensions
    input5 = np.random.randint(0, 5, size=(1, 2, 3, 4, 5)).astype(np.int64)
    input_dict5 = {"input": input5, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def flip_inputs():
    list_of_inputs = []
    # Input 1: 1D float tensor
    input1 = torch.randn(5).numpy()
    dims1 = (0,)
    input_dict1 = {"input": input1, "dims": dims1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D int tensor
    input2 = torch.randint(0, 10, (3, 4)).numpy()
    dims2 = (0, 1)
    input_dict2 = {"input": input2, "dims": dims2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 3D complex tensor
    input3 = torch.randn(2, 3, 4, dtype=torch.complex64).numpy()
    dims3 = (1, 2)
    input_dict3 = {"input": input3, "dims": dims3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 4D tensor with negative dimension
    input4 = torch.randn(1, 2, 3, 4).numpy()
    dims4 = (-1,)
    input_dict4 = {"input": input4, "dims": dims4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 5D tensor
    input5 = torch.randn(2, 2, 2, 2, 2).numpy()
    dims5 = (0, 2, 4)
    input_dict5 = {"input": input5, "dims": dims5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Empty tensor
    input6 = torch.empty(0).numpy()
    dims6 = (0,)
    input_dict6 = {"input": input6, "dims": dims6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: Boolean tensor
    input7 = torch.randint(0, 2, (2, 3), dtype=torch.bool).numpy()
    dims7 = (0, 1)
    input_dict7 = {"input": input7, "dims": dims7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def float_power_inputs():
    list_of_inputs = []
    # Case 1: Basic float input and exponent
    input_tensor = torch.randn(2, 3).numpy()
    exponent_tensor = torch.randn(2, 3).numpy()
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Integer input and float exponent
    input_tensor = torch.randint(1, 5, (3, 4)).numpy()
    exponent_tensor = torch.randn(3, 4).numpy()
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Float input and integer exponent
    input_tensor = torch.randn(4, 2).numpy()
    exponent_tensor = torch.randint(1, 4, (4, 2)).numpy()
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: Scalar input and exponent
    input_tensor = torch.randn(1).item()
    exponent_tensor = torch.randn(1).item()
    input_dict = {"input": np.array(input_tensor), "exponent":  np.array(exponent_tensor)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Negative input and float exponent (check behavior)
    input_tensor = torch.randn(2, 2) * -1.0
    input_tensor = input_tensor.numpy()
    exponent_tensor = torch.randn(2, 2).numpy()
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 6: Different shaped inputs (broadcastable)
    input_tensor = torch.randn(2, 3).numpy()
    exponent_tensor = torch.randn(1, 3).numpy()
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 7: Multi-dimensional input
    input_tensor = torch.randn(2, 3, 4).numpy()
    exponent_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def floor_divide_inputs():
    list_of_inputs = []
    # Example 1: Basic integer division
    input1 = np.array([10, 20, 30]).astype(np.int32)
    other1 = np.array([3, 7, 2]).astype(np.int32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Example 2: Floating point division
    input2 = np.array([10.5, 20.3, 30.9]).astype(np.float32)
    other2 = np.array([3.0, 7.0, 2.0]).astype(np.float32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Example 3: Negative numbers
    input3 = np.array([-10, -20, 30]).astype(np.int64)
    other3 = np.array([3, -7, 2]).astype(np.int64)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Example 4: Multi-dimensional arrays
    input4 = np.array([[10, 20], [30, 40]]).astype(np.int32)
    other4 = np.array([[3, 7], [2, 5]]).astype(np.int32)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Example 5: Different shapes (broadcasting)
    input5 = np.array([[10, 20, 30], [40, 50, 60]]).astype(np.float64)
    other5 = np.array([2, 5, 10]).astype(np.float64)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Example 6: Scalar division
    input6 = np.array([10, 20, 30]).astype(np.int32)
    other6 = np.array(5).astype(np.int32)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Example 7: Zero division (shouldn't error, will produce inf)
    input7 = np.array([10, 20, 30]).astype(np.float32)
    other7 = np.array([0, 5, 0]).astype(np.float32)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def fmin_inputs():
    list_of_inputs = []
    # Case 1: Basic float tensors
    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Case 2: Integer tensors
    input2 = torch.randint(0, 10, (2, 5)).numpy()
    other2 = torch.randint(0, 10, (2, 5)).numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Case 3: Different shapes (but still compatible)
    input3 = torch.randn(2, 3, 4).numpy()
    other3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Case 4: One dimensional tensors
    input4 = torch.randn(10).numpy()
    other4 = torch.randn(10).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Case 5: Negative values
    input5 = torch.randn(5, 5).numpy() - 2
    other5 = torch.randn(5, 5).numpy() - 1
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Case 6: Scalar input
    input6 = torch.randn(1).numpy()
    other6 = torch.randn(1).numpy()
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Case 7: Broadcasting
    input7 = torch.randn(5, 1).numpy()
    other7 = torch.randn(5, 5).numpy()
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    # Case 8: Different dtypes (float64)
    input8 = torch.randn(3, 4, dtype=torch.float64).numpy()
    other8 = torch.randn(3, 4, dtype=torch.float64).numpy()
    input_dict8 = {"input": input8, "other": other8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    return list_of_inputs
def fractional_max_pool2d_inputs():
    list_of_inputs = []
    # Input 1: Basic case with integer output_size
    input1 = torch.randn(1, 1, 32, 32).numpy()
    input_dict1 = {
        "input": input1,
        "kernel_size": (3, 3),
        "output_size": (16, 16),
        "output_ratio": None,
        "return_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Basic case with float output_ratio
    input2 = torch.randn(1, 3, 64, 64).numpy()
    input_dict2 = {
        "input": input2,
        "kernel_size": (2, 2),
        "output_size": None,
        "output_ratio": (0.5, 0.5),
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Different kernel size and input size
    input3 = torch.randn(2, 1, 10, 10).numpy()
    input_dict3 = {
        "input": input3,
        "kernel_size": (2, 2),
        "output_size": (3,3),
        "output_ratio": None,
        "return_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4:  Non-square input
    input4 = torch.randn(1, 1, 20, 30).numpy()
    input_dict4 = {
        "input": input4,
        "kernel_size": (2, 2),
        "output_size": (5, 7),
        "output_ratio": None,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Non-square kernel
    input5 = torch.randn(1, 3, 40, 40).numpy()
    input_dict5 = {
        "input": input5,
        "kernel_size": (2, 3),
        "output_size": None,
        "output_ratio": (0.6, 0.6),
        "return_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def full_inputs():
    list_of_inputs = []
    # Input 1: Basic example with integer size and float fill value
    input_dict = {
        "size": (2, 3),
        "fill_value": 3.14,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Larger size, integer fill value, and int dtype
    input_dict = {
        "size": (5, 5, 5),
        "fill_value": 7,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Negative fill value, size as a single integer, and float64 dtype
    input_dict = {
        "size": (4,),
        "fill_value": -2.5,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Empty tensor size, should produce an empty tensor
    input_dict = {
        "size": (0,),
        "fill_value": 10,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Size with different dimensions, complex fill value and dtype
    input_dict = {
        "size": (2, 1, 4),
        "fill_value": complex(1.0, -1.0),
        "dtype": np.complex64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Using a boolean dtype
    input_dict = {
        "size": (3, 2),
        "fill_value": 1,
        "dtype": np.bool_
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 7: A large size
    input_dict = {
        "size": (100, 100),
        "fill_value": 0.0,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: unsigned int
    input_dict = {
        "size": (2, 2),
        "fill_value": 255,
        "dtype": np.uint8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def ge_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other1 = np.array([[2.0, 1.0], [4.0, 3.0]])
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Integer tensors with negative values
    input2 = np.array([[-1, 0], [1, 2]], dtype=np.int32)
    other2 = np.array([[0, -1], [2, 1]], dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Different shapes (input is a scalar)
    input3 = np.array(5.0)
    other3 = np.array([[4.0, 6.0], [5.0, 3.0]])
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Different shapes (other is a scalar)
    input4 = np.array([[4.0, 6.0], [5.0, 3.0]])
    other4 = np.array(5.0)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 3D tensors
    input5 = np.random.rand(2, 3, 4)
    other5 = np.random.rand(2, 3, 4)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Boolean tensors
    input6 = np.array([[True, False], [False, True]])
    other6 = np.array([[False, True], [True, False]])
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Zero dimension tensors (scalar)
    input7 = np.array(3)
    other7 = np.array(2)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def ger_inputs():
    generated_inputs = []
    # Test case 1: Basic float tensors
    vec1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    vec2 = np.array([4.0, 5.0], dtype=np.float32)
    generated_inputs.append({"vec1": vec1, "vec2": vec2})
    # Test case 2: Integer tensors
    vec1 = np.array([1, 2, 3], dtype=np.int32)
    vec2 = np.array([4, 5], dtype=np.int32)
    generated_inputs.append({"vec1": vec1, "vec2": vec2})
    # Test case 3: Negative values
    vec1 = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    vec2 = np.array([4.0, -5.0], dtype=np.float32)
    generated_inputs.append({"vec1": vec1, "vec2": vec2})
    # Test case 4: Different sizes
    vec1 = np.array([1.0, 2.0], dtype=np.float32)
    vec2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    generated_inputs.append({"vec1": vec1, "vec2": vec2})
    
    # Test case 5: Single element tensors
    vec1 = np.array([1.0], dtype=np.float32)
    vec2 = np.array([4.0], dtype=np.float32)
    generated_inputs.append({"vec1": vec1, "vec2": vec2})
    return generated_inputs
def check_valid(api_name, list_of_inputs):
    for input_dict in list_of_inputs:
        api_driver(input_dict, cpu=True)
def api_driver(input, cpu=True):
    vec1 = torch.tensor(input["vec1"])
    vec2 = torch.tensor(input["vec2"])
    
    if cpu:
        vec1 = vec1.cpu()
        vec2 = vec2.cpu()
    else:
        vec1 = vec1.cuda()
        vec2 = vec2.cuda()
    torch.ger(vec1, vec2)
def groupnorm_inputs():
    list_of_inputs = []
    input1 = torch.randn(2, 6, 5, 5).numpy()
    num_groups1 = 3
    num_channels1 = 6
    eps1 = 1e-5
    affine1 = True
    input_dict1 = {
        "input": input1,
        "num_groups": num_groups1,
        "num_channels": num_channels1,
        "eps": eps1,
        "affine": affine1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(1, 8, 4, 4).numpy()
    num_groups2 = 4
    num_channels2 = 8
    eps2 = 1e-8
    affine2 = False
    input_dict2 = {
        "input": input2,
        "num_groups": num_groups2,
        "num_channels": num_channels2,
        "eps": eps2,
        "affine": affine2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(1, 12, 3, 3).numpy()
    num_groups3 = 1
    num_channels3 = 12
    eps3 = 1e-3
    affine3 = True
    input_dict3 = {
        "input": input3,
        "num_groups": num_groups3,
        "num_channels": num_channels3,
        "eps": eps3,
        "affine": affine3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(4, 16, 2, 2).numpy()
    num_groups4 = 8
    num_channels4 = 16
    eps4 = 1e-6
    affine4 = False
    input_dict4 = {
        "input": input4,
        "num_groups": num_groups4,
        "num_channels": num_channels4,
        "eps": eps4,
        "affine": affine4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(1, 4, 1, 1, 1).numpy()
    num_groups5 = 2
    num_channels5 = 4
    eps5 = 1e-4
    affine5 = True
    input_dict5 = {
        "input": input5,
        "num_groups": num_groups5,
        "num_channels": num_channels5,
        "eps": eps5,
        "affine": affine5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = torch.randn(2, 32, 10).numpy()
    num_groups6 = 16
    num_channels6 = 32
    eps6 = 1e-2
    affine6 = False
    input_dict6 = {
        "input": input6,
        "num_groups": num_groups6,
        "num_channels": num_channels6,
        "eps": eps6,
        "affine": affine6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = torch.randn(1, 2, 3).numpy()
    num_groups7 = 1
    num_channels7 = 2
    eps7 = 1e-7
    affine7 = True
    input_dict7 = {
        "input": input7,
        "num_groups": num_groups7,
        "num_channels": num_channels7,
        "eps": eps7,
        "affine": affine7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def gt_inputs():
    list_of_inputs = []
    input1 = np.array([1, 2, 3, 4, 5])
    other1 = np.array([2, 2, 2, 2, 2])
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other2 = np.array([[0.5, 2.5], [3.5, 3.5]])
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[-1, -2], [3, 4]])
    other3 = np.array([0])
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([1, 2, 3])
    other4 = 2
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    other5 = np.array([[[0, 3], [2, 5]], [[4, 7], [6, 9]]])
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([1, 2, 3], dtype=np.int64)
    other6 = np.array([0, 1, 4], dtype=np.int64)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    other7 = np.array([0.5, 1.5, 4.5], dtype=np.float64)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def heaviside_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensor, value=0.0
    input1 = np.array([-1.0, 0.0, 1.0, 2.5]).astype(np.float32)
    values1 = np.array([0.0]).astype(np.float32)
    input_dict1 = {"input": input1, "values": values1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Integer tensor, value=1
    input2 = np.array([-2, -1, 0, 1, 2]).astype(np.int32)
    values2 = np.array([1]).astype(np.int32)
    input_dict2 = {"input": input2, "values": values2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Multi-dimensional float tensor, value=0.5
    input3 = np.random.randn(2, 3, 4).astype(np.float64)
    values3 = np.array([0.5]).astype(np.float64)
    input_dict3 = {"input": input3, "values": values3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Boolean tensor, value=True (1)
    input4 = np.array([True, False, True, False]).astype(np.bool_)
    values4 = np.array([True]).astype(np.bool_)
    input_dict4 = {"input": input4, "values": values4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Negative value array
    input5 = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]).astype(np.float32)
    values5 = np.array([0.0]).astype(np.float32)
    input_dict5 = {"input": input5, "values": values5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Scalar input and value
    input6 = np.array(0.0).astype(np.float32)
    values6 = np.array(1.0).astype(np.float32)
    input_dict6 = {"input": input6, "values": values6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Float64 array with a different value
    input7 = np.array([-1.5, -0.5, 0.0, 0.5, 1.5]).astype(np.float64)
    values7 = np.array([1.0]).astype(np.float64)
    input_dict7 = {"input": input7, "values": values7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def hypot_inputs():
    list_of_inputs = []
    # Case 1: Basic float tensors
    input1 = np.array([3.0, 4.0], dtype=np.float32)
    input2 = np.array([5.0, 12.0], dtype=np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Negative values
    input1 = np.array([-3.0, 4.0], dtype=np.float32)
    input2 = np.array([5.0, -12.0], dtype=np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Multi-dimensional tensors
    input1 = np.array([[3.0, 4.0], [1.0, 2.0]], dtype=np.float32)
    input2 = np.array([[5.0, 12.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: Different shapes (but broadcastable)
    input1 = np.array([3.0, 4.0], dtype=np.float32)
    input2 = np.array([5.0], dtype=np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Zero values
    input1 = np.array([0.0, 4.0], dtype=np.float32)
    input2 = np.array([5.0, 0.0], dtype=np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: Larger dimensions
    input1 = np.random.rand(2, 3, 4).astype(np.float32)
    input2 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def igamma_inputs():
    list_of_inputs = []
    # Test case 1: Basic float tensors
    input1 = np.random.rand(3, 4).astype(np.float32)
    other1 = np.random.rand(3, 4).astype(np.float32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Test case 2: Integer tensors (should be cast to float)
    input2 = np.random.randint(1, 10, size=(2, 2)).astype(np.float32)
    other2 = np.random.randint(1, 10, size=(2, 2)).astype(np.float32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Test case 3: Scalar inputs
    input3 = np.array(2.5).astype(np.float64)
    other3 = np.array(1.5).astype(np.float64)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Test case 4: Different shapes (but compatible)
    input4 = np.random.rand(5, 1).astype(np.float32)
    other4 = np.random.rand(1, 5).astype(np.float32)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Test case 5: Larger tensors with broadcasting
    input5 = np.random.rand(2, 3, 4).astype(np.float32)
    other5 = np.random.rand(3, 4).astype(np.float32)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Test case 6: 0-dimensional array
    input6 = np.array(np.random.rand()).astype(np.float32)
    other6 = np.array(np.random.rand()).astype(np.float32)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def index_select_inputs():
    list_of_inputs = []
    # Input 1: 2D float tensor, positive indices
    input1 = torch.randn(3, 4).numpy()
    index1 = torch.tensor([0, 2]).long().numpy()
    dim1 = 0
    input_dict1 = {"input": input1, "dim": dim1, "index": index1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 3D int tensor, positive indices
    input2 = torch.randint(0, 10, (2, 3, 5)).int().numpy()
    index2 = torch.tensor([1, 0]).long().numpy()
    dim2 = 1
    input_dict2 = {"input": input2, "dim": dim2, "index": index2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 2D complex tensor, positive indices
    input3 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    index3 = torch.tensor([0]).long().numpy()
    dim3 = 0
    input_dict3 = {"input": input3, "dim": dim3, "index": index3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    return list_of_inputs
def inner_inputs():
    list_of_inputs = []
    # Test case 1: Basic float tensors
    input1 = np.array([1.0, 2.0, 3.0])
    other1 = np.array([4.0, 5.0, 6.0])
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Test case 2: Integer tensors
    input2 = np.array([1, 2, 3], dtype=np.int32)
    other2 = np.array([4, 5, 6], dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Test case 3: Multi-dimensional tensors
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other3 = np.array([[5.0, 6.0], [7.0, 8.0]])
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Test case 4: Negative values
    input4 = np.array([-1.0, -2.0, -3.0])
    other4 = np.array([4.0, 5.0, 6.0])
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Test case 5: Different shapes (but compatible for inner product)
    input5 = np.array([[1.0, 2.0, 3.0]])
    other5 = np.array([4.0, 5.0, 6.0])
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Test case 6: Complex tensors
    input6 = np.array([1+1j, 2+2j, 3+3j])
    other6 = np.array([4+4j, 5+5j, 6+6j])
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Test case 7: 3D tensors, compatible shapes
    input7 = np.random.rand(2, 3, 4)
    other7 = np.random.rand(2, 3, 4)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def interpolate_inputs():
    list_of_inputs = []
    input1 = torch.randn(1, 3, 10, 10).numpy()
    input_dict1 = {"input": input1, "size": (12, 12), "mode": "nearest"}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(2, 4, 5, 5, 5).numpy()
    input_dict2 = {"input": input2, "scale_factor": 2, "mode": "trilinear"}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(1, 1, 15).numpy()
    input_dict3 = {"input": input3, "size": (20,), "mode": "linear"}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(1, 2, 8, 8).numpy()
    input_dict4 = {"input": input4, "scale_factor": 0.5, "mode": "bilinear"}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(1, 3, 7, 7).numpy()
    input_dict5 = {"input": input5, "size": (9, 9), "mode": "bicubic"}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def isclose_inputs():
    list_of_inputs = []
    input1 = np.array([1.0, 2.0, 3.0])
    other1 = np.array([1.0, 2.0, 3.1])
    input_dict1 = {
        "input": input1,
        "other": other1,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([1.0, np.nan, 3.0])
    other2 = np.array([1.0, np.nan, 3.0])
    input_dict2 = {
        "input": input2,
        "other": other2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([1, 2, 3], dtype=np.int32)
    other3 = np.array([1, 2, 3], dtype=np.int32)
    input_dict3 = {
        "input": input3,
        "other": other3,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other4 = np.array([[1.0, 2.1], [3.0, 4.0]])
    input_dict4 = {
        "input": input4,
        "other": other4,
        "rtol": 0.1,
        "atol": 0.0,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([-1.0, -2.0, -3.0])
    other5 = np.array([-1.0, -2.0, -3.1])
    input_dict5 = {
        "input": input5,
        "other": other5,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([1.0 + 1j, 2.0 + 2j])
    other6 = np.array([1.0 + 1j, 2.0 + 2.1j])
    input_dict6 = {
        "input": input6,
        "other": other6,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    other7 = np.array([[[1.0, 2.0], [3.1, 4.0]], [[5.0, 6.0], [7.0, 8.1]]])
    input_dict7 = {
        "input": input7,
        "other": other7,
        "rtol": 0.05,
        "atol": 0.0,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def is_nonzero_inputs():
    list_of_inputs = []
    input1 = np.array(1, dtype=np.int32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array(-2.5, dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array(1, dtype=np.int8)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array(0, dtype=np.int64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array(1, dtype=np.uint8)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array(-1.5, dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array(0.0, dtype=np.float16)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def l1_loss_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 5).numpy()
    target1 = torch.randn(3, 5).numpy()
    input_dict1 = {"input": input1, "target": target1, "reduction": 'mean'}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(2, 4, 6).numpy()
    target2 = torch.randn(2, 4, 6).numpy()
    input_dict2 = {"input": input2, "target": target2, "reduction": 'sum'}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input4 = torch.randn(10).numpy()
    target4 = torch.randn(10).numpy()
    input_dict4 = {"input": input4, "target": target4, "reduction": 'mean'}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(2, 2, 2, 2).numpy()
    target5 = torch.randn(2, 2, 2, 2).numpy()
    input_dict5 = {"input": input5, "target": target5, "reduction": 'sum'}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def l1loss_inputs():
    list_of_inputs = []
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target1 = np.array([1.0, 2.5, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1, "target": target1, "reduction": 'mean'}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target2 = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    input_dict2 = {"input": input2, "target": target2, "reduction": 'mean'}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    target3 = np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)
    input_dict3 = {"input": input3, "target": target3, "reduction": 'mean'}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([1, 2, 3], dtype=np.float32)
    target4 = np.array([2, 3, 4], dtype=np.float32)
    input_dict4 = {"input": input4, "target": target4, "reduction": 'mean'}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    target5 = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    input_dict5 = {"input": input5, "target": target5, "reduction": 'mean'}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def layer_norm_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensor, normalized shape is the last dimension
    input1 = np.random.randn(2, 3, 4).astype(np.float32)
    normalized_shape1 = [4]
    input_dict1 = {
        "input": input1,
        "normalized_shape": normalized_shape1,
        "eps": 1e-5,
        "elementwise_affine": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Float tensor, normalized shape is multiple dimensions
    input2 = np.random.randn(2, 5, 5, 3).astype(np.float32)
    normalized_shape2 = [5, 3]
    input_dict2 = {
        "input": input2,
        "normalized_shape": normalized_shape2,
        "eps": 1e-8,
        "elementwise_affine": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Negative values, different epsilon
    input3 = np.random.randn(1, 5, 7, 7).astype(np.float64) * -1
    normalized_shape3 = [7, 7]
    input_dict3 = {
        "input": input3,
        "normalized_shape": normalized_shape3,
        "eps": 1e-3,
        "elementwise_affine": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 1D input
    input4 = np.random.randn(10).astype(np.float32)
    normalized_shape4 = [10]
    input_dict4 = {
        "input": input4,
        "normalized_shape": normalized_shape4,
        "eps": 1e-5,
        "elementwise_affine": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Larger input tensor, different normalized shape
    input5 = np.random.randn(4, 6, 8, 10).astype(np.float32)
    normalized_shape5 = [10]
    input_dict5 = {
        "input": input5,
        "normalized_shape": normalized_shape5,
        "eps": 1e-6,
        "elementwise_affine": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs
def layernorm_inputs():
    list_of_inputs = []
    # Input 1: 2D tensor, normalized_shape = [2]
    input1 = np.random.randn(3, 2).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "normalized_shape": [2],
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 3D tensor, normalized_shape = [3, 4]
    input2 = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "normalized_shape": [3, 4],
        "eps": 1e-8,
        "elementwise_affine": False,
        "bias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 4D tensor with negative values, normalized_shape = [4]
    input3 = np.random.randn(1, 2, 3, 4).astype(np.float32) * -1
    input_dict3 = {
        "input": input3,
        "normalized_shape": [4],
        "eps": 1e-6,
        "elementwise_affine": True,
        "bias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 1D tensor, normalized_shape = [5]
    input4 = np.random.randn(5).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "normalized_shape": [5],
        "eps": 1e-5,
        "elementwise_affine": False,
        "bias": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: 5D tensor, normalized_shape = [3, 4, 5]
    input5 = np.random.randn(1, 2, 3, 4, 5).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "normalized_shape": [3, 4, 5],
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6:  Float16 Tensor
    input6 = np.random.randn(2, 4).astype(np.float16)
    input_dict6 = {
        "input": input6,
        "normalized_shape": [4],
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def lcm_inputs():
    list_of_inputs = []
    # Input 1: Basic integer tensors
    input1 = np.array([2, 4, 6], dtype=np.int32)
    other1 = np.array([3, 5, 7], dtype=np.int32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Different shapes, but compatible
    input2 = np.array([[2, 4], [6, 8]], dtype=np.int64)
    other2 = np.array([[3, 5], [7, 9]], dtype=np.int64)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Scalars
    input3 = np.array(5, dtype=np.int32)
    other3 = np.array(7, dtype=np.int32)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Broadcasting
    input4 = np.array([2, 4, 6], dtype=np.int64)
    other4 = np.array(3, dtype=np.int64)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Multi-dimensional arrays
    input5 = np.array([[[2, 4], [6, 8]], [[10, 12], [14, 16]]], dtype=np.int32)
    other5 = np.array([[[3, 5], [7, 9]], [[11, 13], [15, 17]]], dtype=np.int32)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Mixed dtypes (int32 and int64)
    input6 = np.array([2, 4, 6], dtype=np.int32)
    other6 = np.array([3, 5, 7], dtype=np.int64)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: Zero value
    input7 = np.array([0, 4, 6], dtype=np.int32)
    other7 = np.array([3, 0, 7], dtype=np.int32)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: Large Numbers
    input8 = np.array([2**30, 4], dtype=np.int64)
    other8 = np.array([3, 2**31], dtype=np.int64)
    input_dict8 = {"input": input8, "other": other8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs
def le_inputs():
    list_of_inputs = []
    # Case 1: Basic float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other1 = np.array([[2.0, 2.0], [1.0, 5.0]])
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Case 2: Integer tensors
    input2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    other2 = np.array([[2, 2], [1, 5]], dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Case 3: Different shapes (broadcasting)
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other3 = np.array([2.0, 3.0])
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Case 4: Scalar
    input4 = np.array([[1, 2], [3, 4]])
    other4 = 3
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Case 5: Negative values
    input5 = np.array([[-1.0, 2.0], [-3.0, 4.0]])
    other5 = np.array([[0.0, 2.0], [-2.0, 3.0]])
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Case 6: Multi-dimensional arrays
    input6 = np.random.rand(2, 3, 4)
    other6 = np.random.rand(2, 3, 4)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Case 7: Bool arrays
    input7 = np.array([[True, False], [False, True]])
    other7 = np.array([[False, True], [True, False]])
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    # Case 8: Zero-dimensional arrays
    input8 = np.array(5.0)
    other8 = np.array(7.0)
    input_dict8 = {"input": input8, "other": other8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    return list_of_inputs
def linear_inputs():
    list_of_inputs = []
    # Input 1: Basic case with float tensors
    input_1 = torch.randn(3, 4).numpy()
    weight_1 = torch.randn(5, 4).numpy()
    bias_1 = torch.randn(5).numpy()
    input_dict_1 = {"input": input_1, "weight": weight_1, "bias": bias_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    # Input 2: Input with batch dimension
    input_2 = torch.randn(2, 3, 4).numpy()
    weight_2 = torch.randn(5, 4).numpy()
    bias_2 = torch.randn(5).numpy()
    input_dict_2 = {"input": input_2, "weight": weight_2, "bias": bias_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    # Input 3: Integer tensors
    input_3 = torch.randint(0, 10, (3, 4)).numpy()
    weight_3 = torch.randint(0, 10, (5, 4)).numpy()
    bias_3 = torch.randint(0, 10, (5,)).numpy()
    input_dict_3 = {"input": input_3, "weight": weight_3, "bias": bias_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    # Input 4: No bias
    input_4 = torch.randn(3, 4).numpy()
    weight_4 = torch.randn(5, 4).numpy()
    input_dict_4 = {"input": input_4, "weight": weight_4, "bias": None}
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    # Input 5: Single input vector
    input_5 = torch.randn(4).numpy()
    weight_5 = torch.randn(5, 4).numpy()
    bias_5 = torch.randn(5).numpy()
    input_dict_5 = {"input": input_5, "weight": weight_5, "bias": bias_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6: Larger input and weight dimensions
    input_6 = torch.randn(10, 20).numpy()
    weight_6 = torch.randn(30, 20).numpy()
    bias_6 = torch.randn(30).numpy()
    input_dict_6 = {"input": input_6, "weight": weight_6, "bias": bias_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Negative Values
    input_7 = torch.randn(3, 4) * -1.0
    weight_7 = torch.randn(5, 4) * -1.0
    bias_7 = torch.randn(5) * -1.0
    input_dict_7 = {"input": input_7.numpy(), "weight": weight_7.numpy(), "bias": bias_7.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    return list_of_inputs
def linear_inputs():
    list_of_inputs = []
    # Input 1: Basic float input with bias
    input1 = torch.randn(3, 5).numpy()
    in_features1 = 5
    out_features1 = 4
    bias1 = True
    input_dict1 = {
        "input": input1,
        "in_features": in_features1,
        "out_features": out_features1,
        "bias": bias1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Integer input without bias
    input2 = torch.randint(0, 10, (2, 6)).float().numpy() # Convert to float
    in_features2 = 6
    out_features2 = 3
    bias2 = False
    input_dict2 = {
        "input": input2,
        "in_features": in_features2,
        "out_features": out_features2,
        "bias": bias2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    # Input 3: Input with negative values and bias
    input3 = torch.randn(4, 7) * -1.0
    input3 = input3.numpy()
    in_features3 = 7
    out_features3 = 2
    bias3 = True
    input_dict3 = {
        "input": input3,
        "in_features": in_features3,
        "out_features": out_features3,
        "bias": bias3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 1D input
    input4 = torch.randn(10).numpy()
    in_features4 = 10
    out_features4 = 5
    bias4 = True
    input_dict4 = {
        "input": input4,
        "in_features": in_features4,
        "out_features": out_features4,
        "bias": bias4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Higher dimensional input
    input5 = torch.randn(2, 3, 4).numpy()
    in_features5 = 4
    out_features5 = 2
    bias5 = False
    input_dict5 = {
        "input": input5,
        "in_features": in_features5,
        "out_features": out_features5,
        "bias": bias5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Zero input, with bias
    input6 = torch.zeros(2, 3).numpy()
    in_features6 = 3
    out_features6 = 4
    bias6 = True
    input_dict6 = {
        "input": input6,
        "in_features": in_features6,
        "out_features": out_features6,
        "bias": bias6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Large in_features and out_features, with bias
    input7 = torch.randn(1, 100).numpy()
    in_features7 = 100
    out_features7 = 50
    bias7 = True
    input_dict7 = {
        "input": input7,
        "in_features": in_features7,
        "out_features": out_features7,
        "bias": bias7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def linspace_inputs():
    generated_inputs = []
    start = np.array(0.0, dtype=np.float32)
    end = np.array(10.0, dtype=np.float32)
    steps = 5
    input_dict = {
        "start": torch.tensor(start),
        "end": torch.tensor(end),
        "steps": steps,
        "dtype": torch.float32,
        "requires_grad": False
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    start = np.array(-5, dtype=np.int32)
    end = np.array(5, dtype=np.int32)
    steps = 11
    input_dict = {
        "start": torch.tensor(start),
        "end": torch.tensor(end),
        "steps": steps,
        "dtype": torch.int32,
        "requires_grad": False
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    start = np.array(1+1j, dtype=np.complex64)
    end = np.array(5+5j, dtype=np.complex64)
    steps = 5
    input_dict = {
        "start": torch.tensor(start),
        "end": torch.tensor(end),
        "steps": steps,
        "dtype": torch.complex64,
        "requires_grad": False
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    start = 0.0
    end = 10.0
    steps = 5
    input_dict = {
        "start": torch.tensor(start),
        "end": torch.tensor(end),
        "steps": steps,
        "dtype": torch.float64,
        "requires_grad": True
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    
    start = -5
    end = 5
    steps = 11
    input_dict = {
        "start": torch.tensor(start),
        "end": torch.tensor(end),
        "steps": steps,
        "dtype": torch.int64,
        "requires_grad": False
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    return generated_inputs
def logcumsumexp_inputs():
    list_of_inputs = []
    input1 = torch.randn(5).numpy()
    dim1 = 0
    input_dict1 = {"input": input1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(3, 4).numpy()
    dim2 = 1
    input_dict2 = {"input": input2, "dim": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(2, 3, 4).numpy()
    dim3 = 0
    input_dict3 = {"input": input3, "dim": dim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(2, 3, 4).numpy()
    dim4 = 2
    input_dict4 = {"input": input4, "dim": dim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(2, 3, 4).numpy()
    dim5 = -1
    input_dict5 = {"input": input5, "dim": dim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = torch.randn(1, 5).numpy()
    dim6 = 1
    input_dict6 = {"input": input6, "dim": dim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = torch.randn(5).double().numpy()
    dim7 = 0
    input_dict7 = {"input": input7, "dim": dim7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.randint(-5, 5, (3, 4)).float().numpy()
    dim8 = 1
    input_dict8 = {"input": input8, "dim": dim8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs
def logspace_inputs():
    list_of_inputs = []
    start = np.array(1.0)
    end = np.array(10.0)
    steps = 5
    base = 10.0
    dtype = torch.float64
    requires_grad = False
    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "base": base,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    start = np.array(0.1)
    end = np.array(100.0)
    steps = 7
    base = 2.0
    dtype = torch.float32
    requires_grad = True
    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "base": base,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    start = np.array(-1.0)
    end = np.array(1.0)
    steps = 6
    base = 5.0
    dtype = torch.float64
    requires_grad = False
    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "base": base,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    start = np.array(0.0)
    end = np.array(5.0)
    steps = 8
    base = 3.0
    dtype = torch.float32
    requires_grad = True
    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "base": base,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    start = np.array(-2.0)
    end = np.array(2.0)
    steps = 9
    base = 10.0
    dtype = torch.float32
    requires_grad = True
    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "base": base,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def lstsq_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensors
    A = np.random.randn(5, 3).astype(np.float32)
    B = np.random.randn(5, 2).astype(np.float32)
    rcond = 1e-15
    input_dict = {"A": A, "B": B, "rcond": rcond}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Different shapes, double tensors
    A = np.random.randn(10, 5).astype(np.float64)
    B = np.random.randn(10, 1).astype(np.float64)
    rcond = 1e-8
    input_dict = {"A": A, "B": B, "rcond": rcond}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Overdetermined system
    A = np.random.randn(10, 3).astype(np.float64)
    B = np.random.randn(10, 5).astype(np.float64)
    rcond = 1e-12
    input_dict = {"A": A, "B": B, "rcond": rcond}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Underdetermined system
    A = np.random.randn(3, 10).astype(np.float32)
    B = np.random.randn(3, 2).astype(np.float32)
    rcond = 1e-6
    input_dict = {"A": A, "B": B, "rcond": rcond}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: A is square matrix, B is vector
    A = np.random.randn(5, 5).astype(np.float32)
    B = np.random.randn(5).astype(np.float32)
    rcond = 1e-14
    input_dict = {"A": A, "B": B, "rcond": rcond}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def lu_solve_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensors
    LU_data = torch.randn(3, 3).numpy()
    LU_pivots = torch.arange(1, 4).numpy()
    b = torch.randn(3, 1).numpy()
    input_dict = {"b": b, "LU_data": LU_data, "LU_pivots": LU_pivots}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Batch of matrices
    LU_data = torch.randn(2, 4, 4).numpy()
    LU_pivots = torch.randint(1, 5, (2, 4)).numpy()
    b = torch.randn(2, 4, 2).numpy()
    input_dict = {"b": b, "LU_data": LU_data, "LU_pivots": LU_pivots}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Single Matrix, Long pivots
    LU_data = torch.randn(5, 5).numpy()
    LU_pivots = torch.arange(1, 6).long().numpy()
    b = torch.randn(5, 3).numpy()
    input_dict = {"b": b, "LU_data": LU_data, "LU_pivots": LU_pivots}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Different sizes
    LU_data = torch.randn(4, 4).numpy()
    LU_pivots = torch.arange(1, 5).numpy()
    b = torch.randn(4, 4).numpy()
    input_dict = {"b": b, "LU_data": LU_data, "LU_pivots": LU_pivots}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Negative Values
    LU_data = torch.randn(3, 3).numpy() * -1
    LU_pivots = torch.arange(1, 4).numpy()
    b = torch.randn(3, 1).numpy() * -1
    input_dict = {"b": b, "LU_data": LU_data, "LU_pivots": LU_pivots}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def margin_ranking_loss_inputs():
    list_of_inputs = []
    # Input 1: Basic case with float tensors
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input2 = np.array([2.0, 2.5, 2.8], dtype=np.float32)
    target = np.array([1, -1, 1], dtype=np.int32)
    input_dict = {
        'input1': input1,
        'input2': input2,
        'target': target,
        'margin': 0.5,
        'size_average': True,
        'reduce': False,
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Different margin and reduction mode
    input1 = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    input2 = np.array([0.0, 0.5, 0.8], dtype=np.float32)
    target = np.array([-1, -1, 1], dtype=np.int32)
    input_dict = {
        'input1': input1,
        'input2': input2,
        'target': target,
        'margin': 1.0,
        'size_average': True,
        'reduce': False,
        'reduction': 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Multidimensional inputs
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input2 = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    target = np.array([[1, -1], [-1, 1]], dtype=np.int32)
    input_dict = {
        'input1': input1,
        'input2': input2,
        'target': target,
        'margin': 0.2,
        'size_average': True,
        'reduce': False,
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Different shapes
    input1 = np.array([1.0, 2.0], dtype=np.float32)
    input2 = np.array([2.0, 3.0], dtype=np.float32)
    target = np.array([1, -1], dtype=np.int32)
    input_dict = {
        'input1': input1,
        'input2': input2,
        'target': target,
        'margin': 0.3,
        'size_average': True,
        'reduce': False,
        'reduction': 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Zero margin
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input2 = np.array([2.0, 2.5, 2.8], dtype=np.float32)
    target = np.array([1, -1, 1], dtype=np.int32)
    input_dict = {
        'input1': input1,
        'input2': input2,
        'target': target,
        'margin': 0.0,
        'size_average': True,
        'reduce': False,
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def margin_ranking_loss_inputs():
    list_of_inputs = []
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input2 = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    target = np.array([1, -1, 1], dtype=np.int32)
    margin = 0.5
    size_average = True
    reduce = True
    reduction = 'mean'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input2 = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    target = np.array([[1, -1], [1,1]], dtype=np.int32)
    margin = 0.2
    size_average = False
    reduce = False
    reduction = 'sum'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    input2 = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    target = np.array([-1, -1, 1, 1], dtype=np.int32)
    margin = 1.0
    size_average = False
    reduce = True
    reduction = 'mean'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input2 = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    target = np.array([[1, -1], [1,1]], dtype=np.int32)
    margin = -0.5
    size_average = True
    reduce = True
    reduction = 'mean'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.array([1.0], dtype=np.float32)
    input2 = np.array([2.0], dtype=np.float32)
    target = np.array([1], dtype=np.int32)
    margin = 0.0
    size_average = True
    reduce = True
    reduction = 'sum'
    input_dict = {
        "input1": input1,
        "input2": input2,
        "target": target,
        "margin": margin,
        "size_average": size_average,
        "reduce": reduce,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def max_pool1d_inputs():
    list_of_inputs = []
    input1 = torch.randn(1, 3, 10).numpy()
    kernel_size1 = 3
    stride1 = 2
    padding1 = 1
    dilation1 = 1
    ceil_mode1 = False
    return_indices1 = False
    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "dilation": dilation1,
        "ceil_mode": ceil_mode1,
        "return_indices": return_indices1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(2, 5, 15).numpy()
    kernel_size2 = 4
    stride2 = 3
    padding2 = 0
    dilation2 = 2
    ceil_mode2 = True
    return_indices2 = False
    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "dilation": dilation2,
        "ceil_mode": ceil_mode2,
        "return_indices": return_indices2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(1, 1, 20).numpy()
    kernel_size3 = 5
    stride3 = 1
    padding3 = 2
    dilation3 = 1
    ceil_mode3 = False
    return_indices3 = True
    input_dict3 = {
        "input": input3,
        "kernel_size": kernel_size3,
        "stride": stride3,
        "padding": padding3,
        "dilation": dilation3,
        "ceil_mode": ceil_mode3,
        "return_indices": return_indices3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(3, 2, 12).numpy()
    kernel_size4 = 2
    stride4 = 2
    padding4 = 0
    dilation4 = 1
    ceil_mode4 = True
    return_indices4 = False
    input_dict4 = {
        "input": input4,
        "kernel_size": kernel_size4,
        "stride": stride4,
        "padding": padding4,
        "dilation": dilation4,
        "ceil_mode": ceil_mode4,
        "return_indices": return_indices4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(1, 4, 8).numpy()
    kernel_size5 = 2
    stride5 = None
    padding5 = 1
    dilation5 = 1
    ceil_mode5 = False
    return_indices5 = False
    input_dict5 = {
        "input": input5,
        "kernel_size": kernel_size5,
        "stride": kernel_size5,
        "padding": padding5,
        "dilation": dilation5,
        "ceil_mode": ceil_mode5,
        "return_indices": return_indices5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs
def max_pool2d_inputs():
    list_of_inputs = []
    input_float = torch.randn(1, 3, 32, 32).numpy()
    input_dict = {
        "input": input_float,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_int = torch.randint(0, 10, (1, 1, 16, 16)).numpy()
    input_dict = {
        "input": input_int,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_neg = torch.randn(2, 4, 28, 28) * -1.0
    input_neg = input_neg.numpy()
    input_dict = {
        "input": input_neg,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": True,
        "return_indices": False #return_indices can only be true for CUDA tensors
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_no_batch = torch.randn(3, 16, 16).numpy()
    input_dict = {
        "input": input_no_batch,
        "kernel_size": 4,
        "stride": 4,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_large = torch.randn(4, 8, 64, 64).numpy()
    input_dict = {
        "input": input_large,
        "kernel_size": 8,
        "stride": 8,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_non_square = torch.randn(1, 1, 32, 64).numpy()
    input_dict = {
        "input": input_non_square,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_padding = torch.randn(1, 1, 16, 16).numpy()
    input_dict = {
        "input": input_padding,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1, #Modified padding to 1
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dilation = torch.randn(1, 1, 16, 16).numpy()
    input_dict = {
        "input": input_dilation,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 2,
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def MaxPool2d_inputs():
    list_of_inputs = []
    input1 = torch.randn(1, 3, 32, 32).numpy()
    kernel_size1 = 3
    stride1 = 2
    padding1 = 1
    dilation1 = 1
    ceil_mode1 = False
    return_indices1 = False
    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "dilation": dilation1,
        "ceil_mode": ceil_mode1,
        "return_indices": return_indices1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randint(0, 10, (2, 1, 16, 16)).numpy()
    kernel_size2 = 2
    stride2 = 1
    padding2 = 0
    dilation2 = 1
    ceil_mode2 = True
    return_indices2 = True
    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "dilation": dilation2,
        "ceil_mode": ceil_mode2,
        "return_indices": return_indices2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(4, 3, 64, 64).numpy()
    kernel_size3 = (3, 2)
    stride3 = (2, 1)
    padding3 = (1, 0)
    dilation3 = (1, 2)
    ceil_mode3 = False
    return_indices3 = False
    input_dict3 = {
        "input": input3,
        "kernel_size": kernel_size3,
        "stride": stride3,
        "padding": padding3,
        "dilation": dilation3,
        "ceil_mode": ceil_mode3,
        "return_indices": return_indices3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(1, 1, 20, 20).numpy()
    kernel_size4 = 5
    stride4 = None
    padding4 = 2
    dilation4 = 1
    ceil_mode4 = False
    return_indices4 = False
    input_dict4 = {
        "input": input4,
        "kernel_size": kernel_size4,
        "stride": stride4,
        "padding": padding4,
        "dilation": dilation4,
        "ceil_mode": ceil_mode4,
        "return_indices": return_indices4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(2, 3, 10, 10).numpy()
    kernel_size5 = (2, 2)
    stride5 = (2, 2)
    padding5 = (1, 1)
    dilation5 = (1, 1)
    ceil_mode5 = True
    return_indices5 = False
    input_dict5 = {
        "input": input5,
        "kernel_size": kernel_size5,
        "stride": stride5,
        "padding": padding5,
        "dilation": dilation5,
        "ceil_mode": ceil_mode5,
        "return_indices": return_indices5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(1, 1, 7, 7).numpy()
    kernel_size6 = 3
    stride6 = 1
    padding6 = 0
    dilation6 = 2
    ceil_mode6 = False
    return_indices6 = False
    input_dict6 = {
        "input": input6,
        "kernel_size": kernel_size6,
        "stride": stride6,
        "padding": padding6,
        "dilation": dilation6,
        "ceil_mode": ceil_mode6,
        "return_indices": return_indices6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.randn(1, 3, 8, 8).numpy()
    kernel_size7 = (2, 3)
    stride7 = None
    padding7 = (1, 0)
    dilation7 = (2, 1)
    ceil_mode7 = True
    return_indices7 = True
    input_dict7 = {
        "input": input7,
        "kernel_size": kernel_size7,
        "stride": stride7,
        "padding": padding7,
        "dilation": dilation7,
        "ceil_mode": ceil_mode7,
        "return_indices": return_indices7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def max_pool3d_inputs():
    list_of_inputs = []
    input1 = torch.randn(1, 3, 10, 10, 10).numpy()
    input_dict1 = {
        "input": input1,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randint(0, 10, (2, 4, 8, 8, 8)).numpy()
    input_dict2 = {
        "input": input2,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": True,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(1, 1, 7, 7, 7).numpy()
    input_dict3 = {
        "input": input3,
        "kernel_size": 2,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(2, 3, 7, 7, 7).numpy()
    input_dict4 = {
        "input": input4,
        "kernel_size": (2, 3, 2),
        "stride": (1, 2, 1),
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "ceil_mode": True,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(1, 2, 9, 9, 9).numpy()
    input_dict5 = {
        "input": input5,
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def MaxPool3d_inputs():
    generated_inputs = []
    input1 = torch.randn(1, 3, 32, 32, 32).numpy()
    input_dict1 = {
        "input": input1,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    generated_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(2, 4, 16, 16, 16).numpy()
    input_dict2 = {
        "input": input2,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": True,
        "return_indices": False
    }
    generated_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randint(0, 10, (1, 1, 64, 64, 64)).numpy()
    input_dict3 = {
        "input": input3,
        "kernel_size": 4,
        "stride": 4,
        "padding": 2,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": True
    }
    generated_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(1, 2, 8, 8, 8).numpy()
    input_dict4 = {
        "input": input4,
        "kernel_size": 2,
        "stride": 2,
        "padding": 1,
        "dilation": 2,
        "ceil_mode": True,
        "return_indices": False
    }
    generated_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(4, 3, 12, 12, 12).numpy()
    input_dict5 = {
        "input": input5,
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": True
    }
    generated_inputs.append(copy.deepcopy(input_dict5))
    input6 = torch.randn(1, 1, 5, 5, 5).numpy()
    input_dict6 = {
        "input": input6,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    generated_inputs.append(copy.deepcopy(input_dict6))
    return generated_inputs
def mean_inputs():
    list_of_inputs = []
    # Case 1: 1D float tensor, no dim specified
    input1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1, "dim": None, "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Case 2: 2D int tensor, dim=0, keepdim=True
    input2 = torch.randint(-10, 10, (3, 4)).numpy()
    input_dict2 = {"input": input2, "dim": 0, "keepdim": True, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Case 3: 3D float tensor, dim=2, keepdim=False
    input3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {"input": input3, "dim": 2, "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Case 4: 2D complex tensor, dim=1, keepdim=True
    input4 = (torch.randn(2, 2) + 1j * torch.randn(2, 2)).numpy()
    input_dict4 = {"input": input4, "dim": 1, "keepdim": True, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Case 5: 0D float tensor (scalar), no dim
    input5 = torch.randn(1).numpy().item()
    input_dict5 = {"input": np.array(input5), "dim": None, "keepdim": False, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def mm_inputs():
    generated_inputs = []
    # Input 1: Basic float tensors
    input1 = np.random.rand(3, 4).astype(np.float32)
    mat2_1 = np.random.rand(4, 5).astype(np.float32)
    generated_inputs.append({"input": input1, "mat2": mat2_1})
    # Input 2: Float tensors with different shapes
    input2 = np.random.rand(1, 5).astype(np.float32)
    mat2_2 = np.random.rand(5, 1).astype(np.float32)
    generated_inputs.append({"input": input2, "mat2": mat2_2})
    # Input 3: Float tensors with negative values
    input3 = np.random.randn(2, 3).astype(np.float32)
    mat2_3 = np.random.randn(3, 2).astype(np.float32)
    generated_inputs.append({"input": input3, "mat2": mat2_3})
    # Input 4: Double tensors
    input4 = np.random.rand(3, 4).astype(np.float64)
    mat2_4 = np.random.rand(4, 5).astype(np.float64)
    generated_inputs.append({"input": input4, "mat2": mat2_4})
    # Input 5: Complex tensors
    input5 = (np.random.rand(2, 3) + 1j * np.random.rand(2, 3)).astype(np.complex64)
    mat2_5 = (np.random.rand(3, 2) + 1j * np.random.rand(3, 2)).astype(np.complex64)
    generated_inputs.append({"input": input5, "mat2": mat2_5})
    # Input 6: Larger tensors
    input6 = np.random.rand(10, 20).astype(np.float32)
    mat2_6 = np.random.rand(20, 10).astype(np.float32)
    generated_inputs.append({"input": input6, "mat2": mat2_6})
    # Input 7: Single element tensors
    input7 = np.random.rand(1, 1).astype(np.float32)
    mat2_7 = np.random.rand(1, 1).astype(np.float32)
    generated_inputs.append({"input": input7, "mat2": mat2_7})
    # Input 8: Tensors with zero values
    input8 = np.zeros((3, 4), dtype=np.float32)
    mat2_8 = np.zeros((4, 5), dtype=np.float32)
    generated_inputs.append({"input": input8, "mat2": mat2_8})
    return generated_inputs
def movedim_inputs():
    list_of_inputs = []
    # Input 1: 2D tensor, move axis 0 to 1
    input1 = torch.randn(2, 3).numpy()
    input_dict1 = {
        "input": input1,
        "source": 0,
        "destination": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 3D tensor, move axis 1 to 0
    input2 = torch.randint(0, 10, (2, 3, 4)).numpy()
    input_dict2 = {
        "input": input2,
        "source": 1,
        "destination": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 4D tensor, move axis 3 to -1 (last)
    input3 = torch.randn(2, 3, 4, 5).numpy()
    input_dict3 = {
        "input": input3,
        "source": 3,
        "destination": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 5D tensor, move axis -2 to 0
    input4 = torch.randn(2, 3, 4, 5, 6).numpy()
    input_dict4 = {
        "input": input4,
        "source": -2,
        "destination": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 3D tensor, move axis 0 to -1
    input5 = torch.randn(3, 4, 5).numpy()
    input_dict5 = {
        "input": input5,
        "source": 0,
        "destination": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: 1D tensor, move axis 0 to 0 (no change, but valid)
    input6 = torch.arange(5).numpy()
    input_dict6 = {
        "input": input6,
        "source": 0,
        "destination": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: 3D tensor, move axis -1 to 1
    input7 = torch.randn(2, 3, 4).numpy()
    input_dict7 = {
        "input": input7,
        "source": -1,
        "destination": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    # Input 8: Complex tensor, move axis 0 to 1
    input8 = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict8 = {
        "input": input8,
        "source": 0,
        "destination": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    # Input 9: Float tensor, move axis 1 to -2
    input9 = torch.randn(2, 3, 4, 5).numpy()
    input_dict9 = {
        "input": input9,
        "source": 1,
        "destination": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    return list_of_inputs
def mse_loss_inputs():
    list_of_inputs = []
    # Case 1: Basic float tensors, reduction='mean'
    input1 = np.random.randn(3, 5).astype(np.float32)
    target1 = np.random.randn(3, 5).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "target": target1,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Case 2: Integer tensors, reduction='sum' (Cast to float)
    input2 = np.random.randint(0, 10, size=(2, 4)).astype(np.float32)
    target2 = np.random.randint(0, 10, size=(2, 4)).astype(np.float32)
    input_dict2 = {
        "input": input2,
        "target": target2,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Case 3: 1D tensors, reduction='none' (removed because of error)
    # Case 4: Higher dimensional tensors (4D), reduction='mean'
    input4 = np.random.randn(1, 3, 28, 28).astype(np.float32)
    target4 = np.random.randn(1, 3, 28, 28).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "target": target4,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Case 5: Negative values, reduction='sum'
    input5 = np.random.randn(2, 2) * -1.0
    target5 = np.random.randn(2, 2) * -1.0
    input_dict5 = {
        "input": input5.astype(np.float32),
        "target": target5.astype(np.float32),
        "size_average": None,
        "reduce": None,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Case 6: Different shapes that are broadcastable, reduction='mean'
    input6 = np.random.randn(1, 5).astype(np.float32)
    target6 = np.random.randn(5).astype(np.float32)
    input_dict6 = {
        "input": input6,
        "target": target6,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def MSELoss_inputs():
    list_of_inputs = []
    # Case 1: Basic float tensors, same shape
    input1 = np.random.randn(3, 4).astype(np.float32)
    target1 = np.random.randn(3, 4).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "target": target1,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Case 2: One dimensional tensor
    input4 = np.random.randn(5).astype(np.float32)
    target4 = np.random.randn(5).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "target": target4,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Case 3: Negative values
    input6 = np.random.randn(2, 2).astype(np.float32) * -1
    target6 = np.random.randn(2, 2).astype(np.float32) * -1
    input_dict6 = {
        "input": input6,
        "target": target6,
        "size_average": None,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Case 4: Different type
    input2 = np.random.randn(3, 4).astype(np.float64)
    target2 = np.random.randn(3, 4).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "target": target2,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    # Case 5: Float tensors, same shape
    input7 = np.random.randn(2, 3).astype(np.float32)
    target7 = np.random.randn(2, 3).astype(np.float32)
    input_dict7 = {
        "input": input7,
        "target": target7,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def multilabel_soft_margin_loss_inputs():
    list_of_inputs = []
    # Case 1: Basic example with float tensors
    input_tensor = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    target_tensor = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.float32)
    weight_tensor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Example with different reduction method
    input_tensor = np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float32)
    target_tensor = np.array([[1, 0], [0, 1]], dtype=np.float32)
    weight_tensor = None
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "size_average": False,
        "reduce": True,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Example with no reduction
    input_tensor = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    target_tensor = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.float32)
    weight_tensor = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "size_average": False,
        "reduce": False,
        "reduction": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: Example with negative values
    input_tensor = np.array([[-0.1, 0.2], [0.4, -0.5]], dtype=np.float32)
    target_tensor = np.array([[0, 1], [1, 0]], dtype=np.float32)
    weight_tensor = None
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Example with scalar weight
    input_tensor = np.array([[0.6, 0.7], [0.8, 0.9]], dtype=np.float32)
    target_tensor = np.array([[1, 0], [0, 1]], dtype=np.float32)
    weight_tensor = np.array([2.0, 0.5], dtype=np.float32)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "size_average": False,
        "reduce": True,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 6: 3D input and target
    input_tensor = np.random.rand(2, 3, 4).astype(np.float32)
    target_tensor = np.random.randint(0, 2, size=(2, 3, 4)).astype(np.float32)
    weight_tensor = np.random.rand(4).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: Example with different data
    input_tensor = np.array([[0.7, -0.8], [-0.9, 1.0]], dtype=np.float32)
    target_tensor = np.array([[0, 1], [1, 0]], dtype=np.float32)
    weight_tensor = None
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "size_average": False,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def MultiLabelSoftMarginLoss_inputs():
    list_of_inputs = []
    input1 = np.random.randn(3, 5).astype(np.float32)
    target1 = np.random.randint(0, 2, size=(3, 5)).astype(np.float32)
    weight1 = np.random.rand(5).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "target": target1,
        "weight": weight1,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.random.randn(2, 4).astype(np.float64)
    target2 = np.random.randint(0, 2, size=(2, 4)).astype(np.float64)
    weight2 = None
    input_dict2 = {
        "input": input2,
        "target": target2,
        "weight": weight2,
        "size_average": False,
        "reduce": True,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.random.randn(1, 7).astype(np.float32)
    target3 = np.random.randint(0, 2, size=(1, 7)).astype(np.float32)
    weight3 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "target": target3,
        "weight": weight3,
        "size_average": False,
        "reduce": False,
        "reduction": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.random.randn(4, 3, 2).astype(np.float32)
    target4 = np.random.randint(0, 2, size=(4, 3, 2)).astype(np.float32)
    weight4 = None
    input_dict4 = {
        "input": input4,
        "target": target4,
        "weight": weight4,
        "size_average": True,
        "reduce": False,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    target5 = np.random.randint(0, 2, size=(2, 2, 2, 2)).astype(np.float32)
    weight5 = None
    input_dict5 = {
        "input": input5,
        "target": target5,
        "weight": weight5,
        "size_average": False,
        "reduce": False,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def multimarginloss_inputs():
    list_of_inputs = []
    # Example 1: Basic case
    input_1 = np.array([[1.0, -0.5, 0.2], [2.0, 0.3, -0.1]], dtype=np.float32)
    target_1 = np.array([0, 2], dtype=np.int64)
    input_dict_1 = {"input": input_1, "target": target_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    # Example 2: Different input size and target values
    input_2 = np.array([[0.5, 1.2, -0.8, 0.1], [-1.0, 0.4, 2.0, -0.5]], dtype=np.float32)
    target_2 = np.array([3, 1], dtype=np.int64)
    input_dict_2 = {"input": input_2, "target": target_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    # Example 3: More examples
    input_3 = np.array([[1.5, 0.2, -0.7, 0.9, -0.3], [0.1, -1.2, 0.5, -0.6, 1.8]], dtype=np.float32)
    target_3 = np.array([4, 0], dtype=np.int64)
    input_dict_3 = {"input": input_3, "target": target_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    # Example 4: Larger input and target
    input_4 = np.array([[0.8, -0.2, 1.1, -0.5, 0.3, -1.0, 0.6], [-0.4, 1.3, -0.9, 0.2, -0.7, 0.1, 1.2]], dtype=np.float32)
    target_4 = np.array([6, 1], dtype=np.int64)
    input_dict_4 = {"input": input_4, "target": target_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    # Example 5: Different batch size
    input_5 = np.array([[0.3, -0.1, 0.5], [-0.2, 0.4, 0.1], [0.6, 0.2, -0.3]], dtype=np.float32)
    target_5 = np.array([1, 2, 0], dtype=np.int64)
    input_dict_5 = {"input": input_5, "target": target_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    # Example 6: all values positive
    input_6 = np.array([[0.3, 0.1, 0.5], [0.2, 0.4, 0.1], [0.6, 0.2, 0.3]], dtype=np.float32)
    target_6 = np.array([1, 2, 0], dtype=np.int64)
    input_dict_6 = {"input": input_6, "target": target_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    # Example 7: Using float64
    input_7 = np.array([[1.0, -0.5, 0.2], [2.0, 0.3, -0.1]], dtype=np.float64)
    target_7 = np.array([0, 2], dtype=np.int64)
    input_dict_7 = {"input": input_7, "target": target_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    return list_of_inputs
def mv_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensors
    input_matrix = np.random.randn(5, 3).astype(np.float32)
    input_vector = np.random.randn(3).astype(np.float32)
    input_dict = {"input": input_matrix, "vec": input_vector}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Integer tensors
    input_matrix = np.random.randint(1, 10, size=(4, 5)).astype(np.int32)
    input_vector = np.random.randint(1, 10, size=(5)).astype(np.int32)
    input_dict = {"input": input_matrix, "vec": input_vector}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Negative values
    input_matrix = np.random.randn(6, 4).astype(np.float32) * -1
    input_vector = np.random.randn(4).astype(np.float32) * -1
    input_dict = {"input": input_matrix, "vec": input_vector}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Different shapes
    input_matrix = np.random.randn(2, 7).astype(np.float32)
    input_vector = np.random.randn(7).astype(np.float32)
    input_dict = {"input": input_matrix, "vec": input_vector}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Double precision
    input_matrix = np.random.randn(3, 2).astype(np.float64)
    input_vector = np.random.randn(2).astype(np.float64)
    input_dict = {"input": input_matrix, "vec": input_vector}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Complex tensors
    input_matrix = (np.random.randn(4, 3) + 1j * np.random.randn(4, 3)).astype(np.complex64)
    input_vector = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    input_dict = {"input": input_matrix, "vec": input_vector}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 7: Matrix with a single row
    input_matrix = np.random.randn(1, 5).astype(np.float32)
    input_vector = np.random.randn(5).astype(np.float32)
    input_dict = {"input": input_matrix, "vec": input_vector}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Matrix with a single column
    input_matrix = np.random.randn(5, 1).astype(np.float32)
    input_vector = np.random.randn(1).astype(np.float32)
    input_dict = {"input": input_matrix, "vec": input_vector}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def nansum_inputs():
    list_of_inputs = []
    input1 = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 5.0]])
    dim1 = (0,)
    keepdim1 = False
    dtype1 = np.float32
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "keepdim": keepdim1,
        "dtype": dtype1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[[1.0, 2.0, np.nan], [3.0, np.nan, 5.0]], [[6.0, 7.0, 8.0], [9.0, 10.0, np.nan]]])
    dim2 = (0, 2)
    keepdim2 = True
    dtype2 = np.float64
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "keepdim": keepdim2,
        "dtype": dtype2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([np.nan, np.nan, np.nan])
    dim3 = (0,)
    keepdim3 = False
    dtype3 = np.float64
    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "keepdim": keepdim3,
        "dtype": dtype3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    dim4 = (1,)
    keepdim4 = True
    dtype4 = np.float32
    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "keepdim": keepdim4,
        "dtype": dtype4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([[[1.0, 2.0], [np.nan, 4.0]], [[5.0, np.nan], [7.0, 8.0]]])
    dim5 = (0, 1, 2)
    keepdim5 = False
    dtype5 = np.float16
    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "keepdim": keepdim5,
        "dtype": dtype5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def narrow_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 4, 5).numpy()
    dim1 = 1
    start1 = 0
    length1 = 2
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "start": start1,
        "length": length1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randint(0, 10, (2, 5, 7)).numpy()
    dim2 = 0
    start2 = 1
    length2 = 1
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "start": start2,
        "length": length2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(4, 6).numpy()
    dim3 = 1
    start3 = 2
    length3 = 3
    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "start": start3,
        "length": length3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(2, 3, 4, 5).numpy()
    dim4 = 2
    start4 = 1
    length4 = 2
    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "start": start4,
        "length": length4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(5).numpy()
    dim5 = 0
    start5 = 2
    length5 = 2
    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "start": start5,
        "length": length5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = torch.randn(1, 2, 3, 4, 5).numpy()
    dim6 = 3
    start6 = 0
    length6 = 3
    input_dict6 = {
        "input": input6,
        "dim": dim6,
        "start": start6,
        "length": length6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = torch.randn(2,2).numpy()
    dim7 = 0
    start7 = 0
    length7 = 2
    input_dict7 = {
        "input": input7,
        "dim": dim7,
        "start": start7,
        "length": length7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def nll_loss_inputs():
    list_of_inputs = []
    # Case 1: Basic case with float input and long target
    input_dict = {
        "input": np.array([[-0.5, -0.2, -0.3], [-0.1, -0.8, -0.1]], dtype=np.float64),
        "target": np.array([0, 2], dtype=np.int64),
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2:  Input with different shape and weights provided
    input_dict = {
        "input": np.array([[-1.2, -0.3, -0.5, -0.1]], dtype=np.float64),
        "target": np.array([3], dtype=np.int64),
        "weight": np.array([0.2, 0.3, 0.1, 0.4], dtype=np.float64),
        "ignore_index": -100,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Input with ignore_index
    input_dict = {
        "input": np.array([[-0.7, -0.1, -0.2], [-0.4, -0.5, -0.1], [-0.2, -0.6, -0.2]], dtype=np.float64),
        "target": np.array([0, 1, 2], dtype=np.int64),
        "weight": None,
        "ignore_index": 1,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: Input with 'none' reduction
    input_dict = {
        "input": np.array([[-0.1, -0.9, -0.0], [-0.3, -0.4, -0.3]], dtype=np.float64),
        "target": np.array([2, 0], dtype=np.int64),
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Input with a weight and ignore_index specified.
    input_dict = {
        "input": np.array([[-0.6, -0.2, -0.2], [-0.3, -0.3, -0.4]], dtype=np.float64),
        "target": np.array([0, 1], dtype=np.int64),
        "weight": np.array([0.5, 0.3, 0.2], dtype=np.float64),
        "ignore_index": 0,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def NLLLoss_inputs():
    list_of_inputs = []
    # Case 1: Basic case with 1D input and target
    input = np.array([[-0.8, -0.2, -0.3]], dtype=np.float32)
    target = np.array([0], dtype=np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: 2D input and 1D target, different reduction
    input = np.array([[-0.8, -0.2, -0.3], [-0.1, -0.9, -0.5]], dtype=np.float32)
    target = np.array([0, 1], dtype=np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: With weight
    input = np.array([[-0.8, -0.2, -0.3], [-0.1, -0.9, -0.5]], dtype=np.float32)
    target = np.array([0, 1], dtype=np.int64)
    weight = np.array([0.2, 0.8, 0.5], dtype=np.float32)
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": -100,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: With ignore_index
    input = np.array([[-0.8, -0.2, -0.3], [-0.1, -0.9, -0.5]], dtype=np.float32)
    target = np.array([0, -100], dtype=np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Different input shape
    input = np.array([[-0.8, -0.2], [-0.1, -0.9], [-0.5, -0.3]], dtype=np.float32)
    target = np.array([0, 1, 0], dtype=np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def normalize_inputs():
    list_of_inputs = []
    input1 = np.random.rand(3, 4).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "dim": 1,
        "eps": 1e-12,
        "p": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "dim": 0,
        "eps": 1e-8,
        "p": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.random.rand(5,).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "dim": 0,
        "eps": 1e-6,
        "p": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.random.rand(2, 2, 2, 2).astype(np.float64)
    input_dict4 = {
        "input": input4,
        "dim": 2,
        "eps": 1e-10,
        "p": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = (np.random.rand(3, 4) - 0.5).astype(np.float32) #negative values
    input_dict5 = {
        "input": input5,
        "dim": 1,
        "eps": 1e-12,
        "p": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.random.rand(1, 5).astype(np.float64)
    input_dict6 = {
        "input": input6,
        "dim": 1,
        "eps": 1e-8,
        "p": 1.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([[1,2],[3,4]]).astype(np.float32)
    input_dict7 = {
        "input": input7,
        "dim": 0,
        "eps": 1e-6,
        "p": float('inf')
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def ones_inputs():
    list_of_inputs = []
    input1 = {
        "size": [5]
    }
    list_of_inputs.append(copy.deepcopy(input1))
    input2 = {
        "size": [2, 3]
    }
    list_of_inputs.append(copy.deepcopy(input2))
    input3 = {
        "size": [4, 5, 6]
    }
    list_of_inputs.append(copy.deepcopy(input3))
    input4 = {
        "size": [7, 8]
    }
    list_of_inputs.append(copy.deepcopy(input4))
    input5 = {
        "size": [1, 2, 3, 4]
    }
    list_of_inputs.append(copy.deepcopy(input5))
    input6 = {
        "size": [1, 1]
    }
    list_of_inputs.append(copy.deepcopy(input6))
    
    return list_of_inputs
def pairwise_distance_inputs():
    list_of_inputs = []
    x1 = np.random.randn(10, 5).astype(np.float32)
    x2 = np.random.randn(10, 5).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 2.0,
        "eps": 1e-6,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = np.random.randn(5, 3, 2).astype(np.float64)
    x2 = np.random.randn(5, 3, 2).astype(np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 1.5,
        "eps": 1e-8,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = np.random.randn(20, 1).astype(np.float32)
    x2 = np.random.randn(20, 1).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 0.0,
        "eps": 1e-4,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = np.random.randn(3, 4, 5).astype(np.float32)
    x2 = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 3.0,
        "eps": 0.0,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = np.random.randn(7, 2, 3, 4).astype(np.float32)
    x2 = np.random.randn(7, 2, 3, 4).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 1.0,
        "eps": 1e-12,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = np.random.randn(4, 2).astype(np.float32)
    x2 = np.random.randn(4, 2).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 2.0,
        "eps": 1e-6,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.random.randn(8, 3).astype(np.float32)
    x2 = np.random.randn(8, 3).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": 2.5,
        "eps": 1e-5,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def pixel_shuffle_inputs():
    list_of_inputs = []
    # Input 1: Basic test case with float32 and upscale_factor = 2
    input1 = torch.randn(1, 4, 5, 5).numpy()
    upscale_factor1 = 2
    input_dict1 = {"input": input1, "upscale_factor": upscale_factor1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Different input size, upscale_factor = 3
    input2 = torch.randn(1, 9, 4, 4).numpy()
    upscale_factor2 = 3
    input_dict2 = {"input": input2, "upscale_factor": upscale_factor2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Larger input with upscale_factor = 4
    input3 = torch.randn(2, 16, 8, 8).numpy()
    upscale_factor3 = 4
    input_dict3 = {"input": input3, "upscale_factor": upscale_factor3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Single channel input
    input4 = torch.randn(1, 4, 5, 5).numpy()
    upscale_factor4 = 2
    input_dict4 = {"input": input4, "upscale_factor": upscale_factor4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Different batch size, float64 input
    input5 = torch.randn(3, 4, 6, 6, dtype=torch.float64).numpy()
    upscale_factor5 = 2
    input_dict5 = {"input": input5, "upscale_factor": upscale_factor5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Channel divisible by upscale_factor squared
    input6 = torch.randn(1, 8, 5, 5).numpy()
    upscale_factor6 = 2
    input_dict6 = {"input": input6, "upscale_factor": upscale_factor6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Input with height and width of size 1
    input7 = torch.randn(1, 4, 1, 1).numpy()
    upscale_factor7 = 2
    input_dict7 = {"input": input7, "upscale_factor": upscale_factor7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    # Input 8: Different channel size and upscale factor
    input8 = torch.randn(1, 36, 4, 4).numpy()
    upscale_factor8 = 6
    input_dict8 = {"input": input8, "upscale_factor": upscale_factor8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9: Another valid case
    input9 = torch.randn(2, 16, 5, 5).numpy()
    upscale_factor9 = 4
    input_dict9 = {"input": input9, "upscale_factor": upscale_factor9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    return list_of_inputs
def pixelshuffle_inputs():
    list_of_inputs = []
    input1 = np.random.rand(1, 9, 4, 4).astype(np.float32)
    upscale_factor1 = 3
    input_dict1 = {"input": input1, "upscale_factor": upscale_factor1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.random.rand(2, 4, 5, 6).astype(np.float64)
    upscale_factor2 = 2
    input_dict2 = {"input": input2, "upscale_factor": upscale_factor2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.random.rand(1, 16, 3, 3).astype(np.int32)
    upscale_factor3 = 4
    input_dict3 = {"input": input3, "upscale_factor": upscale_factor3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.random.rand(4, 1, 8, 8).astype(np.float16)
    upscale_factor4 = 1
    input_dict4 = {"input": input4, "upscale_factor": upscale_factor4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.random.rand(1, 25, 2, 2).astype(np.float32)
    upscale_factor5 = 5
    input_dict5 = {"input": input5, "upscale_factor": upscale_factor5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.random.rand(1, 64, 1, 1).astype(np.float32)
    upscale_factor6 = 8
    input_dict6 = {"input": input6, "upscale_factor": upscale_factor6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.random.rand(2, 4, 10, 12).astype(np.float32)
    upscale_factor7 = 2
    input_dict7 = {"input": input7, "upscale_factor": upscale_factor7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def poisson_nll_loss_inputs():
    list_of_inputs = []
    # Input 1: Basic example with float tensors, log_input=True
    input_dict = {
        "input": torch.randn(3, 5).numpy(),
        "target": torch.randint(0, 10, (3, 5)).float().numpy(),
        "log_input": True,
        "full": False,
        "size_average": None,
        "eps": 1e-8,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Integer target, log_input=False
    input_dict = {
        "input": torch.rand(2, 4).numpy(),
        "target": torch.randint(0, 5, (2, 4)).float().numpy(),
        "log_input": False,
        "full": True,
        "size_average": None,
        "eps": 1e-6,
        "reduce": None,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Different dimensions, log_input=True, reduction = 'none' - REMOVED THIS AS IT CAUSED ERROR
    # input_dict = {
    #     "input": torch.randn(1, 3, 8, 8).numpy(),
    #     "target": torch.randint(0, 5, (1, 3, 8, 8)).float().numpy(),
    #     "log_input": True,
    #     "full": False,
    #     "size_average": None,
    #     "eps": 1e-8,
    #     "reduce": None,
    #     "reduction": 'none'
    # }
    # list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: 1D tensors
    input_dict = {
        "input": torch.randn(10).numpy(),
        "target": torch.randint(0, 5, (10,)).float().numpy(),
        "log_input": False,
        "full": True,
        "size_average": None,
        "eps": 1e-8,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: No reduction - using 'mean' instead for scalar output
    input_dict = {
        "input": torch.randn(2, 3).numpy(),
        "target": torch.randint(0, 5, (2, 3)).float().numpy(),
        "log_input": True,
        "full": False,
        "size_average": None,
        "eps": 1e-8,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Different shape, log_input=False
    input_dict = {
        "input": torch.rand(5, 2, 2).numpy(),
        "target": torch.randint(0, 5, (5, 2, 2)).float().numpy(),
        "log_input": False,
        "full": True,
        "size_average": None,
        "eps": 1e-8,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def PoissonNLLLoss_inputs():
    list_of_inputs = []
    # Input 1: Basic example with log_input=True, full=False
    input1 = np.random.randn(3, 5).astype(np.float32)
    target1 = np.random.randint(0, 10, size=(3, 5)).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "target": target1,
        "log_input": True,
        "full": False,
        "eps": 1e-8,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: log_input=False, full=True, different reduction
    input2 = np.random.rand(2, 4, 6).astype(np.float64) * 10  # Positive values for exp
    target2 = np.random.randint(0, 5, size=(2, 4, 6)).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "target": target2,
        "log_input": False,
        "full": True,
        "eps": 1e-6,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Different shape, log_input=True, full=False
    input3 = np.random.randn(1, 7, 7).astype(np.float32)
    target3 = np.random.randint(0, 8, size=(1, 7, 7)).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "target": target3,
        "log_input": True,
        "full": False,
        "eps": 1e-10,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Scalar input and target, log_input=False, full=True
    input4 = np.array(5.0).astype(np.float64)
    target4 = np.array(2).astype(np.float64)
    input_dict4 = {
        "input": input4,
        "target": target4,
        "log_input": False,
        "full": True,
        "eps": 1e-5,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: 1D tensor
    input5 = np.random.randn(10).astype(np.float32)
    target5 = np.random.randint(0, 10, size=(10)).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "target": target5,
        "log_input": True,
        "full": False,
        "eps": 1e-8,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def polar_inputs():
    list_of_inputs = []
    abs_val = np.array([1.0, 2.0, 3.0])
    angle_val = np.array([0.0, np.pi/2, np.pi])
    input_dict = {"abs": abs_val, "angle": angle_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    abs_val = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    angle_val = np.array([[np.pi, 0.0], [np.pi/4, np.pi/2]])
    input_dict = {"abs": abs_val, "angle": angle_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    abs_val = np.array([1, 2, 3], dtype=np.int32)
    angle_val = np.array([0, 1, 2], dtype=np.int32) * np.pi / 4
    input_dict = {"abs": abs_val.astype(np.float32), "angle": angle_val.astype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    abs_val = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    angle_val = np.array([[[0.0, np.pi], [np.pi/2, np.pi/4]], [[np.pi/3, np.pi/6], [np.pi/8, np.pi/5]]])
    input_dict = {"abs": abs_val, "angle": angle_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    abs_val = np.array([1.5, 2.5, 3.5])
    angle_val = np.array([-np.pi/2, -np.pi/4, 0.0])
    input_dict = {"abs": abs_val, "angle": angle_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    abs_val = np.array([0.0, 0.0, 0.0])
    angle_val = np.array([0.0, np.pi, -np.pi])
    input_dict = {"abs": abs_val, "angle": angle_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    abs_val = np.array([1.0])
    angle_val = np.array([np.pi/2])
    input_dict = {"abs": abs_val, "angle": angle_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def pow_inputs():
    list_of_inputs = []
    # Case 1: Float input and float exponent
    input_tensor = np.random.rand(2, 3).astype(np.float32)
    exponent_tensor = np.array(2.0).astype(np.float32)
    input_dict = {"input": torch.from_numpy(input_tensor), "exponent": torch.from_numpy(exponent_tensor)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Int input and int exponent
    input_tensor = np.random.randint(1, 5, size=(3, 4), dtype=np.int32)
    exponent_tensor = np.array(3).astype(np.int32)
    input_dict = {"input": torch.from_numpy(input_tensor), "exponent": torch.from_numpy(exponent_tensor)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Float input and float tensor exponent
    input_tensor = np.random.rand(4, 2).astype(np.float64)
    exponent_tensor = np.random.rand(4, 2).astype(np.float64)
    input_dict = {"input": torch.from_numpy(input_tensor), "exponent": torch.from_numpy(exponent_tensor)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: Int input and float exponent
    input_tensor = np.random.randint(-5, 5, size=(2, 2), dtype=np.int64)
    exponent_tensor = np.array(0.5).astype(np.float64)
    input_dict = {"input": torch.from_numpy(input_tensor), "exponent": torch.tensor(exponent_tensor)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Negative input and float exponent
    input_tensor = np.random.randn(3, 3).astype(np.float32) * -1
    exponent_tensor = np.array(2.0).astype(np.float32)
    input_dict = {"input": torch.from_numpy(input_tensor), "exponent": torch.from_numpy(exponent_tensor)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def prelu_inputs():
    list_of_inputs = []
    # Input 1: Basic float input and weight
    input1 = np.random.randn(3, 4).astype(np.float32)
    weight1 = np.array([0.25], dtype=np.float32)
    input_dict1 = {"input": input1, "weight": weight1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Input with negative values and a different alpha
    input2 = np.random.randn(2, 2, 2).astype(np.float32)
    weight2 = np.array([-0.1], dtype=np.float32)
    input_dict2 = {"input": input2, "weight": weight2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3:  1D input with a single alpha value
    input3 = np.random.randn(5).astype(np.float32)
    weight3 = np.array([0.01], dtype=np.float32)
    input_dict3 = {"input": input3, "weight": weight3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4:  Larger input with a different alpha
    input4 = np.random.randn(1, 3, 28, 28).astype(np.float32)
    weight4 = np.array([0.0], dtype=np.float32)
    input_dict4 = {"input": input4, "weight": weight4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Multiple alpha values for each channel (requires input to have channels)
    input5 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    weight5 = np.array([0.25, -0.25, 0.0], dtype=np.float32)
    input_dict5 = {"input": input5, "weight": weight5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6:  Scalar input
    input6 = np.array(1.5, dtype=np.float32)
    weight6 = np.array([0.3], dtype=np.float32)
    input_dict6 = {"input": input6, "weight": weight6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs
def PReLU_inputs():
    list_of_inputs = []
    input1 = np.random.randn(3, 4).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "num_parameters": 1,
        "init": 0.25
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict2 = {
        "input": input2,
        "num_parameters": 1,
        "init": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.random.randn(1, 5, 5, 5).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "num_parameters": 1,
        "init": -0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "num_parameters": 1,
        "init": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.random.randn(4).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "num_parameters": 1,
        "init": 0.01
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict6 = {
        "input": input6,
        "num_parameters": 3,
        "init": 0.25
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict7 = {
        "input": input7,
        "num_parameters": 3,
        "init": -0.1
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def rand_inputs():
    generated_inputs = []
    generated_inputs.append({"size": np.array([1]).item()})
    generated_inputs.append({"size": np.array([2, 3]).tolist()})
    generated_inputs.append({"size": np.array([4, 5, 6]).tolist()})
    generated_inputs.append({"size": np.array([2, 3, 4, 5]).tolist()})
    generated_inputs.append({"size": np.array([1, 1, 1, 1, 1]).tolist()})
    generated_inputs.append({"size": np.array([10]).item()})
    generated_inputs.append({"size": np.array([2, 7]).tolist()})
    generated_inputs.append({"size": np.array([3, 1, 5]).tolist()})
    generated_inputs.append({"size": np.array([6, 2, 8, 3]).tolist()})
    generated_inputs.append({"size": np.array([1, 2, 3, 4, 5, 6]).tolist()})
    generated_inputs.append({"size": np.array([2, 1, 4]).tolist()})
    return generated_inputs
def reshape_inputs():
    list_of_inputs = []
    # Input 1: 1D float tensor
    input1 = torch.randn(10).numpy()
    shape1 = (2, 5)
    input_dict1 = {"input": input1, "shape": shape1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D int tensor
    input2 = torch.randint(0, 10, (5, 4)).numpy()
    shape2 = (2, 2, 5)
    input_dict2 = {"input": input2, "shape": shape2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 3D complex tensor
    input3 = torch.randn(2, 3, 4, dtype=torch.complex64).numpy()
    shape3 = (3, 8)
    input_dict3 = {"input": input3, "shape": shape3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 4D bool tensor
    input4 = torch.randint(0, 2, (2, 2, 2, 2)).bool().numpy()
    shape4 = (4, 4)
    input_dict4 = {"input": input4, "shape": shape4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 5D float tensor
    input5 = torch.randn(1, 2, 3, 4, 5).numpy()
    shape5 = (2, 3, 20)
    input_dict5 = {"input": input5, "shape": shape5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Reshape to -1 (infer size)
    input6 = torch.randn(2, 3, 4).numpy()
    shape6 = (-1,)
    input_dict6 = {"input": input6, "shape": shape6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: Reshape from -1
    input7 = torch.randn(24).numpy()
    shape7 = (2, 3, -1)
    input_dict7 = {"input": input7, "shape": shape7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    # Input 8: empty tensor
    input8 = torch.empty(0).numpy()
    shape8 = (0,)
    input_dict8 = {"input": input8, "shape": shape8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs
def rot90_inputs():
    list_of_inputs = []
    # Input 1: 2D float tensor, k=1, dims=(0, 1)
    input1 = torch.randn(4, 5).numpy()
    input_dict1 = {
        "input": input1,
        "k": 1,
        "dims": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 3D int tensor, k=2, dims=(1, 2)
    input2 = torch.randint(0, 10, (3, 6, 7)).numpy()
    input_dict2 = {
        "input": input2,
        "k": 2,
        "dims": (1, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 4D complex tensor, k=-1, dims=(2, 3)
    input3 = torch.randn(2, 3, 4, 5, dtype=torch.complex64).numpy()
    input_dict3 = {
        "input": input3,
        "k": -1,
        "dims": (2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 2D float tensor, k=0, dims=(0, 1)
    input4 = torch.randn(5, 5).numpy()
    input_dict4 = {
        "input": input4,
        "k": 0,
        "dims": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 3D float tensor, k=3, dims=(0, 2)
    input5 = torch.randn(2, 3, 4).numpy()
    input_dict5 = {
        "input": input5,
        "k": 3,
        "dims": (0, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: 5D int tensor, k=-2, dims=(1, 4)
    input6 = torch.randint(0, 10, (1, 2, 3, 4, 5)).numpy()
    input_dict6 = {
        "input": input6,
        "k": -2,
        "dims": (1, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: 2D bool tensor, k=1, dims=(0, 1)
    input7 = torch.randint(0, 2, (4, 5)).bool().numpy()
    input_dict7 = {
        "input": input7,
        "k": 1,
        "dims": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    # Input 8: 4D tensor
    input8 = torch.randn(2, 3, 4, 5).numpy()
    input_dict8 = {
        "input": input8,
        "k": 1,
        "dims": (0, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
        
    return list_of_inputs
def softmax_inputs():
    list_of_inputs = []
    # Input 1: 1D tensor
    input1 = np.array([1.0, 2.0, 3.0])
    dim1 = 0
    input_dict1 = {"input": input1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D tensor with negative values
    input2 = np.array([[-1.0, 0.5, 2.0], [3.0, -2.0, 1.5]])
    dim2 = 1
    input_dict2 = {"input": input2, "dim": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 3D tensor
    input3 = np.random.rand(2, 3, 4)
    dim3 = 2
    input_dict3 = {"input": input3, "dim": dim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 4D tensor
    input4 = np.random.randn(1, 5, 5, 2)
    dim4 = 3
    input_dict4 = {"input": input4, "dim": dim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: large values
    input5 = np.array([[1000.0, 1001.0, 1002.0], [1003.0, 999.0, 1000.5]])
    dim5 = 1
    input_dict5 = {"input": input5, "dim": dim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: tensor with a different dim value
    input6 = np.random.rand(2, 3, 4)
    dim6 = 0
    input_dict6 = {"input": input6, "dim": dim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def softmin_inputs():
    generated_inputs = []
    # Input 1: 1D tensor, dim=0
    input1 = np.array([1.0, 2.0, 3.0])
    input_dict1 = {"input": input1, "dim": 0}
    generated_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D tensor, dim=0
    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input_dict2 = {"input": input2, "dim": 0}
    generated_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 2D tensor, dim=1
    input3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input_dict3 = {"input": input3, "dim": 1}
    generated_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 3D tensor, dim=2
    input4 = np.random.rand(2, 3, 4)
    input_dict4 = {"input": input4, "dim": 2}
    generated_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 1D tensor with negative values, dim=0
    input5 = np.array([-1.0, -2.0, -3.0])
    input_dict5 = {"input": input5, "dim": 0}
    generated_inputs.append(copy.deepcopy(input_dict5))
    return generated_inputs
def solve_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensors, left=True
    A = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    input_dict = {"A": A, "B": B, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Float tensors, left=False
    A = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    input_dict = {"A": A, "B": B, "left": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Different shapes, left=True
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 7.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    B = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], dtype=np.float32)
    input_dict = {"A": A, "B": B, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Different shapes, left=False
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 7.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    B = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], dtype=np.float32)
    input_dict = {"A": A, "B": B, "left": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Negative values, left=True
    A = np.array([[-1.0, 2.0], [3.0, -5.0]], dtype=np.float32)
    B = np.array([[5.0, -6.0], [-7.0, 8.0]], dtype=np.float32)
    input_dict = {"A": A, "B": B, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def split_inputs():
    list_of_inputs = []
    # Input 1: Basic case, split into equal chunks
    tensor = torch.randn(10).numpy()
    split_size_or_sections = 2
    dim = 0
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Split into unequal chunks
    tensor = torch.randn(10).numpy()
    split_size_or_sections = [2, 3, 5]
    dim = 0
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Multi-dimensional tensor
    tensor = torch.randn(4, 4).numpy()
    split_size_or_sections = 2
    dim = 1
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Splitting along different dimension
    tensor = torch.randn(2, 3, 4).numpy()
    split_size_or_sections = 2
    dim = 0
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Splitting a 3D tensor into sections
    tensor = torch.randn(2, 6, 4).numpy()
    split_size_or_sections = [1, 2, 3]
    dim = 1
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Splitting a tensor where split_size is larger than the dimension
    tensor = torch.randn(5).numpy()
    split_size_or_sections = 10
    dim = 0
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Integer tensor
    tensor = torch.randint(0, 10, (5,)).numpy()
    split_size_or_sections = 2
    dim = 0
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 8: Empty tensor
    tensor = torch.empty(0).numpy()
    split_size_or_sections = 1
    dim = 0
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def std_mean_inputs():
    list_of_inputs = []
    input1 = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "dim": 1,
        "correction": 0,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.random.randn(2, 2, 2).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "dim": 0,
        "correction": 1,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input5 = np.random.randn(1, 5).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "dim": 1,
        "correction": 0,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict6 = {
        "input": input6,
        "dim": (0, 2),
        "correction": 2,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([[1, 2], [3, 4]]).astype(np.float32)
    input_dict7 = {
        "input": input7,
        "dim": None,
        "correction": 0,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    input8 = np.random.randn(4).astype(np.float32)
    input_dict8 = {
        "input": input8,
        "dim": 0,
        "correction": 0,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([[-1.0, -2.0], [-3.0, -4.0]]).astype(np.float32)
    input_dict9 = {
        "input": input9,
        "dim": 1,
        "correction": 0,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([1.0 + 1j, 2.0 + 2j, 3.0 + 3j, 4.0 + 4j]).astype(np.complex64)
    input_dict10 = {
        "input": input10,
        "dim": 0,
        "correction": 0,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
        
    return list_of_inputs
def std_inputs():
    list_of_inputs = []
    # Input 1: 1D float tensor, dim=None, correction=1, keepdim=False
    input1 = torch.randn(5).numpy()
    input_dict1 = {
        "input": input1,
        "dim": None,
        "correction": 1,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D float tensor, dim=(0,), correction=0, keepdim=True
    input2 = torch.randn(3, 4).numpy()
    input_dict2 = {
        "input": input2,
        "dim": (0,),
        "correction": 0,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 3D float tensor, dim=(1, 2), correction=2, keepdim=False
    input3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {
        "input": input3,
        "dim": (1, 2),
        "correction": 2,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 4D float tensor with negative values, dim=(0, 2), correction=1, keepdim=True
    input4 = (torch.randn(2, 3, 2, 3) * -1).numpy()
    input_dict4 = {
        "input": input4,
        "dim": (0, 2),
        "correction": 1,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 2D complex tensor, dim=(1,), correction=1, keepdim=False
    input5 = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict5 = {
        "input": input5,
        "dim": (1,),
        "correction": 1,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: 1D tensor, dim=None, correction=0, keepdim=True
    input6 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    input_dict6 = {
        "input": input6,
        "dim": None,
        "correction": 0,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def sub_inputs():
    list_of_inputs = []
    # Input 1: Basic subtraction with float tensors
    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randn(3, 4).numpy()
    alpha1 = 1.0
    input_dict1 = {"input": input1, "other": other1, "alpha": alpha1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Subtraction with integer tensors
    input2 = torch.randint(0, 10, (2, 2), dtype=torch.int32).numpy()
    other2 = torch.randint(0, 5, (2, 2), dtype=torch.int32).numpy()
    alpha2 = 1
    input_dict2 = {"input": input2, "other": other2, "alpha": alpha2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Subtraction with different shaped tensors (broadcasting)
    input3 = torch.randn(5, 1).numpy()
    other3 = torch.randn(1, 5).numpy()
    alpha3 = 1.0
    input_dict3 = {"input": input3, "other": other3, "alpha": alpha3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Subtraction with scalar other and alpha
    input4 = torch.randn(2, 3, 4).numpy()
    other4 = np.float64(2.0)
    alpha4 = 0.5
    input_dict4 = {"input": input4, "other": other4, "alpha": alpha4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Subtraction with complex tensors
    input5 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    other5 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    alpha5 = 1.0
    input_dict5 = {"input": input5, "other": other5, "alpha": alpha5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def tensordot_inputs():
    list_of_inputs = []
    a = np.random.rand(3, 4, 5).astype(np.float32)
    b = np.random.rand(4, 5, 6).astype(np.float32)
    dims = ([1, 2], [0, 1])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    a = np.random.rand(2, 3).astype(np.int32)
    b = np.random.rand(3, 4).astype(np.int32)
    dims = ([1], [0])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    a = np.random.rand(2, 3, 4).astype(np.float64)
    b = np.random.rand(4, 5).astype(np.float64)
    dims = ([2], [0])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    a = np.random.rand(2, 3).astype(np.complex64)
    b = np.random.rand(3, 4).astype(np.complex64)
    dims = ([1], [0])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.random.rand(5).astype(np.float32)
    b = np.random.rand(5).astype(np.float32)
    dims = ([0], [0])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    a = np.random.rand(2, 2, 2).astype(np.float32)
    b = np.random.rand(2, 2, 2).astype(np.float32)
    dims = ([0,1,2], [0,1,2])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    a = np.random.rand(2, 3, 4, 5).astype(np.float32)
    b = np.random.rand(5, 6, 7).astype(np.float32)
    dims = ([3], [0])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    a = np.random.rand(1).astype(np.float32)
    b = np.random.rand(1).astype(np.float32)
    dims = ([0], [0])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[1,2],[3,4]]).astype(np.float32)
    b = np.array([[5,6],[7,8]]).astype(np.float32)
    dims = (([0, 1]), ([0, 1]))
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def transpose_inputs():
    list_of_inputs = []
    # Input 1: 2D float tensor
    input1 = torch.randn(2, 3).numpy()
    input_dict1 = {
        "input": input1,
        "dim0": 0,
        "dim1": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 3D int tensor
    input2 = torch.randint(0, 10, (2, 3, 4)).numpy()
    input_dict2 = {
        "input": input2,
        "dim0": 1,
        "dim1": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 4D tensor with negative dimensions
    input3 = torch.randn(2, 3, 4, 5).numpy()
    input_dict3 = {
        "input": input3,
        "dim0": 0,
        "dim1": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 1D tensor (should still work, though effectively a no-op)
    input4 = torch.randn(5).numpy()
    input_dict4 = {
        "input": input4,
        "dim0": 0,
        "dim1": 0  # No effect since only one dimension
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Complex tensor
    input5 = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict5 = {
        "input": input5,
        "dim0": 0,
        "dim1": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6:  Another 3D tensor with different dimensions to test dim swap
    input6 = torch.randn(5, 2, 3).numpy()
    input_dict6 = {
        "input": input6,
        "dim0": 0,
        "dim1": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7:  Double tensor
    input7 = torch.randn(4, 2, dtype=torch.float64).numpy()
    input_dict7 = {
        "input": input7,
        "dim0": 0,
        "dim1": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def trapz_inputs():
    list_of_inputs = []
    y = np.array([1, 2, 3], dtype=np.float32)
    x = np.array([4, 6, 8], dtype=np.float32)
    dx = 1.0
    dim = 0
    input_dict = {"y": y, "x": x, "dx": dx, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    y = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    x = np.array([[4, 6, 8], [10, 12, 14]], dtype=np.float32)
    dx = 1.0
    dim = 1
    input_dict = {"y": y, "x": x, "dx": dx, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    y = np.array([1, 2, 3], dtype=np.float32)
    dx = 2.0
    dim = 0
    input_dict = {"y": y, "x": np.array([0, 1, 2], dtype=np.float32), "dx": dx, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    y = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    dx = 0.5
    dim = 1
    input_dict = {"y": y, "x": np.array([[0, 1, 2], [0, 1, 2]], dtype=np.float32), "dx": dx, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    y = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    dx = 0.5
    dim = 0
    input_dict = {"y": y, "x": np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32), "dx": dx, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def triangular_solve_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensors
    A = np.random.rand(3, 3).astype(np.float32)
    b = np.random.rand(3, 1).astype(np.float32)
    input_dict = {
        "A": A,
        "b": b,
        "upper": False,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Upper triangular, transposed, unitriangular
    A = np.triu(np.random.rand(5, 5)).astype(np.float64)
    b = np.random.rand(5, 3).astype(np.float64)
    input_dict = {
        "A": A,
        "b": b,
        "upper": True,
        "transpose": True,
        "unitriangular": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Complex tensors
    A = (np.random.rand(2, 2) + 1j * np.random.rand(2, 2)).astype(np.complex64)
    b = (np.random.rand(2, 1) + 1j * np.random.rand(2, 1)).astype(np.complex64)
    input_dict = {
        "A": A,
        "b": b,
        "upper": False,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Batched input, negative values
    A = np.tril(np.random.randn(2, 3, 3)).astype(np.float32)
    b = np.random.randn(2, 3, 2).astype(np.float32)
    input_dict = {
        "A": A,
        "b": b,
        "upper": False,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Higher dimension b - Correcting dimension issue - Second attempt
    A = np.random.rand(4, 4).astype(np.float32)
    b = np.random.rand(4, 2).astype(np.float32) # Corrected dimensions to be compatible with A
    input_dict = {
        "A": A,
        "b": b,
        "upper": False,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def unsqueeze_inputs():
    list_of_inputs = []
    # Input 1: 1D float tensor
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D int tensor, negative dim
    input2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict2 = {"input": input2, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 3D complex tensor
    input3 = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex64)
    input_dict3 = {"input": input3, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 4D float tensor, dim in the middle
    input4 = np.random.randn(2, 3, 4, 5).astype(np.float64)
    input_dict4 = {"input": input4, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Scalar tensor
    input5 = np.array(5, dtype=np.int64)
    input_dict5 = {"input": input5, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs
def var_mean_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensor, dim=None, correction=1, keepdim=False
    input_tensor = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "dim": None,
        "correction": 1,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Float tensor, dim=0, correction=0, keepdim=True
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "correction": 0,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: 2D tensor, dim=1, correction=2, keepdim=False
    input_tensor = np.random.randn(5, 5).astype(np.float64)
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "correction": 2,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: 1D tensor, dim=0, correction=1, keepdim=True
    input_tensor = np.random.randn(10).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "correction": 1,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5:  tensor with negative values, dim=(0, 2), correction=1, keepdim=True
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32) - 0.5
    input_dict = {
        "input": input_tensor,
        "dim": (0, 2),
        "correction": 1,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: 4D tensor, dim=(1, 2), correction=1, keepdim=False
    input_tensor = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "dim": (1, 2),
        "correction": 1,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Empty tensor, dim=None, correction=1, keepdim=False
    input_tensor = np.array([]).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "dim": None,
        "correction": 1,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def var_inputs():
    list_of_inputs = []
    input_np = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
    input_dict = {
        "input": input_np,
        "dim": 0,
        "correction": 1,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_np = np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float64)
    input_dict = {
        "input": input_np,
        "dim": 1,
        "correction": 1,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_np = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32)
    input_dict = {
        "input": input_np,
        "dim": 2,
        "correction": 0,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_np = np.array([[-1.0, 2.0], [3.0, -4.0]]).astype(np.float32)
    input_dict = {
        "input": input_np,
        "dim": 1,
        "correction": 1,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_np = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {
        "input": input_np,
        "dim": 1,
        "correction": 1,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_np = np.array([[1.0 + 1j, 2.0 - 2j], [3.0 + 3j, 4.0 - 4j]]).astype(np.complex64)
    input_dict = {
        "input": input_np,
        "dim": 1,
        "correction": 1,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def where_inputs():
    list_of_inputs = []
    # Case 1: Basic case with boolean condition and float tensors
    condition = np.array([[True, False], [False, True]])
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    other_tensor = np.array([[5.0, 6.0], [7.0, 8.0]])
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Integer tensors and condition
    condition = np.array([[True, False, True], [False, True, False]])
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]])
    other_tensor = np.array([[7, 8, 9], [10, 11, 12]])
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Different shapes (condition matches input/other)
    condition = np.array([True, False, True])
    input_tensor = np.array([1.0, 2.0, 3.0])
    other_tensor = np.array([4.0, 5.0, 6.0])
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: 3D tensors
    condition = np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    other_tensor = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Negative values
    condition = np.array([[True, False], [False, True]])
    input_tensor = np.array([[-1.0, 2.0], [3.0, -4.0]])
    other_tensor = np.array([[5.0, -6.0], [-7.0, 8.0]])
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 6: condition as a numpy array of integers (0 and 1)
    condition = np.array([[1, 0], [0, 1]])
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    other_tensor = np.array([[5.0, 6.0], [7.0, 8.0]])
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: Single element tensors
    condition = np.array(True)
    input_tensor = np.array(10.0)
    other_tensor = np.array(20.0)
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def zeros_inputs():
    generated_inputs = []
    input1 = {
        "size": (5,)
    }
    generated_inputs.append(copy.deepcopy(input1))
    input2 = {
        "size": (2, 3)
    }
    generated_inputs.append(copy.deepcopy(input2))
    input3 = {
        "size": (2, 3, 4)
    }
    generated_inputs.append(copy.deepcopy(input3))
    
    input4 = {
        "size": (1, 1, 1, 1)
    }
    generated_inputs.append(copy.deepcopy(input4))
    input5 = {
        "size": (6, 7)
    }
    generated_inputs.append(copy.deepcopy(input5))
    return generated_inputs
def addbmm_inputs():
    list_of_inputs = []
    # Input 1: Basic case with float tensors
    input_dict = {
        "input": torch.randn(4, 2).numpy(),
        "batch1": torch.randn(3, 4, 5).numpy(),
        "batch2": torch.randn(3, 5, 2).numpy(),
        "beta": 1.0,
        "alpha": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Integer tensors
    input_dict = {
        "input": torch.randint(0, 10, (4, 5)).numpy(),
        "batch1": torch.randint(0, 10, (2, 4, 3)).numpy(),
        "batch2": torch.randint(0, 10, (2, 3, 5)).numpy(),
        "beta": 2.0,
        "alpha": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Different beta and alpha, negative values
    input_dict = {
        "input": torch.randn(5, 3).numpy(),
        "batch1": torch.randn(4, 5, 6).numpy(),
        "batch2": torch.randn(4, 6, 3).numpy(),
        "beta": -0.5,
        "alpha": -2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Larger dimensions
    input_dict = {
        "input": torch.randn(6, 4).numpy(),
        "batch1": torch.randn(5, 6, 7).numpy(),
        "batch2": torch.randn(5, 7, 4).numpy(),
        "beta": 0.0,
        "alpha": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: beta = 0
    input_dict = {
        "input": torch.randn(4, 2).numpy(),
        "batch1": torch.randn(3, 4, 5).numpy(),
        "batch2": torch.randn(3, 5, 2).numpy(),
        "beta": 0.0,
        "alpha": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def cholesky_solve_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensors
    A = np.random.rand(3, 3)
    A = A @ A.T  # Make A positive definite
    B = np.random.rand(3, 2)
    input_dict = {"A": A, "B": B}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Different shape, float type
    A = np.random.rand(5, 5)
    A = A @ A.T
    B = np.random.rand(5, 1)
    input_dict = {"A": A, "B": B}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Double type
    A = np.random.rand(4, 4).astype(np.float64)
    A = A @ A.T
    B = np.random.rand(4, 3).astype(np.float64)
    input_dict = {"A": A, "B": B}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Different B shape and A is complex
    A = np.random.rand(2, 2) + 1j*np.random.rand(2, 2)
    A = A @ A.conj().T
    B = np.random.rand(2, 4) + 1j*np.random.rand(2, 4)
    input_dict = {"A": A, "B": B}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: A and B are complex with more dims
    A = (np.random.rand(2, 3, 3) + 1j*np.random.rand(2, 3, 3))
    A = np.einsum('aij,ajk->aik', A, A.conj().transpose(0, 2, 1))
    B = (np.random.rand(2, 3, 4) + 1j*np.random.rand(2, 3, 4))
    input_dict = {"A": A, "B": B}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Larger matrices
    A = np.random.rand(10, 10)
    A = A @ A.T
    B = np.random.rand(10, 5)
    input_dict = {"A": A, "B": B}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def lstm_cell_inputs():
    list_of_inputs = []
    input_size = 10
    hidden_size = 20
    # Input 1: Basic float input
    input_val = torch.randn(5, input_size).numpy()
    hx = torch.randn(5, hidden_size).numpy()
    cx = torch.randn(5, hidden_size).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size).numpy()
    bias_ih = torch.randn(4 * hidden_size).numpy()
    bias_hh = torch.randn(4 * hidden_size).numpy()
    input_dict = {
        "input": input_val,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Different batch size and sizes, no bias
    input_size = 5
    hidden_size = 10
    input_val = torch.randn(3, input_size).numpy()
    hx = torch.randn(3, hidden_size).numpy()
    cx = torch.randn(3, hidden_size).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size).numpy()
    bias_ih = np.zeros(4 * hidden_size).astype(np.float32)
    bias_hh = np.zeros(4 * hidden_size).astype(np.float32)
    input_dict = {
        "input": input_val,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": False,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Different dtype (double)
    input_size = 12
    hidden_size = 8
    input_val = torch.randn(2, input_size, dtype=torch.float64).numpy()
    hx = torch.randn(2, hidden_size, dtype=torch.float64).numpy()
    cx = torch.randn(2, hidden_size, dtype=torch.float64).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size, dtype=torch.float64).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size, dtype=torch.float64).numpy()
    bias_ih = torch.randn(4 * hidden_size, dtype=torch.float64).numpy()
    bias_hh = torch.randn(4 * hidden_size, dtype=torch.float64).numpy()
    input_dict = {
        "input": input_val,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Negative values, larger input size
    input_size = 30
    hidden_size = 15
    input_val = torch.randn(4, input_size) * -1.0
    input_val = input_val.numpy()
    hx = torch.randn(4, hidden_size) * -1.0
    hx = hx.numpy()
    cx = torch.randn(4, hidden_size) * -1.0
    cx = cx.numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size).numpy()
    bias_ih = torch.randn(4 * hidden_size).numpy()
    bias_hh = torch.randn(4 * hidden_size).numpy()
    input_dict = {
        "input": input_val,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Bias is False and input is of type Float16
    input_size = 8
    hidden_size = 16
    input_val = torch.randn(1, input_size, dtype=torch.float16).numpy()
    hx = torch.randn(1, hidden_size, dtype=torch.float16).numpy()
    cx = torch.randn(1, hidden_size, dtype=torch.float16).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size, dtype=torch.float16).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size, dtype=torch.float16).numpy()
    bias_ih = np.zeros(4 * hidden_size).astype(np.float16)
    bias_hh = np.zeros(4 * hidden_size).astype(np.float16)
    input_dict = {
        "input": input_val,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": False,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Bias is True and input is of type Float32
    input_size = 8
    hidden_size = 16
    input_val = torch.randn(1, input_size, dtype=torch.float32).numpy()
    hx = torch.randn(1, hidden_size, dtype=torch.float32).numpy()
    cx = torch.randn(1, hidden_size, dtype=torch.float32).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size, dtype=torch.float32).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size, dtype=torch.float32).numpy()
    bias_ih = torch.randn(4 * hidden_size, dtype=torch.float32).numpy()
    bias_hh = torch.randn(4 * hidden_size, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_val,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def matrix_rank_inputs():
    list_of_inputs = []
    # Input 1: Simple float matrix
    A = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "tol": 1e-8,
        "atol": 1e-8,
        "rtol": 1e-5,
        "hermitian": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Float matrix with zero row
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "tol": 1e-8,
        "atol": 1e-8,
        "rtol": 1e-5,
        "hermitian": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Complex Hermitian matrix
    A = np.array([[1, 1j], [-1j, 2]], dtype=np.complex64)
    input_dict = {
        "A": A,
        "tol": 1e-8,
        "atol": 1e-8,
        "rtol": 1e-5,
        "hermitian": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Float matrix with a small tolerance
    A = np.array([[1.0, 0.0001], [0.0001, 1.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "tol": 0.01,
        "atol": 1e-8,
        "rtol": 1e-5,
        "hermitian": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: A matrix with non-default atol and rtol
    A = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "tol": 1e-8,
        "atol": 0.1,
        "rtol": 0.1,
        "hermitian": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: A matrix with negative values
    A = np.array([[1.0, -2.0], [-2.0, 4.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "tol": 1e-8,
        "atol": 1e-8,
        "rtol": 1e-5,
        "hermitian": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def max_pool3d_inputs():
    list_of_inputs = []
    input1 = torch.randn(1, 3, 10, 10, 10).numpy()
    kernel_size1 = 2
    stride1 = 2
    padding1 = 0
    dilation1 = 1
    ceil_mode1 = False
    return_indices1 = False
    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "dilation": dilation1,
        "ceil_mode": ceil_mode1,
        "return_indices": return_indices1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(2, 4, 8, 8, 8).numpy()
    kernel_size2 = (2, 2, 2)
    stride2 = (2, 2, 2)
    padding2 = (1, 1, 1)
    dilation2 = (1, 1, 1)
    ceil_mode2 = True
    return_indices2 = False
    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "dilation": dilation2,
        "ceil_mode": ceil_mode2,
        "return_indices": return_indices2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(1, 1, 5, 5, 5).numpy()
    kernel_size3 = 3
    stride3 = 1
    padding3 = 1
    dilation3 = 1
    ceil_mode3 = False
    return_indices3 = True
    input_dict3 = {
        "input": input3,
        "kernel_size": kernel_size3,
        "stride": stride3,
        "padding": padding3,
        "dilation": dilation3,
        "ceil_mode": ceil_mode3,
        "return_indices": return_indices3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randint(0, 10, (1, 2, 7, 7, 7)).float().numpy()
    kernel_size4 = (2, 2, 2)
    stride4 = (1, 1, 1)
    padding4 = (0, 0, 0)
    dilation4 = (1, 1, 1)
    ceil_mode4 = False
    return_indices4 = False
    input_dict4 = {
        "input": input4,
        "kernel_size": kernel_size4,
        "stride": stride4,
        "padding": padding4,
        "dilation": dilation4,
        "ceil_mode": ceil_mode4,
        "return_indices": return_indices4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(1, 1, 12, 12, 12).numpy()
    kernel_size5 = (3, 3, 3)
    stride5 = (2, 2, 2)
    padding5 = (1, 1, 1)
    dilation5 = (1, 1, 1)
    ceil_mode5 = True
    return_indices5 = False
    input_dict5 = {
        "input": input5,
        "kernel_size": kernel_size5,
        "stride": stride5,
        "padding": padding5,
        "dilation": dilation5,
        "ceil_mode": ceil_mode5,
        "return_indices": return_indices5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(1, 1, 9, 9, 9).numpy()
    kernel_size6 = 3
    stride6 = 2
    padding6 = 1
    dilation6 = 1
    ceil_mode6 = False
    return_indices6 = True
    input_dict6 = {
        "input": input6,
        "kernel_size": kernel_size6,
        "stride": stride6,
        "padding": padding6,
        "dilation": dilation6,
        "ceil_mode": ceil_mode6,
        "return_indices": return_indices6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = torch.randn(1, 3, 10, 10, 10).double().numpy()
    kernel_size7 = 2
    stride7 = 2
    padding7 = 0
    dilation7 = 1
    ceil_mode7 = False
    return_indices7 = False
    input_dict7 = {
        "input": input7,
        "kernel_size": kernel_size7,
        "stride": stride7,
        "padding": padding7,
        "dilation": dilation7,
        "ceil_mode": ceil_mode7,
        "return_indices": return_indices7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.randn(1, 3, 10, 10, 10).int().numpy()
    kernel_size8 = 2
    stride8 = 2
    padding8 = 0
    dilation8 = 1
    ceil_mode8 = False
    return_indices8 = False
    input_dict8 = {
        "input": input8,
        "kernel_size": kernel_size8,
        "stride": stride8,
        "padding": padding8,
        "dilation": dilation8,
        "ceil_mode": ceil_mode8,
        "return_indices": return_indices8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    return list_of_inputs
def max_unpool2d_inputs():
    list_of_inputs = []
    # Test case 1: Basic case with specified output size
    input1 = torch.randn(1, 1, 2, 2).numpy()
    indices1 = torch.randint(0, 4, (1, 1, 2, 2)).numpy()
    input_dict1 = {
        "input": input1,
        "indices": indices1,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "output_size": (4, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Test case 2: Different kernel size and stride
    input2 = torch.randn(1, 1, 3, 3).numpy()
    indices2 = torch.randint(0, 9, (1, 1, 3, 3)).numpy()
    input_dict2 = {
        "input": input2,
        "indices": indices2,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Test case 3: Multiple channels
    input3 = torch.randn(1, 2, 2, 2).numpy()
    indices3 = torch.randint(0, 16, (1, 2, 2, 2)).numpy()
    input_dict3 = {
        "input": input3,
        "indices": indices3,
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Test case 4: Batch size > 1
    input4 = torch.randn(2, 1, 2, 2).numpy()
    indices4 = torch.randint(0, 16, (2, 1, 2, 2)).numpy()
    input_dict4 = {
        "input": input4,
        "indices": indices4,
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Test case 5: Different padding
    input5 = torch.randn(1, 1, 2, 2).numpy()
    indices5 = torch.randint(0, 9, (1, 1, 2, 2)).numpy()
    input_dict5 = {
        "input": input5,
        "indices": indices5,
        "kernel_size": 3,
        "stride": 1,
        "padding": 2,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs
def max_unpool2d_inputs():
    list_of_inputs = []
    # Input 1: Basic case with specified output size
    input1 = torch.randn(1, 1, 2, 2).numpy()
    indices1 = torch.randint(0, 4, (1, 1, 2, 2)).numpy()
    kernel_size1 = 2
    stride1 = 2
    padding1 = 0
    output_size1 = (5, 5)
    input_dict1 = {
        "input": input1,
        "indices": indices1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "output_size": output_size1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Different kernel size and stride
    input2 = torch.randn(1, 3, 3, 3).numpy()
    indices2 = torch.randint(0, 9, (1, 3, 3, 3)).numpy()
    kernel_size2 = 3
    stride2 = 1
    padding2 = 1
    output_size2 = None
    input_dict2 = {
        "input": input2,
        "indices": indices2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "output_size": output_size2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Non-square kernel size
    input3 = torch.randn(1, 1, 3, 4).numpy()
    indices3 = torch.randint(0, 12, (1, 1, 3, 4)).numpy()
    kernel_size3 = (3, 2)
    stride3 = (2, 1)
    padding3 = (1, 0)
    output_size3 = None
    input_dict3 = {
        "input": input3,
        "indices": indices3,
        "kernel_size": kernel_size3,
        "stride": stride3,
        "padding": padding3,
        "output_size": output_size3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Larger input with batch size > 1
    input4 = torch.randn(2, 4, 5, 5).numpy()
    indices4 = torch.randint(0, 25, (2, 4, 5, 5)).numpy()
    kernel_size4 = 2
    stride4 = 2
    padding4 = 0
    output_size4 = (10, 10)
    input_dict4 = {
        "input": input4,
        "indices": indices4,
        "kernel_size": kernel_size4,
        "stride": stride4,
        "padding": padding4,
        "output_size": output_size4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Output size is None
    input5 = torch.randn(1, 2, 4, 4).numpy()
    indices5 = torch.randint(0, 16, (1, 2, 4, 4)).numpy()
    kernel_size5 = 4
    stride5 = 4
    padding5 = 0
    output_size5 = None
    input_dict5 = {
        "input": input5,
        "indices": indices5,
        "kernel_size": kernel_size5,
        "stride": stride5,
        "padding": padding5,
        "output_size": output_size5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Single int for kernel_size, stride, padding
    input6 = torch.randn(1, 1, 2, 2).numpy()
    indices6 = torch.randint(0, 4, (1, 1, 2, 2)).numpy()
    kernel_size6 = 2
    stride6 = 1
    padding6 = 0
    output_size6 = None
    input_dict6 = {
        "input": input6,
        "indices": indices6,
        "kernel_size": kernel_size6,
        "stride": stride6,
        "padding": padding6,
        "output_size": output_size6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Tuple for kernel_size, stride, padding
    input7 = torch.randn(1, 1, 2, 2).numpy()
    indices7 = torch.randint(0, 4, (1, 1, 2, 2)).numpy()
    kernel_size7 = (2, 2)
    stride7 = (1, 1)
    padding7 = (0, 0)
    output_size7 = None
    input_dict7 = {
        "input": input7,
        "indices": indices7,
        "kernel_size": kernel_size7,
        "stride": stride7,
        "padding": padding7,
        "output_size": output_size7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def pad_inputs():
    list_of_inputs = []
    # Input 1: 2D float tensor, constant mode
    input1 = torch.randn(2, 3).numpy()
    pad1 = (1, 1, 2, 2)  # left, right, top, bottom
    mode1 = 'constant'
    value1 = 0.5
    input_dict1 = {"input": input1, "pad": pad1, "mode": mode1, "value": value1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 3D int tensor, reflect mode
    input2 = torch.randint(0, 10, (1, 4, 5)).numpy()
    pad2 = (2, 2, 1, 1, 0, 0)  # last dim, second last dim, first dim
    mode2 = 'constant' # Changed from reflect to constant
    value2 = 0.0  # Value is ignored for reflect mode
    input_dict2 = {"input": input2, "pad": pad2, "mode": mode2, "value": value2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 4D float tensor, replicate mode
    input3 = torch.randn(1, 2, 3, 4).numpy()
    pad3 = (0, 1, 2, 0, 0, 0, 1, 1)  # W, H, D, N  (last to first dimension)
    mode3 = 'constant' # Changed from replicate to constant
    value3 = 0.0  # Value is ignored for replicate mode
    input_dict3 = {"input": input3, "pad": pad3, "mode": mode3, "value": value3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 2D float tensor, circular mode
    input4 = torch.randn(3, 3).numpy()
    pad4 = (1, 1, 1, 1)
    mode4 = 'constant' # Changed from circular to constant
    value4 = 0.0  # Value is ignored for circular mode
    input_dict4 = {"input": input4, "pad": pad4, "mode": mode4, "value": value4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 1D long tensor, constant mode, different value
    input5 = torch.randint(0, 10, (5,)).long().numpy()
    pad5 = (2, 3)
    mode5 = 'constant'
    value5 = 5.0
    input_dict5 = {"input": input5, "pad": pad5, "mode": mode5, "value": value5}
    #list_of_inputs.append(copy.deepcopy(input_dict5)) #Removed because 1D tensors are not accepted
    # Input 6: 5D float tensor, constant mode
    input6 = torch.randn(1, 1, 2, 2, 2).numpy()
    pad6 = (1, 0, 0, 1, 1, 0, 0, 1, 0, 1)
    mode6 = 'constant'
    value6 = -1.0
    input_dict6 = {"input": input6, "pad": pad6, "mode": mode6, "value": value6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs
def rand_like_inputs():
    generated_inputs = []
    input1 = np.random.randn(3, 4).astype(np.float32)
    generated_inputs.append({"input": input1})
    input2 = np.random.randint(0, 10, size=(2, 2), dtype=np.int64)
    generated_inputs.append({"input": input2})
    input3 = np.random.rand(1, 5, 5).astype(np.float64)
    generated_inputs.append({"input": input3})
    input4 = np.array([-1, 0, 1]).astype(np.int32)
    generated_inputs.append({"input": input4})
    input5 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    generated_inputs.append({"input": input5})
    input6 = np.array([1+1j, 2+2j, 3+3j]).astype(np.complex128)
    generated_inputs.append({"input": input6})
    
    input7 = np.zeros((2,3)).astype(np.bool_)
    generated_inputs.append({"input": input7})
    return generated_inputs
def repeat_interleave_inputs():
    list_of_inputs = []
    # Test case 1: Basic 1D tensor with scalar repeats
    input_tensor = torch.tensor([1, 2, 3])
    repeats = 2
    input_dict = {"input": input_tensor, "repeats": repeats}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 2: 2D tensor with scalar repeats
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    repeats = 3
    input_dict = {"input": input_tensor, "repeats": repeats}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 3: 1D tensor with tensor repeats
    input_tensor = torch.tensor([1, 2, 3])
    repeats = torch.tensor([1, 2, 3])
    input_dict = {"input": input_tensor, "repeats": repeats}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 4: 2D tensor with tensor repeats and dim specified
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    repeats = torch.tensor([2, 1])
    input_dict = {"input": input_tensor, "repeats": repeats, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 5: 3D tensor with scalar repeats
    input_tensor = torch.randn(2, 3, 4)
    repeats = 2
    input_dict = {"input": input_tensor, "repeats": repeats}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def sparse_coo_tensor_inputs():
    list_of_inputs = []
    # Input 1: Basic example with int values
    indices = np.array([[0, 1], [1, 2], [2, 0]]).T
    values = np.array([1, 2, 3], dtype=np.int64)
    size = (3, 3)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": np.int64,
        "requires_grad": False,
        "check_invariants": True,
        "is_coalesced": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Float values, different size
    indices = np.array([[0, 0], [1, 1], [2, 2], [3, 3]]).T
    values = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    size = (4, 4)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": np.float32,
        "requires_grad": False,
        "check_invariants": False,
        "is_coalesced": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: 3D tensor
    indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]).T
    values = np.array([4, 5, 6], dtype=np.int32)
    size = (3, 3, 3)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": np.int32,
        "requires_grad": False,
        "check_invariants": True,
        "is_coalesced": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Negative values and different dtype
    indices = np.array([[0, 1], [1, 0]]).T
    values = np.array([-1.5, 2.5], dtype=np.float64)
    size = (2, 2)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": np.float64,
        "requires_grad": False,
        "check_invariants": False,
        "is_coalesced": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Complex values
    indices = np.array([[0, 0], [1, 1]]).T
    values = np.array([1 + 1j, 2 - 2j], dtype=np.complex64)
    size = (2, 2)
    input_dict = {
        "indices": indices,
        "values": values,
        "size": size,
        "dtype": np.complex64,
        "requires_grad": False,
        "check_invariants": True,
        "is_coalesced": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def unsqueeze_inputs():
    list_of_inputs = []
    # Input 1: 1D float tensor
    input_1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict_1 = {"input": input_1, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    # Input 2: 2D int tensor
    input_2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict_2 = {"input": input_2, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    # Input 3: 3D complex tensor
    input_3 = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex64)
    input_dict_3 = {"input": input_3, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    # Input 4: 4D bool tensor
    input_4 = np.array([[[[True, False], [False, True]]]], dtype=np.bool_)
    input_dict_4 = {"input": input_4, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    # Input 5: 2D float tensor with negative dimension
    input_5 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    input_dict_5 = {"input": input_5, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    # Input 6: 0D integer tensor
    input_6 = np.array(5, dtype=np.int64)
    input_dict_6 = {"input": input_6, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: 3D float tensor, dim = 1
    input_7 = np.random.rand(2,3,4).astype(np.float32)
    input_dict_7 = {"input": input_7, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    return list_of_inputs
def matrix_exp_inputs():
    list_of_inputs = []
    # Input 1: Simple 2x2 float matrix
    A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: 3x3 matrix with negative values
    A = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: 4x4 complex matrix
    A = np.array([[1+1j, 0, 0, 0], [0, 2+2j, 0, 0], [0, 0, 3+3j, 0], [0, 0, 0, 4+4j]], dtype=np.complex64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Stacked matrices (batch of 2)
    A = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]], dtype=np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Larger matrix (5x5)
    A = np.random.rand(5, 5).astype(np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Another complex matrix
    A = np.array([[1j, 1], [-1, -1j]], dtype=np.complex128)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 7: 2x2 matrix with high values
    A = np.array([[10.0, 5.0], [2.0, 8.0]], dtype=np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
from typing import Dict, List, Optional
def annotate_inputs():
    list_of_inputs = []
    # Example 1: Empty dictionary
    input_dict = {
        "the_type": "Dict[str, int]",
        "the_value": {}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 2: Optional type with a value
    input_dict = {
        "the_type": "Optional[int]",
        "the_value": 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 3: Empty list
    input_dict = {
        "the_type": "List[float]",
        "the_value": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 4: List of tensors
    input_dict = {
        "the_type": "List[torch.Tensor]",
        "the_value": [torch.randn(2, 3).numpy(), torch.randn(4, 5).numpy()]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 5: Dictionary of tensors
    input_dict = {
        "the_type": "Dict[str, torch.Tensor]",
        "the_value": {"a": torch.randn(2, 2).numpy(), "b": torch.randn(3, 3).numpy()}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 6: Optional tensor
    input_dict = {
        "the_type": "Optional[torch.Tensor]",
        "the_value": torch.randn(1, 1).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 7: None value for optional tensor
    input_dict = {
        "the_type": "Optional[torch.Tensor]",
        "the_value": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 8: List of ints
    input_dict = {
        "the_type": "List[int]",
        "the_value": [1, 2, 3, 4, 5]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 9: Dict of string and float
    input_dict = {
        "the_type": "Dict[str, float]",
        "the_value": {"key1": 1.0, "key2": 2.0}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def meshgrid_inputs():
    list_of_inputs = []
    # Input 1: Basic case with two 1D tensors (int) and ij indexing
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([4, 5, 6, 7], dtype=np.int64)
    input_dict = {
        "tensors": [x, y],
        "indexing": 'ij'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Two 1D tensors (float) and xy indexing
    x = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
    y = np.array([5.5, 6.6, 7.7], dtype=np.float64)
    input_dict = {
        "tensors": [x, y],
        "indexing": 'xy'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Three 1D tensors (int, float, int) and ij indexing
    x = np.array([1, 2], dtype=np.float64)
    y = np.array([3.3, 4.4, 5.5], dtype=np.float64)
    z = np.array([6, 7, 8, 9], dtype=np.float64)
    input_dict = {
        "tensors": [x, y, z],
        "indexing": 'ij'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Two 1D tensors with negative values (int) and xy indexing
    x = np.array([-1, -2, -3], dtype=np.int64)
    y = np.array([-4, -5, -6, -7], dtype=np.int64)
    input_dict = {
        "tensors": [x, y],
        "indexing": 'xy'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Three 1D tensors with a scalar and ij indexing
    x = np.array([1], dtype=np.int64)
    y = np.array([2, 3, 4], dtype=np.int64)
    z = np.array([5, 6], dtype=np.int64)
    input_dict = {
        "tensors": [x, y, z],
        "indexing": 'ij'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Two 1D tensors with a large range and xy indexing
    x = np.linspace(-10, 10, 5, dtype=np.float64)
    y = np.linspace(-5, 5, 3, dtype=np.float64)
    input_dict = {
        "tensors": [x, y],
        "indexing": 'xy'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def tan__inputs():
    list_of_inputs = []
    input1 = np.random.randn(3, 4).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.random.randn(2, 2, 2).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input4 = np.random.randn(1, 5, 5, 5).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([1.5 + 1j, 2.5 - 2j]).astype(np.complex64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([[-1.0, 2.0], [-3.0, 4.0]]).astype(np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([0.0]).astype(np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def softmax_inputs():
    list_of_inputs = []
    # Case 1: 1D tensor, float32
    input_1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict_1 = {"input": input_1, "dim": 0, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    # Case 2: 2D tensor, float64, negative values
    input_2 = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]], dtype=np.float64)
    input_dict_2 = {"input": input_2, "dim": 1, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    # Case 3: 3D tensor, float32, dim=2
    input_3 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict_3 = {"input": input_3, "dim": 2, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    # Case 4: 2D tensor, float16, dim=0, specify dtype
    input_4 = np.array([[0.5, 0.2], [0.8, 0.1]], dtype=np.float16)
    input_dict_4 = {"input": input_4, "dim": 0, "dtype": torch.float16}
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    # Case 5: 4D tensor, float64, dim=3
    input_5 = np.random.randn(1, 2, 3, 4).astype(np.float64)
    input_dict_5 = {"input": input_5, "dim": 3, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    # Case 6: 2D tensor, mixed positive and negative
    input_6 = np.array([[-1.5, 2.5], [0.0, -0.5]], dtype=np.float32)
    input_dict_6 = {"input": input_6, "dim": 1, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Case 7: 1D tensor with a single element
    input_7 = np.array([5.0], dtype=np.float32)
    input_dict_7 = {"input": input_7, "dim": 0, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    return list_of_inputs
def celu_inputs():
    list_of_inputs = []
    input1 = np.random.randn(2).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.random.randn(2, 3).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "alpha": 0.5,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.random.randn(1, 2, 3).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "alpha": 2.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.random.randn(1, 2, 3, 4).astype(np.float64)
    input_dict4 = {
        "input": input4,
        "alpha": 0.75,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.random.randn(5).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "alpha": 1.25,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]).astype(np.float32)
    input_dict6 = {
        "input": input6,
        "alpha": 1.5,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.random.randn(2, 2, 2, 2, 2).astype(np.float64)
    input_dict7 = {
        "input": input7,
        "alpha": 0.25,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def moveaxis_inputs():
    list_of_inputs = []
    # Case 1: Simple 3D tensor, moving axis 1 to 0
    t = torch.randn(3, 2, 1).numpy()
    input_dict = {"input": t, "source": 1, "destination": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: 4D tensor, moving axis 0 to 2
    t = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {"input": t, "source": 0, "destination": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: 3D tensor, moving axis 2 to -1 (same as 2)
    t = torch.randn(3, 2, 1).numpy()
    input_dict = {"input": t, "source": 2, "destination": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: 5D tensor, move axis -1 to axis 0
    t = torch.randn(2, 3, 4, 5, 6).numpy()
    input_dict = {"input": t, "source": -1, "destination": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: 2D tensor, move axis 0 to axis 1
    t = torch.randn(2, 3).numpy()
    input_dict = {"input": t, "source": 0, "destination": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 6: Float64 tensor, moving axis 1 to 0
    t = torch.randn(3, 2, 1, dtype=torch.float64).numpy()
    input_dict = {"input": t, "source": 1, "destination": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: Int tensor
    t = torch.randint(0, 10, (3, 2, 1)).numpy()
    input_dict = {"input": t, "source": 1, "destination": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 8: Complex tensor
    t = torch.randn(3, 2, 1, dtype=torch.complex64).numpy()
    input_dict = {"input": t, "source": 1, "destination": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def clear_autocast_cache_inputs():
    list_of_inputs = []
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def set_autocast_cpu_enabled_inputs():
    list_of_inputs = []
    input_dict = {
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "enabled": bool(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "enabled": bool(0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def native_channel_shuffle_inputs():
    list_of_inputs = []
    # Input 1: Basic 4D tensor, groups = 2
    input1 = torch.randn(1, 4, 10, 10).numpy()
    groups1 = 2
    input_dict1 = {"input": input1, "groups": groups1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 4D tensor, groups = 4 (equal to number of channels)
    input2 = torch.randn(1, 4, 5, 5).numpy()
    groups2 = 4
    input_dict2 = {"input": input2, "groups": groups2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 4D tensor, groups = 1 (no shuffle)
    input3 = torch.randn(1, 8, 7, 7).numpy()
    groups3 = 1
    input_dict3 = {"input": input3, "groups": groups3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: 4D tensor, different batch size
    input4 = torch.randn(2, 6, 8, 8).numpy()
    groups4 = 3
    input_dict4 = {"input": input4, "groups": groups4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Different spatial dimensions
    input5 = torch.randn(1, 12, 15, 20).numpy()
    groups5 = 4
    input_dict5 = {"input": input5, "groups": groups5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6:  3D tensor
    input6 = torch.randn(3, 6, 6).numpy()
    groups6 = 2
    input_dict6 = {"input": input6, "groups": groups6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Float64 tensor
    input7 = torch.randn(1, 4, 10, 10, dtype=torch.float64).numpy()
    groups7 = 2
    input_dict7 = {"input": input7, "groups": groups7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
        
    return list_of_inputs
def rfftfreq_inputs():
    list_of_inputs = []
    input1 = {
        "n": np.int32(5),
        "d": np.float64(1.0),
        "device": None
    }
    list_of_inputs.append(copy.deepcopy(input1))
    input2 = {
        "n": np.int64(4),
        "d": np.float32(0.5),
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input2))
    input3 = {
        "n": np.int32(10),
        "d": np.float64(2.0),
        "device": None
    }
    list_of_inputs.append(copy.deepcopy(input3))
    
    input4 = {
        "n": np.int64(7),
        "d": np.float32(1.0),
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input4))
    
    input5 = {
        "n": np.int32(16),
        "d": np.float64(0.1),
        "device": None
    }
    list_of_inputs.append(copy.deepcopy(input5))
    input6 = {
        "n": np.int64(32),
        "d": np.float32(0.25),
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input6))
    input7 = {
        "n": np.int32(64),
        "d": np.float64(0.05),
        "device": None
    }
    list_of_inputs.append(copy.deepcopy(input7))
    return list_of_inputs
def torch_empty_inputs():
    list_of_inputs = []
    # Example 1: Basic 2D tensor
    input_dict = {
        "size": (2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 2: 3D tensor
    input_dict = {
        "size": (4, 5, 6)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 3: 1D tensor
    input_dict = {
        "size": (10,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 5: Large tensor
    input_dict = {
        "size": (100, 100)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 6: Different dtypes
    input_dict = {
        "size": (2, 3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 7: Different layouts
    input_dict = {
        "size": (2, 3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def eig_inputs():
    list_of_inputs = []
    # Input 1: Simple 2x2 float matrix, eigenvectors=True
    input1 = np.random.rand(2, 2).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "eigenvectors": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Simple 3x3 float matrix, eigenvectors=False
    input2 = np.random.rand(3, 3).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "eigenvectors": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Complex matrix, eigenvectors=True
    input3 = (np.random.rand(2, 2) + 1j * np.random.rand(2, 2)).astype(np.complex64)
    input_dict3 = {
        "input": input3,
        "eigenvectors": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Larger matrix, eigenvectors=False
    input4 = np.random.rand(5, 5).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "eigenvectors": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Matrix with negative values, eigenvectors=True
    input5 = (np.random.rand(3, 3) - 0.5).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "eigenvectors": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Complex matrix, eigenvectors=False
    input6 = (np.random.rand(4, 4) + 1j * np.random.rand(4, 4)).astype(np.complex128)
    input_dict6 = {
        "input": input6,
        "eigenvectors": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Non-square matrix (should raise an error), eigenvectors=True
    input7 = np.random.rand(2, 3).astype(np.float32)
    input_dict7 = {
        "input": input7,
        "eigenvectors": True
    }
    # This one is skipped because the error case is to be handeled in testing
    #list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def is_floating_point_inputs():
    list_of_inputs = []
    # Input 1: Float tensor
    input1 = np.random.randn(3, 4).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Double tensor
    input2 = np.random.randn(2, 2, 2).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Half tensor
    input3 = np.random.randn(5).astype(np.float16)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Int tensor (should return False)
    input4 = np.random.randint(0, 10, size=(1, 5)).astype(np.int64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Complex Float Tensor
    input5 = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(np.complex64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Scalar Float
    input6 = np.float32(3.14)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: Zero-dimensional float
    input7 = np.array(5.0).astype(np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def svdvals_inputs():
    list_of_inputs = []
    A = torch.randn(5, 3).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(3, 5).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(2, 5, 3).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(2, 3, 5).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(1, 2, 5, 3).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(5, 3, dtype=torch.float64).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(3, 5, dtype=torch.float64).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(5, 3, dtype=torch.complex64).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(3, 5, dtype=torch.complex128).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = (torch.randn(5, 3) - 1).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = np.random.rand(4, 4)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = np.random.rand(3, 2, 4, 4)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def jit_error_inputs():
    list_of_inputs = []
    input_dict_1 = {
        "msg": "This is a test error message."
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    input_dict_2 = {
        "msg": "Another error occurred during compilation."
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    input_dict_3 = {
        "msg": "Type mismatch in compiled code."
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    input_dict_4 = {
        "msg": "Index out of bounds."
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    input_dict_5 = {
        "msg": "Division by zero error."
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    input_dict_6 = {
        "msg": "Unexpected keyword argument."
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    input_dict_7 = {
        "msg": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    return list_of_inputs
def Tanhshrink_inputs():
    generated_inputs = []
    input1 = np.random.randn(2).astype(np.float32)
    generated_inputs.append({"input": input1})
    input2 = np.random.randn(2, 3).astype(np.float64)
    generated_inputs.append({"input": input2})
    input3 = np.random.randn(2, 3, 4).astype(np.float32)
    generated_inputs.append({"input": input3})
    input4 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]).astype(np.float32)
    generated_inputs.append({"input": input4})
    
    input5 = np.random.randn(1, 1, 5, 5).astype(np.float64)
    generated_inputs.append({"input": input5})
    return generated_inputs
def erfc_inputs():
    list_of_inputs = []
    input1 = np.random.randn(5).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.random.randn(2, 3).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([-1.0, 0.0, 1.0]).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.randn(2, 2, 2).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.random.randn(1, 5, 5).astype(np.float16)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([0.5, 1.5, 2.5, 3.5]).astype(np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([[-0.5, -1.5], [-2.5, -3.5]]).astype(np.float64)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def arcsinh_inputs():
    list_of_inputs = []
    # Input 1: Float tensor, 1D
    input1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Float tensor, 2D
    input2 = np.array([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Float tensor, multi-dimensional with a zero
    input4 = np.array([[[0.0, 1.0], [-1.0, 0.0]], [[2.0, -2.0], [-3.0, 3.0]]], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 4: Float tensor, larger values
    input5 = np.array([[-10.0, 10.0], [5.0, -5.0]], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 5: Float tensor, scalar
    input6 = np.array(0.7, dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def arctanh_inputs():
    list_of_inputs = []
    input1 = np.array([0.1, 0.2, 0.3]).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([-0.4, -0.5, -0.6]).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[0.7, 0.8], [0.9, 0.0]]).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([[-0.1, 0.2, -0.3], [0.4, -0.5, 0.6]]).astype(np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([0.5]).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]).astype(np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([0.99]).astype(np.float64)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def solve_ex_inputs():
    list_of_inputs = []
    # Case 1: Basic float tensors
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B = np.array([[5.0], [11.0]], dtype=np.float32)
    input_dict = {"A": A, "B": B, "check_errors": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Complex tensors
    A = np.array([[1.0 + 1j, 2.0 - 1j], [3.0 + 0j, 4.0 + 2j]], dtype=np.complex64)
    B = np.array([[5.0 + 0j], [11.0 - 1j]], dtype=np.complex64)
    input_dict = {"A": A, "B": B, "check_errors": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Different shapes, solvable system
    A = np.array([[1.0, 2.0, 3.0], [2.0, 5.0, 2.0], [1.0, 0.0, 8.0]], dtype=np.float32)
    B = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    input_dict = {"A": A, "B": B, "check_errors": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: Multiple right-hand sides
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B = np.array([[5.0, 6.0], [11.0, 12.0]], dtype=np.float32)
    input_dict = {"A": A, "B": B, "check_errors": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Negative values
    A = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    B = np.array([[5.0], [-11.0]], dtype=np.float32)
    input_dict = {"A": A, "B": B, "check_errors": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def sin__inputs():
    list_of_inputs = []
    input1 = np.random.randn(3, 4).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.random.randint(-5, 5, size=(2, 2), dtype=np.int32).astype(np.float32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.random.randn(1, 5, 5).astype(np.float64)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi]).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.random.randn(1).astype(np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def is_scripting_inputs():
    list_of_inputs = []
    
    list_of_inputs.append({"input": np.array([1])})
    
    return list_of_inputs
def ldexp_inputs():
    list_of_inputs = []
    input_tensor = torch.tensor([1.0]).numpy()
    other_tensor = torch.tensor([1]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.tensor([1.0]).numpy()
    other_tensor = torch.tensor([1, 2, 3, 4]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other_tensor = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    other_tensor = torch.tensor([[1, 2], [3, 4]]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    other_tensor = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other_tensor = torch.tensor([-1, -2, -3]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.tensor([1.5, 2.5, 3.5]).numpy()
    other_tensor = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(2, 3).numpy()
    other_tensor = torch.randint(1, 5, (2, 3)).numpy()
    input_dict = {"input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def adjoint_inputs():
    generated_inputs = []
    # Real valued tensor, 2D
    input1 = torch.randn(2, 3).numpy()
    generated_inputs.append({"input": input1})
    # Real valued tensor, 3D
    input2 = torch.randn(2, 3, 4).numpy()
    generated_inputs.append({"input": input2})
    # Real valued tensor, 4D
    input3 = torch.randn(2, 3, 4, 5).numpy()
    generated_inputs.append({"input": input3})
    # Complex valued tensor, 2D
    input4 = torch.randn(2, 3, dtype=torch.float)
    input4 = torch.complex(input4, torch.randn(2, 3, dtype=torch.float)).numpy()
    generated_inputs.append({"input": input4})
    # Complex valued tensor, 3D
    input5 = torch.randn(2, 3, 4, dtype=torch.float)
    input5 = torch.complex(input5, torch.randn(2, 3, 4, dtype=torch.float)).numpy()
    generated_inputs.append({"input": input5})
    # Integer tensor, 2D
    input6 = torch.randint(-10, 10, (2, 3)).numpy()
    generated_inputs.append({"input": input6})
    # Integer tensor, 3D
    input7 = torch.randint(-10, 10, (2, 3, 4)).numpy()
    generated_inputs.append({"input": input7})
    # Complex valued tensor, 4D
    input8 = torch.randn(2, 3, 4, 5, dtype=torch.float)
    input8 = torch.complex(input8, torch.randn(2, 3, 4, 5, dtype=torch.float)).numpy()
    generated_inputs.append({"input": input8})
    # Real valued tensor with negative values
    input9 = torch.randn(5, 5).numpy() * -1
    generated_inputs.append({"input": input9})
    return generated_inputs
def acos_inputs():
    list_of_inputs = []
    input1 = np.array([0.5, -0.2, 0.9, -0.8]).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[0.1, 0.2], [0.3, 0.4]]).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[-1, -0.5], [0, 0.5], [1, 0]]).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([[[0.2, 0.3], [0.4, 0.5]], [[0.6, 0.7], [0.8, 0.9]]]).astype(np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([1, -1, 0]).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([0.1 + 0.1j, 0.2 - 0.2j, 0.3 + 0.3j]).astype(np.complex64)
    bounded_input6 = np.clip(input6.real, -1, 1) + 1j * np.clip(input6.imag, -1, 1)
    input_dict6 = {"input": bounded_input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]).astype(np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    input8 = np.array([-0.9, -0.5, 0, 0.5, 0.9]).astype(np.float64)
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    return list_of_inputs
def set_anomaly_enabled_inputs():
    list_of_inputs = []
    
    input1 = {'mode': True}
    list_of_inputs.append(copy.deepcopy(input1))
    
    input2 = {'mode': False}
    list_of_inputs.append(copy.deepcopy(input2))
    
    input3 = {'mode': np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input3))
    
    input4 = {'mode': np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input4))
    
    input5 = {'mode': np.array(True)}
    list_of_inputs.append(copy.deepcopy(input5))
    input6 = {'mode': np.array(False)}
    list_of_inputs.append(copy.deepcopy(input6))
    
    return list_of_inputs
def arccosh_inputs():
    list_of_inputs = []
    input1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([1, 2, 3], dtype=np.int64) + 0.0
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float32).reshape(2,3)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([1, 5, 10, 20]).reshape(2, 2).astype(np.float64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape(2, 2, 2).astype(np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def fractionalmaxpool3d_inputs():
    list_of_inputs = []
    # Case 1: Using output_size
    input_dict = {
        'kernel_size': (3, 3, 3),
        'output_size': (13, 12, 11),
        'output_ratio': None,
        'return_indices': False,
        '_random_samples': torch.rand(1, 1, 1).numpy(),
        'input': torch.randn(20, 16, 50, 32, 16).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Using output_ratio
    input_dict = {
        'kernel_size': (3, 3, 3),
        'output_size': None,
        'output_ratio': (0.5, 0.5, 0.5),
        'return_indices': False,
        '_random_samples': torch.rand(1, 1, 1).numpy(),
        'input': torch.randn(20, 16, 50, 32, 16).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Integer kernel_size and return_indices=True
    input_dict = {
        'kernel_size': 2,
        'output_size': (7, 8, 9),
        'output_ratio': None,
        'return_indices': True,
        '_random_samples': torch.rand(1, 1, 1).numpy(),
        'input': torch.randn(20, 16, 50, 32, 16).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def is_autocast_enabled_inputs():
    list_of_inputs = []
    list_of_inputs.append({})
    list_of_inputs.append({})
    list_of_inputs.append({})
    list_of_inputs.append({})
    list_of_inputs.append({})
    
    return list_of_inputs
def spherical_bessel_j0_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensor
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Negative values
    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Zero value
    input3 = np.array([0.0], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Multi-dimensional array
    input4 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Large values
    input5 = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def all_inputs():
    list_of_inputs = []
    # Input 1: Basic boolean tensor, dim=0, keepdim=False
    input1 = torch.tensor([[True, True], [False, True]]).bool().numpy()
    input_dict1 = {"input": input1, "dim": 0, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Integer tensor, dim=1, keepdim=True
    input2 = torch.tensor([[1, 1], [0, 1]]).int().numpy()
    input_dict2 = {"input": input2, "dim": 1, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Float tensor, dim=0, keepdim=True
    input3 = torch.tensor([[1.0, 1.0], [0.0, 1.0]]).float().numpy()
    input_dict3 = {"input": input3, "dim": 0, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 3D boolean tensor, dim=2, keepdim=False
    input4 = torch.tensor([[[True, True], [False, True]], [[True, False], [True, True]]]).bool().numpy()
    input_dict4 = {"input": input4, "dim": 2, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 2D integer tensor, dim=0, keepdim=False
    input5 = torch.tensor([[1, 1, 1], [1, 1, 1]]).int().numpy()
    input_dict5 = {"input": input5, "dim": 0, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: 2D boolean tensor, dim=1, keepdim=True
    input6 = torch.tensor([[True, True, True], [False, True, False]]).bool().numpy()
    input_dict6 = {"input": input6, "dim": 1, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: 3D float tensor, dim=1, keepdim=True
    input7 = torch.randn(2, 3, 4).float().numpy()
    input_dict7 = {"input": input7, "dim": 1, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    # Input 8: 1D integer tensor, dim=0 (should raise error due to scalar reduction), keepdim=False
    input8 = torch.tensor([1, 1, 0, 1]).int().numpy()
    input_dict8 = {"input": input8, "dim": 0, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9: 2D float tensor, dim = (0,1), keepdim = True
    input9 = torch.randn(2, 2).float().numpy()
    input_dict9 = {"input": input9, "dim": (0,1), "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    return list_of_inputs
def quantize_per_tensor_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensor
    input1 = torch.randn(3, 4).numpy()
    scale1 = 0.5
    zero_point1 = 0
    dtype1 = torch.qint8
    input_dict1 = {
        "input": input1,
        "scale": scale1,
        "zero_point": zero_point1,
        "dtype": dtype1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Float tensor with negative values
    input2 = torch.randn(2, 2, 2).numpy()
    scale2 = 0.1
    zero_point2 = 5
    dtype2 = torch.quint8
    input_dict2 = {
        "input": input2,
        "scale": scale2,
        "zero_point": zero_point2,
        "dtype": dtype2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 1D tensor
    input3 = torch.randn(10).numpy()
    scale3 = 1.0
    zero_point3 = 2
    dtype3 = torch.quint8
    input_dict3 = {
        "input": input3,
        "scale": scale3,
        "zero_point": zero_point3,
        "dtype": dtype3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Large tensor
    input4 = torch.randn(100, 100).numpy()
    scale4 = 0.01
    zero_point4 = 128
    dtype4 = torch.quint8
    
    input_dict4 = {
        "input": input4,
        "scale": scale4,
        "zero_point": zero_point4,
        "dtype": dtype4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Tensor with large scale
    input5 = torch.randn(5, 5).numpy()
    scale5 = 10.0
    zero_point5 = 0
    dtype5 = torch.qint8
    input_dict5 = {
        "input": input5,
        "scale": scale5,
        "zero_point": zero_point5,
        "dtype": dtype5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: 4D Tensor
    input6 = torch.randn(1, 3, 224, 224).numpy()
    scale6 = 0.005
    zero_point6 = 0
    dtype6 = torch.qint8
    
    input_dict6 = {
        "input": input6,
        "scale": scale6,
        "zero_point": zero_point6,
        "dtype": dtype6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def matrix_power_inputs():
    list_of_inputs = []
    # Input 1: Basic float matrix
    input1 = np.random.rand(3, 3).astype(np.float32)
    n1 = 2
    input_dict1 = {"input": input1, "n": n1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Integer matrix
    input2 = np.random.randint(1, 5, size=(2, 2)).astype(np.int32)
    n2 = 3
    input_dict2 = {"input": input2, "n": n2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Complex matrix
    input3 = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
    input3 = input3.astype(np.complex64)
    n3 = 2
    input_dict3 = {"input": input3, "n": n3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Larger matrix
    input4 = np.random.rand(5, 5).astype(np.float64)
    n4 = 4
    input_dict4 = {"input": input4, "n": n4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Negative power
    input5 = np.random.rand(2, 2).astype(np.float32)
    n5 = -2
    input_dict5 = {"input": input5, "n": n5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Identity matrix
    input6 = np.eye(4).astype(np.float32)
    n6 = 5
    input_dict6 = {"input": input6, "n": n6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: Small matrix with negative values
    input7 = np.random.randn(3, 3).astype(np.float32)
    n7 = 2
    input_dict7 = {"input": input7, "n": n7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

def mish_inputs():
    list_of_inputs = []
    # Input 1: 1D tensor, float
    input1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D tensor, float, negative values
    input2 = torch.randn(2, 3) * -1.0
    input2 = input2.numpy()
    input_dict2 = {"input": input2, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 3D tensor, float
    input3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {"input": input3, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 4D tensor, float
    input4 = torch.randn(2, 3, 4, 5).numpy()
    input_dict4 = {"input": input4, "inplace": True}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 5D tensor, float
    input5 = torch.randn(2, 3, 4, 5, 6).numpy()
    input_dict5 = {"input": input5, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Scalar float
    input7 = torch.randn(1).item()
    input_dict7 = {"input": np.array(input7), "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def cosh__inputs():
    list_of_inputs = []
    input1 = np.array([0.0, 0.5, 1.0, -0.5, -1.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[0.0, 0.5], [1.0, -0.5]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([0.0 + 1.0j, 0.5 - 0.5j, -1.0 + 0.0j], dtype=np.complex64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float16)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def use_deterministic_algorithms_inputs():
    list_of_inputs = []
    input1 = {"mode": True}
    list_of_inputs.append(copy.deepcopy(input1))
    input2 = {"mode": False}
    list_of_inputs.append(copy.deepcopy(input2))
    input3 = {"mode": np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input3))
    input4 = {"mode": np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input4))
    input5 = {"mode": np.array(True)}
    list_of_inputs.append(copy.deepcopy(input5))
    input6 = {"mode": np.array(False)}
    list_of_inputs.append(copy.deepcopy(input6))
    
    return list_of_inputs
def isreal_inputs():
    list_of_inputs = []
    # Input 1: Integer tensor
    input1 = torch.tensor([1, 2, 3, -4, 5]).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Float tensor
    input2 = torch.tensor([1.0, 2.5, -3.2, 4.7, 5.0]).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Complex tensor with zero imaginary parts
    input3 = torch.tensor([1 + 0j, 2 + 0j, 3 + 0j, -4 + 0j, 5 + 0j]).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Complex tensor with non-zero imaginary parts
    input4 = torch.tensor([1 + 1j, 2 - 2j, 3 + 0.5j, -4 - 1j, 5 + 2j]).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Multidimensional tensor (2D) with mixed types
    input5 = torch.tensor([[1, 2.0, 3 + 0j], [4, -5.0, 6 + 1j]]).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Empty tensor
    input6 = torch.tensor([]).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Multidimensional tensor (3D) with mixed types
    input7 = torch.tensor([[[1, 2.0, 3 + 0j], [4, -5.0, 6 + 1j]],[[7, 8.0, 9 + 0j], [10, -11.0, 12 + 1j]]]).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def adaptive_avg_pool3d_inputs():
    list_of_inputs = []
    # Test case 1: Tuple (D, H, W) with integers
    input_dict = {
        "output_size": (5, 7, 9),
        "input": np.random.randn(1, 64, 8, 9, 10).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 2: Single integer (cube)
    input_dict = {
        "output_size": 7,
        "input": np.random.randn(1, 64, 10, 9, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 3: Tuple with None values
    input_dict = {
        "output_size": (7, None, None),
        "input": np.random.randn(1, 64, 10, 9, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 4: Tuple with mixed int and None
    input_dict = {
        "output_size": (None, 9, None),
        "input": np.random.randn(1, 64, 10, 9, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 5: All None values (should be same as input)
    input_dict = {
        "output_size": (None, None, None),
        "input": np.random.randn(1, 64, 10, 9, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def any_inputs():
    list_of_inputs = []
    input = torch.randn(2, 3).bool().numpy()
    dim = 1
    keepdim = False
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randint(0, 2, (4, 4)).bool().numpy()
    dim = 0
    keepdim = True
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 5, 5) < 0.5
    input = input.numpy()
    dim = 2
    keepdim = False
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(3, 2, 4) > 0
    input = input.numpy()
    dim = 0
    keepdim = True
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = (torch.arange(12).reshape(3, 4) % 2).bool().numpy()
    dim = 1
    keepdim = False
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.tensor([[[True, False], [False, True]], [[False, False], [True, True]]]).numpy()
    dim = (0, 1)
    keepdim = True
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.tensor([[[True, False], [False, True]], [[False, False], [True, True]]]).numpy()
    dim = (0, 2)
    keepdim = False
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def ReLU_inputs():
    list_of_inputs = []
    # Input 1: 1D float tensor with positive and negative values
    input1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_dict1 = {
        "input": input1,
        "inplace": False
    }
    list_of_inputs.append(input_dict1)
    # Input 2: 2D float tensor with only positive values
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict2 = {
        "input": input2,
        "inplace": True
    }
    list_of_inputs.append(input_dict2)
    # Input 3: 3D float tensor with mixed values
    input3 = np.array([[[ -1.0, 2.0], [0.0, -4.0]], [[5.0, -6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict3 = {
        "input": input3,
        "inplace": False
    }
    list_of_inputs.append(input_dict3)
    # Input 4: 1D int tensor with mixed values
    input4 = np.array([-1, 0, 1, 2, -2], dtype=np.int32)
    input_dict4 = {
        "input": input4,
        "inplace": True
    }
    list_of_inputs.append(input_dict4)
    
    # Input 5: 2D int tensor with all negative values
    input5 = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    input_dict5 = {
        "input": input5,
        "inplace": False
    }
    list_of_inputs.append(input_dict5)
    return list_of_inputs
def lu_unpack_inputs():
    list_of_inputs = []
    # Input 1: Basic case with float data and pivots
    LU_data = torch.randn(3, 3).numpy()
    LU_pivots = torch.arange(1, 4).int().numpy()
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": True,
        "unpack_pivots": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Integer data and pivots
    LU_data = torch.randint(1, 10, (4, 4)).numpy()
    LU_pivots = torch.arange(1, 5).int().numpy()
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": True,
        "unpack_pivots": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Complex data and pivots
    LU_data = (torch.randn(2, 2) + 1j * torch.randn(2, 2)).numpy()
    LU_pivots = torch.arange(1, 3).int().numpy()
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": True,
        "unpack_pivots": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: 2D data, 1D pivots, no unpacking of pivots
    LU_data = torch.randn(5, 5).numpy()
    LU_pivots = torch.arange(1, 6).int().numpy()
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": True,
        "unpack_pivots": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D data, 1D pivots, no unpacking of data
    LU_data = torch.randn(5, 5).numpy()
    LU_pivots = torch.arange(1, 6).int().numpy()
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": False,
        "unpack_pivots": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def BCEWithLogitsLoss_inputs():
    list_of_inputs = []
    # Case 1: No optional arguments, provide dummy input and target
    input_val = np.random.rand(3, 5).astype(np.float32)
    target_val = np.random.randint(0, 2, size=(3, 5)).astype(np.float32)
    input_dict = {"input": input_val, "target": target_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: weight
    weight = np.random.rand(5).astype(np.float32)
    input_val = np.random.rand(3, 5).astype(np.float32)
    target_val = np.random.randint(0, 2, size=(3, 5)).astype(np.float32)
    input_dict = {"input": input_val, "target": target_val, "weight": weight}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: pos_weight
    pos_weight = np.random.rand(5).astype(np.float32)
    input_val = np.random.rand(3, 5).astype(np.float32)
    target_val = np.random.randint(0, 2, size=(3, 5)).astype(np.float32)
    input_dict = {"input": input_val, "target": target_val, "pos_weight": pos_weight}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: reduction = 'sum'
    input_val = np.random.rand(3, 5).astype(np.float32)
    target_val = np.random.randint(0, 2, size=(3, 5)).astype(np.float32)
    input_dict = {"input": input_val, "target": target_val, "reduction": "sum"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: reduction = 'none'
    input_val = np.random.rand(3, 5).astype(np.float32)
    target_val = np.random.randint(0, 2, size=(3, 5)).astype(np.float32)
    input_dict = {"input": input_val, "target": target_val, "reduction": "none"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def clamp_max_inputs():
    list_of_inputs = []
    input1 = np.array([-1, 0, 1, 2, 3], dtype=np.int32)
    max1 = 2.0
    input_dict1 = {"input": input1, "max": max1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([-1.5, 0.0, 1.5, 2.5, 3.5], dtype=np.float32)
    max2 = 2.0
    input_dict2 = {"input": input2, "max": max2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.int64)
    max3 = 3.0
    input_dict3 = {"input": input3, "max": max3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([[-1.5, 0.0, 1.5], [2.5, 3.5, 4.5]], dtype=np.float64)
    max4 = 3.5
    input_dict4 = {"input": input4, "max": max4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16)
    max5 = 5.0
    input_dict5 = {"input": input5, "max": max5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=np.float16)
    max6 = 6.6
    input_dict6 = {"input": input6, "max": max6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([1, 2, 3], dtype=np.uint8)
    max7 = 2.0
    input_dict7 = {"input": input7, "max": max7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def pixel_unshuffle_inputs():
    list_of_inputs = []
    downscale_factor = 2
    input1 = torch.randn(1, 1, 4, 4).numpy()
    input_dict1 = {
        "downscale_factor": downscale_factor,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    downscale_factor = 3
    input2 = torch.randn(1, 1, 12, 12).numpy()
    input_dict2 = {
        "downscale_factor": downscale_factor,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    downscale_factor = 4
    input3 = torch.randn(2, 3, 20, 20).numpy()
    input_dict3 = {
        "downscale_factor": downscale_factor,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    downscale_factor = 2
    input4 = torch.randn(1, 5, 6, 6).numpy()
    input_dict4 = {
        "downscale_factor": downscale_factor,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    downscale_factor = 3
    input5 = torch.randn(4, 2, 9, 9).numpy()
    input_dict5 = {
        "downscale_factor": downscale_factor,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    downscale_factor = 2
    input6 = torch.randn(1, 1, 4, 4, 4).numpy()
    input_dict6 = {
        "downscale_factor": downscale_factor,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    downscale_factor = 2
    input7 = torch.randn(1, 1, 8, 8).numpy()
    input_dict7 = {
        "downscale_factor": downscale_factor,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def linalg_eigvals_inputs():
    list_of_inputs = []
    A = np.random.rand(2, 2).astype(np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = np.random.rand(3, 3).astype(np.float64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
    A = A.astype(np.complex64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    A = A.astype(np.complex128)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = np.random.randn(4, 4).astype(np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = np.random.rand(2, 3, 3).astype(np.float64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = np.random.rand(2, 2, 2) + 1j * np.random.rand(2, 2, 2)
    A = A.astype(np.complex64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = np.random.rand(4, 5, 5) + 1j * np.random.rand(4, 5, 5)
    A = A.astype(np.complex128)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = np.random.rand(2, 2).astype(np.float32) * -1
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = np.random.rand(3, 3).astype(np.float64) * -1
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def prepare_multiprocessing_environment_inputs():
    list_of_inputs = []
    # Input 1: rank = 0
    input_dict = {
        "rank": 0,
        "input": np.array([0])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: rank = 1
    input_dict = {
        "rank": 1,
        "input": np.array([1])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: rank = 2
    input_dict = {
        "rank": 2,
        "input": np.array([2])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: rank = 10
    input_dict = {
        "rank": 10,
        "input": np.array([10])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: rank = 100
    input_dict = {
        "rank": 100,
        "input": np.array([100])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def get_device_inputs():
    list_of_inputs = []
    # Input 1: Float tensor on CPU
    input_tensor_float = torch.randn(2, 3, 4, 5)
    input_dict_1 = {"input": input_tensor_float.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    # Input 2: Int tensor on CPU
    input_tensor_int = torch.randint(0, 10, (3, 2, 5))
    input_dict_2 = {"input": input_tensor_int.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    # Input 3: Long tensor on CPU
    input_tensor_long = torch.randint(0, 100, (2, 2, 2), dtype=torch.long)
    input_dict_3 = {"input": input_tensor_long.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    # Input 4: Bool tensor on CPU
    input_tensor_bool = torch.randint(0, 2, (4, 4)).bool()
    input_dict_4 = {"input": input_tensor_bool.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    # Input 5: Double tensor on CPU
    input_tensor_double = torch.randn(5, 5, dtype=torch.float64)
    input_dict_5 = {"input": input_tensor_double.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    # Input 6: One-dimensional tensor
    input_tensor_1d = torch.randn(10)
    input_dict_6 = {"input": input_tensor_1d.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    # Input 7: Zero-dimensional tensor
    input_tensor_0d = torch.tensor(5.0)
    input_dict_7 = {"input": input_tensor_0d.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    # Input 8: Tensor with large values
    input_tensor_large = torch.randn(2, 2) * 1e5
    input_dict_8 = {"input": input_tensor_large.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    return list_of_inputs
def arccos_inputs():
    list_of_inputs = []
    input1 = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[0.2, 0.4, 0.6], [-0.2, -0.4, -0.6]])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[[0.1, 0.2], [0.3, 0.4]], [[-0.1, -0.2], [-0.3, -0.4]]])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([0.0]).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([0.0]).astype(np.float64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([0.99999])
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([-0.99999])
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def gather_inputs():
    list_of_inputs = []
    # Case 1: 2D tensor, dim=0
    input = torch.tensor([[1, 2], [3, 4]]).numpy()
    index = torch.tensor([[0, 0], [1, 0]]).long().numpy()
    dim = 1
    input_dict = {"input": input, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: 3D tensor, dim=1
    input = torch.randn(2, 3, 4).numpy()
    index = torch.randint(0, 3, (2, 2, 4)).long().numpy()
    dim = 1
    input_dict = {"input": input, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: 2D tensor, dim=1, negative indices - REMOVED as it was causing errors
    # input = torch.randn(3, 5).numpy()
    # index = torch.tensor([[0, -1, 2], [1, -2, 3], [2, -3, 4]]).long().numpy()
    # dim = 1
    # input_dict = {"input": input, "dim": dim, "index": index}
    # list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: 1D tensor, dim=0
    input = torch.arange(5).float().numpy()
    index = torch.tensor([0, 2, 4]).long().numpy()
    dim = 0
    input_dict = {"input": input, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: 4D tensor, dim=2
    input = torch.randn(1, 2, 3, 4).numpy()
    index = torch.randint(0, 3, (1, 2, 2, 4)).long().numpy()
    dim = 2
    input_dict = {"input": input, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 6: 2D int tensor
    input = torch.randint(0, 10, (2, 3)).int().numpy()
    index = torch.tensor([[0, 1], [2, 0]]).long().numpy()
    dim = 1
    input_dict = {"input": input, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 7: 3D tensor, dim=0
    input = torch.randn(3, 4, 5).numpy()
    index = torch.randint(0, 3, (2, 4, 5)).long().numpy()
    dim = 0
    input_dict = {"input": input, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 8: 2D tensor, dim=-1 (last dimension) - Modified to prevent index out of bound
    input = torch.randn(3, 5).numpy()
    index = torch.randint(0, 5, (3, 2)).long().numpy() # Ensure index < 5
    dim = -1
    input_dict = {"input": input, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def is_autocast_cache_enabled_inputs():
    list_of_inputs = []
    
    input1 = {}
    list_of_inputs.append(copy.deepcopy(input1))
    
    input2 = {}
    list_of_inputs.append(copy.deepcopy(input2))
    
    input3 = {}
    list_of_inputs.append(copy.deepcopy(input3))
    
    input4 = {}
    list_of_inputs.append(copy.deepcopy(input4))
    
    input5 = {}
    list_of_inputs.append(copy.deepcopy(input5))
    
    input6 = {}
    list_of_inputs.append(copy.deepcopy(input6))
    return list_of_inputs
def set_grad_enabled_inputs():
    list_of_inputs = []
    input1 = {'set_grad_enabled': True}
    list_of_inputs.append(input1)
    input2 = {'set_grad_enabled': False}
    list_of_inputs.append(input2)
    return list_of_inputs
def smoothl1loss_inputs():
    list_of_inputs = []
    # Test case 1: Default parameters
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.1, 2.2, 3.3]),
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 2: Different reduction
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.1, 2.2, 3.3]),
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 3: No reduction
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.1, 2.2, 3.3]),
        "size_average": None,
        "reduce": None,
        "reduction": 'none',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 4: Different beta
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.1, 2.2, 3.3]),
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "beta": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 5: size_average = False
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.1, 2.2, 3.3]),
        "size_average": False,
        "reduce": None,
        "reduction": 'mean',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def asinh_inputs():
    list_of_inputs = []
    input1 = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[0.5, 1.5], [-0.5, -1.5]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[-10.0, 10.0], [-5.0, 5.0]], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(2, 2)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float16)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([[[1.0, -1.0], [2.0, -2.0]], [[3.0, -3.0], [4.0, -4.0]]], dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([0.0], dtype=np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def multi_dot_inputs():
    list_of_inputs = []
    # Test case 1: Basic 2D matrices
    a = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(3, 4).astype(np.float32)
    input_dict = {'tensors': [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 2: Chain of 3 matrices
    a = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(3, 4).astype(np.float32)
    c = np.random.rand(4, 2).astype(np.float32)
    input_dict = {'tensors': [a, b, c]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 3: Matrices with negative values
    a = (np.random.rand(2, 3) - 0.5).astype(np.float32)
    b = (np.random.rand(3, 4) - 0.5).astype(np.float32)
    input_dict = {'tensors': [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 4: A larger chain of matrices
    a = np.random.rand(5, 3).astype(np.float32)
    b = np.random.rand(3, 2).astype(np.float32)
    c = np.random.rand(2, 7).astype(np.float32)
    d = np.random.rand(7, 2).astype(np.float32)
    e = np.random.rand(2, 4).astype(np.float32)
    input_dict = {'tensors': [a, b, c, d, e]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 5: Matrices with float64
    a = np.random.rand(2, 3).astype(np.float64)
    b = np.random.rand(3, 4).astype(np.float64)
    input_dict = {'tensors': [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def identity_inputs():
    generated_inputs = []
    # Input 1: 1D float tensor
    input1 = torch.randn(5).numpy()
    generated_inputs.append({"input": input1})
    # Input 2: 2D int tensor
    input2 = torch.randint(-10, 10, (3, 4)).numpy().astype(np.int64)
    generated_inputs.append({"input": input2})
    # Input 3: 3D float tensor (complex causing issues)
    input3 = torch.randn(2, 3, 4).numpy()
    generated_inputs.append({"input": input3})
    # Input 4: 4D bool tensor
    input4 = torch.randint(0, 2, (2, 2, 2, 2)).bool().numpy()
    generated_inputs.append({"input": input4})
    # Input 5: 5D float tensor with negative values
    input5 = torch.randn(1, 2, 3, 4, 5).numpy()
    generated_inputs.append({"input": input5})
    return generated_inputs
def narrow_copy_inputs():
    list_of_inputs = []
    # Case 1: Basic 2D tensor, positive start and length
    input_tensor = torch.randn(5, 5).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "start": 1,
        "length": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: 3D tensor, negative start, length covering end
    input_tensor = torch.arange(24).reshape(2, 3, 4).float().numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "start": -2,
        "length": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: 1D tensor, start at 0, full length
    input_tensor = torch.arange(10).int().numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "start": 0,
        "length": 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: Complex tensor, specifying dim = 0
    input_tensor = torch.randn(3, 3, dtype=torch.complex64).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "start": 0,
        "length": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: 4D tensor, narrowing along the last dimension
    input_tensor = torch.randn(2, 3, 4, 5).double().numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 3,
        "start": 2,
        "length": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 6: Bool tensor
    input_tensor = torch.tensor([[True, False], [False, True]]).bool().numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "start": 0,
        "length": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def lazy_instance_norm1d_inputs():
    list_of_inputs = []
    # Input 1: Basic case with default parameters
    input_dict = {
        "input": torch.randn(1, 3, 10).numpy(),
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Different eps and momentum values
    input_dict = {
        "input": torch.randn(1, 5, 20).numpy(),
        "eps": 1e-04,
        "momentum": 0.2,
        "affine": False,
        "track_running_stats": False,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Using a specific dtype (torch.float64)
    input_dict = {
        "input": torch.randn(1, 2, 15).numpy(),
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: No affine transformation, keeping track of running stats
    input_dict = {
        "input": torch.randn(1, 4, 25).numpy(),
        "eps": 1e-06,
        "momentum": 0.05,
        "affine": False,
        "track_running_stats": True,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Affine transformation, no tracking of running stats
    input_dict = {
        "input": torch.randn(1, 6, 30).numpy(),
        "eps": 1e-07,
        "momentum": 0.15,
        "affine": True,
        "track_running_stats": False,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def diff_inputs():
    list_of_inputs = []
    # Input 1: Basic 1D tensor
    input1 = np.array([1, 3, 2])
    input_dict1 = {"input": input1, "n": 1, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D tensor, different dim, higher order difference
    input2 = np.array([[1, 2, 3], [3, 4, 5]])
    input_dict2 = {"input": input2, "n": 2, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 1D tensor with prepend and append
    input3 = np.array([1, 3, 2])
    prepend3 = np.array([0])
    append3 = np.array([4])
    input_dict3 = {"input": input3, "n": 1, "dim": 0, "prepend": prepend3, "append": append3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 3D tensor, negative values
    input4 = np.array([[[1, 2, 3], [-4, -5, -6]], [[7, 8, 9], [-10, -11, -12]]])
    input_dict4 = {"input": input4, "n": 1, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Empty tensor
    input5 = np.array([])
    input_dict5 = {"input": input5, "n": 1, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: float tensor
    input6 = np.array([1.5, 2.5, 3.5])
    input_dict6 = {"input": input6, "n": 1, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: n = 0, should return original array
    input7 = np.array([1, 2, 3])
    input_dict7 = {"input": input7, "n": 0, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def bilinear_inputs():
    list_of_inputs = []
    # Input 1
    input1 = np.random.randn(2, 3).astype(np.float32)
    input2 = np.random.randn(2, 4).astype(np.float32)
    weight = np.random.randn(5, 3, 4).astype(np.float32)
    bias = np.random.randn(5).astype(np.float32)
    input_dict = {"input1": input1, "input2": input2, "weight": weight, "bias": bias}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2
    input1 = np.random.randn(1, 5).astype(np.float64)
    input2 = np.random.randn(1, 2).astype(np.float64)
    weight = np.random.randn(3, 5, 2).astype(np.float64)
    bias = np.random.randn(3).astype(np.float64)
    input_dict = {"input1": input1, "input2": input2, "weight": weight, "bias": bias}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3 - negative values
    input1 = np.random.randn(3, 2).astype(np.float32) * -1
    input2 = np.random.randn(3, 6).astype(np.float32) * -1
    weight = np.random.randn(4, 2, 6).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32) * -1
    input_dict = {"input1": input1, "input2": input2, "weight": weight, "bias": bias}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - different sizes
    input1 = np.random.randn(4, 1).astype(np.float32)
    input2 = np.random.randn(4, 7).astype(np.float32)
    weight = np.random.randn(2, 1, 7).astype(np.float32)
    bias = np.random.randn(2).astype(np.float32)
    input_dict = {"input1": input1, "input2": input2, "weight": weight, "bias": bias}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5 - Larger sizes
    input1 = np.random.randn(5, 10).astype(np.float32)
    input2 = np.random.randn(5, 8).astype(np.float32)
    weight = np.random.randn(7, 10, 8).astype(np.float32)
    bias = np.random.randn(7).astype(np.float32)
    input_dict = {"input1": input1, "input2": input2, "weight": weight, "bias": bias}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def concat_inputs():
    list_of_inputs = []
    # Input 1: Basic 2D float tensors, dim=0
    tensors1 = [np.random.randn(2, 3).astype(np.float32), np.random.randn(3, 3).astype(np.float32)]
    input_dict1 = {"tensors": tensors1, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D int tensors, dim=1
    tensors2 = [np.random.randint(0, 10, size=(2, 4), dtype=np.int32), np.random.randint(0, 10, size=(2, 5), dtype=np.int32)]
    input_dict2 = {"tensors": tensors2, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 3D tensors, dim=2
    tensors3 = [np.random.randn(2, 3, 4).astype(np.float64), np.random.randn(2, 3, 5).astype(np.float64)]
    input_dict3 = {"tensors": tensors3, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 1D tensors, dim=0
    tensors4 = [np.array([1, 2, 3], dtype=np.int64), np.array([4, 5, 6], dtype=np.int64)]
    input_dict4 = {"tensors": tensors4, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: List of single tensor, dim = 0
    tensors5 = [np.random.randn(5, 5).astype(np.float32)]
    input_dict5 = {"tensors": tensors5, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Tensors with negative values, dim = 1
    tensors6 = [np.random.randn(2, 3).astype(np.float32), np.random.randn(2, 4).astype(np.float32)]
    input_dict6 = {"tensors": tensors6, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: Different dtypes in the list
    tensors7 = [np.random.randn(2, 3).astype(np.float32), np.random.randn(2, 3).astype(np.float64)]
    input_dict7 = {"tensors": tensors7, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def lerp_inputs():
    list_of_inputs = []
    # Test case 1: Basic float tensors with scalar weight
    start = torch.arange(1., 5.)
    end = torch.full_like(torch.arange(1., 5.), 10.)
    weight = 0.5
    input_dict = {"input": start.numpy(), "end": end.numpy(), "weight": weight}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 2: Float tensors with tensor weight
    start = torch.arange(1., 5.)
    end = torch.full_like(torch.arange(1., 5.), 10.)
    weight = torch.full_like(torch.arange(1., 5.), 0.5)
    input_dict = {"input": start.numpy(), "end": end.numpy(), "weight": weight.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 3: Int tensors with scalar weight
    start = torch.arange(1, 5).float()
    end = torch.full_like(torch.arange(1, 5).float(), 10.)
    weight = 0.5
    input_dict = {"input": start.numpy(), "end": end.numpy(), "weight": weight}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 4: Int tensors with tensor weight
    start = torch.arange(1, 5).float()
    end = torch.full_like(torch.arange(1, 5).float(), 10.)
    weight = torch.full_like(torch.arange(1, 5, dtype=torch.float32), 0.5)
    input_dict = {"input": start.numpy(), "end": end.numpy(), "weight": weight.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 5: Negative values and different weight
    start = torch.arange(-2., 2.)
    end = torch.full_like(torch.arange(-2., 2.), -5.)
    weight = 0.75
    input_dict = {"input": start.numpy(), "end": end.numpy(), "weight": weight}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def square__inputs():
    list_of_inputs = []
    # Input 1: 1D float tensor with positive and negative values
    input1 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D int tensor
    input2 = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.int32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 3D float tensor
    input3 = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 1D complex tensor
    input4 = np.array([1+1j, 2-2j, 3+0j], dtype=np.complex64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 2D complex tensor
    input5 = np.array([[1+1j, 2-2j], [3+0j, 0-1j]], dtype=np.complex128)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: 0D tensor (scalar)
    input6 = np.array(-5, dtype=np.int64)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs
def unravel_index_inputs():
    list_of_inputs = []
    indices = np.array([22, 41, 37]).astype(np.int64)
    shape = (7, 6)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).astype(np.int64)
    shape = (3, 4)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    indices = np.array([[0, 1], [2, 3], [4, 5]]).astype(np.int64)
    shape = (2, 3)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    indices = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]).astype(np.int64)
    shape = (2, 2)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    indices = np.array(7).astype(np.int64)
    shape = (3, 5)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    indices = np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.int64)
    shape = (2, 2, 2)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    indices = np.array([1, 2, 3, 4]).astype(np.int64)
    shape = (2, 2)
    input_dict = {"indices": indices, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def torch_linalg_pinv_inputs():
    generated_inputs = []
    # Input 1: Basic float tensor
    A = torch.randn(3, 5).numpy()
    input_dict = {"A": A, "rtol": 1e-5, "atol": 1e-8, "hermitian": False}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Batch of matrices, different rtol and atol
    A = torch.randn(2, 6, 3).numpy()
    input_dict = {"A": A, "rtol": 1e-3, "atol": 1e-6, "hermitian": False}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Complex Hermitian matrix
    A = torch.randn(3, 3, dtype=torch.complex64)
    A = (A + A.T.conj()).numpy()
    input_dict = {"A": A, "rtol": 1e-5, "atol": 1e-8, "hermitian": True}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Double precision tensor with specified atol only
    A = torch.randn(4, 4, dtype=torch.float64).numpy()
    input_dict = {"A": A, "rtol": None, "atol": 1e-9, "hermitian": False}
    generated_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Single value matrix with negative value
    A = torch.tensor([[-2.0]]).numpy()
    input_dict = {"A": A, "rtol": 1e-5, "atol": 1e-8, "hermitian": False}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Different shape matrix
    A = torch.randn(5, 2).numpy()
    input_dict = {"A": A, "rtol": 1e-5, "atol": 1e-8, "hermitian": False}
    generated_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Matrix of zeros
    A = torch.zeros(2, 3).numpy()
    input_dict = {"A": A, "rtol": 1e-5, "atol": 1e-8, "hermitian": False}
    generated_inputs.append(copy.deepcopy(input_dict))
    return generated_inputs
def index_put_inputs():
    list_of_inputs = []
    # Input 1: Basic example with float tensor, boolean indices, float values
    input_tensor = torch.randn(5, 5).numpy()
    indices = ([torch.tensor([0, 2, 4]).numpy(), torch.tensor([1, 3, 0]).numpy()],)
    values = torch.randn(3).numpy()
    accumulate = False
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": accumulate
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Integer tensor, Long indices, Integer values, accumulate=True
    input_tensor = torch.randint(0, 10, (3, 3)).numpy()
    indices = ([torch.tensor([0, 1]).long().numpy(), torch.tensor([1, 2]).long().numpy()],)
    values = torch.randint(0, 5, (2,)).numpy()
    accumulate = True
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": accumulate
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: 3D tensor, mixed indices, values as tensor, accumulate=False
    input_tensor = torch.randn(2, 3, 4).numpy()
    indices = ([torch.tensor([0, 1]).numpy(), torch.tensor([0, 1]).numpy(), torch.tensor([1, 2]).numpy()],)
    values = torch.randn(2).numpy()
    accumulate = False
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": accumulate
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: 1D tensor, single index, single value
    input_tensor = torch.arange(5).float().numpy()
    indices = ([torch.tensor([2]).numpy()],)
    values = torch.tensor([10.0]).numpy()
    accumulate = False
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": accumulate
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Negative indices, float tensor
    input_tensor = torch.randn(4, 4).numpy()
    indices = ([torch.tensor([0, -1]).numpy(), torch.tensor([1, -2]).numpy()],)
    values = torch.randn(2).numpy()
    accumulate = False
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": accumulate
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: boolean tensor, boolean indices, boolean values
    input_tensor = torch.randint(0, 2, (2, 2), dtype=torch.bool).numpy()
    indices = ([torch.tensor([0, 1]).numpy(), torch.tensor([0, 1]).numpy()],)
    values = torch.tensor([True, False]).bool().numpy()
    accumulate = False
    input_dict = {
        "input": input_tensor,
        "indices": indices,
        "values": values,
        "accumulate": accumulate
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def threshold_inputs():
    list_of_inputs = []
    input_dict = {
        "input": np.array([1.0, 0.0, -1.0]),
        "threshold": 0.5,
        "value": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "input": np.array([-0.5, -0.1, 0.2]),
        "threshold": -0.2,
        "value": 0.0,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "input": np.array([0.0, -0.5, 0.5]),
        "threshold": 0.0,
        "value": -1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "input": np.array([2.0, 0.5, -0.5]),
        "threshold": 1.0,
        "value": 10.0,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "input": np.array([-2.0, -0.5, 0.5]),
        "threshold": -1.0,
        "value": -5.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def get_default_dtype_inputs():
    list_of_inputs = []
    list_of_inputs.append({})
    list_of_inputs.append({})
    list_of_inputs.append({})
    list_of_inputs.append({})
    list_of_inputs.append({})
    return list_of_inputs
def scatter_reduce_inputs():
    list_of_inputs = []
    # Test case 1: Basic sum reduction
    input1 = np.zeros((5,), dtype=np.float32)
    index1 = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    src1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    dim1 = 0
    reduce1 = "sum"
    include_self1 = False
    input_dict1 = {"input": input1, "dim": dim1, "index": index1, "src": src1, "reduce": reduce1, "include_self": include_self1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Test case 2:  Multi-dimensional input with mean reduction
    input2 = np.zeros((2, 3), dtype=np.float32)
    index2 = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
    src2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    dim2 = 1
    reduce2 = "mean"
    include_self2 = False
    input_dict2 = {"input": input2, "dim": dim2, "index": index2, "src": src2, "reduce": reduce2, "include_self": include_self2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Test case 3: Use include_self = True
    input3 = np.ones((5,), dtype=np.float32)
    index3 = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    src3 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    dim3 = 0
    reduce3 = "sum"
    include_self3 = True
    input_dict3 = {"input": input3, "dim": dim3, "index": index3, "src": src3, "reduce": reduce3, "include_self": include_self3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Test case 4: min reduction
    input4 = np.array([5, 5, 5, 5, 5], dtype=np.float32)
    index4 = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    src4 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    dim4 = 0
    reduce4 = "min"
    include_self4 = True
    input_dict4 = {"input": input4, "dim": dim4, "index": index4, "src": src4, "reduce": reduce4, "include_self": include_self4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Test case 5: max reduction
    input5 = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    index5 = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    src5 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    dim5 = 0
    reduce5 = "max"
    include_self5 = True
    input_dict5 = {"input": input5, "dim": dim5, "index": index5, "src": src5, "reduce": reduce5, "include_self": include_self5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Test case 6: prod reduction
    input6 = np.ones((5,), dtype=np.float32)
    index6 = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    src6 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    dim6 = 0
    reduce6 = "prod"
    include_self6 = True
    input_dict6 = {"input": input6, "dim": dim6, "index": index6, "src": src6, "reduce": reduce6, "include_self": include_self6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Test case 7: float64
    input7 = np.zeros((5,), dtype=np.float64)
    index7 = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    src7 = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    dim7 = 0
    reduce7 = "sum"
    include_self7 = False
    input_dict7 = {"input": input7, "dim": dim7, "index": index7, "src": src7, "reduce": reduce7, "include_self": include_self7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def quantile_inputs():
    list_of_inputs = []
    # Example 1: Basic example with float input and scalar q
    input_dict = {
        "input": torch.randn(2, 3).numpy(),
        "q": np.array(0.5, dtype=np.float32),
        "dim": 1,
        "keepdim": True,
        "interpolation": "linear"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 2: Float input and 1D tensor q
    input_dict = {
        "input": torch.randn(4, 5).numpy(),
        "q": np.array([0.25, 0.5, 0.75], dtype=np.float32),
        "dim": 0,
        "keepdim": False,
        "interpolation": "linear"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 3: Integer input and scalar q, different interpolation
    input_dict = {
        "input": torch.randint(0, 10, (3, 4)).float().numpy(),
        "q": np.array(0.6, dtype=np.float32),
        "dim": 1,
        "keepdim": True,
        "interpolation": "higher"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 4: Negative values and 'nearest' interpolation
    input_dict = {
        "input": torch.randint(-5, 5, (2, 2)).float().numpy(),
        "q": np.array(0.3, dtype=np.float32),
        "dim": 1,
        "keepdim": False,
        "interpolation": "nearest"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: Multiple dimensions, 'midpoint' interpolation, flatten
    input_dict = {
        "input": torch.randn(2, 3, 4).numpy(),
        "q": np.array(0.5, dtype=np.float32),
        "dim": None,
        "keepdim": False,
        "interpolation": "midpoint"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 6: scalar q, dim = None
    input_dict = {
        "input": torch.arange(10).float().numpy(),
        "q": np.array(0.7, dtype=np.float32),
        "dim": None,
        "keepdim": False,
        "interpolation": "linear"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 7: 1D Tensor Input and scalar q
    input_dict = {
        "input": torch.arange(5).float().numpy(),
        "q": np.array(0.4, dtype=np.float32),
        "dim": 0,
        "keepdim": False,
        "interpolation": "linear"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def clip_inputs():
    list_of_inputs = []
    input_float = np.random.randn(3, 4).astype(np.float32)
    min_val = -0.5
    max_val = 1.5
    input_dict = {"input": input_float, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_int = np.random.randint(-5, 5, size=(2, 2), dtype=np.int32)
    min_val = -2
    max_val = 3
    input_dict = {"input": input_int.astype(np.float32), "min": float(min_val), "max": float(max_val)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_neg = np.random.randn(1, 5) * -1.0
    input_neg = input_neg.astype(np.float64)
    min_val = -2.0
    max_val = -0.5
    input_dict = {"input": input_neg, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_3d = np.random.rand(2, 3, 4).astype(np.float32)
    min_val = 0.2
    max_val = 0.8
    input_dict = {"input": input_3d, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_bool = np.random.choice([True, False], size=(2, 3)).astype(np.float32)
    min_val = 0.0
    max_val = 1.0
    input_dict = {"input": input_bool, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def Tanh_inputs():
    generated_inputs = []
    # Input 1: Scalar
    input1 = np.array(0.5)
    generated_inputs.append({"input": input1})
    # Input 2: 1D array (vector)
    input2 = np.array([-1.0, 0.0, 1.0, 2.0])
    generated_inputs.append({"input": input2})
    # Input 3: 2D array (matrix)
    input3 = np.array([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]])
    generated_inputs.append({"input": input3})
    # Input 4: 3D array
    input4 = np.random.rand(2, 3, 4)
    generated_inputs.append({"input": input4})
    # Input 5: Array with negative values and zeros
    input5 = np.array([[-1, 0, 1], [-2, 0, 2]])
    generated_inputs.append({"input": input5})
    
    return generated_inputs
def torch_histogram_inputs():
    list_of_inputs = []
    # Example 1: Basic usage with integer bins and default range
    input_tensor = torch.tensor([1.0, 2.0, 2.5, 3.0, 3.5, 4.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": 5,
        "range": None,
        "weight": None,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 2: Specifying range
    input_tensor = torch.tensor([1.0, 2.0, 2.5, 3.0, 3.5, 4.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": 5,
        "range": (0.0, 5.0),
        "weight": None,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 3: Using weights
    input_tensor = torch.tensor([1.0, 2.0, 2.5, 3.0, 3.5, 4.0]).numpy()
    weight_tensor = torch.tensor([1.0, 2.0, 1.5, 1.0, 0.5, 0.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": 5,
        "range": (0.0, 5.0),
        "weight": weight_tensor,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 4: Using density
    input_tensor = torch.tensor([1.0, 2.0, 2.5, 3.0, 3.5, 4.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": 5,
        "range": (0.0, 5.0),
        "weight": None,
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 5: Using tensor as bins
    input_tensor = torch.tensor([1.0, 2.0, 2.5, 3.0, 3.5, 4.0]).numpy()
    bins_tensor = torch.tensor([0.0, 1.5, 2.5, 3.5, 4.5, 5.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": bins_tensor,
        "range": None,
        "weight": None,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 6: Input tensor with negative values
    input_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": 4,
        "range": (-2.0, 3.0),
        "weight": None,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 7: Input tensor with float64
    input_tensor = torch.tensor([1.0, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=torch.float64).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": 5,
        "range": None,
        "weight": None,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def unflatten_inputs():
    list_of_inputs = []
    # Example 1
    input_tensor = torch.randn(2, 50).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "unflattened_size": (2, 5, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 2
    input_tensor = torch.randn(1, 100).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "unflattened_size": (10, 10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 3
    input_tensor = torch.randn(2, 25).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "unflattened_size": (5, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 4: Negative dimension
    input_tensor = torch.randn(2, 50).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": -1,
        "unflattened_size": (5, 10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 5: Unflatten dimension 0
    input_tensor = torch.randn(10).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "unflattened_size": (2, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def equal_inputs():
    list_of_inputs = []
    # Test case 1: Equal 1D integer tensors
    input1 = np.array([1, 2, 3])
    input2 = np.array([1, 2, 3])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 2: Unequal 1D integer tensors
    input1 = np.array([1, 2, 3])
    input2 = np.array([1, 2, 4])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 3: Equal 2D float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 4: Unequal 2D float tensors with different shapes
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = np.array([[1.0, 2.0]])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 5: Equal 3D boolean tensors
    input1 = np.array([[[True, False], [True, True]], [[False, False], [True, False]]])
    input2 = np.array([[[True, False], [True, True]], [[False, False], [True, False]]])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 6: Unequal 1D tensors with different data types
    input1 = np.array([1, 2, 3], dtype=np.int32)
    input2 = np.array([1, 2, 3], dtype=np.float64)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 7: Equal 1D complex tensors
    input1 = np.array([1+1j, 2+2j, 3+3j])
    input2 = np.array([1+1j, 2+2j, 3+3j])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 8: Unequal 1D complex tensors
    input1 = np.array([1+1j, 2+2j, 3+3j])
    input2 = np.array([1+1j, 2+2j, 4+4j])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 9: Equal 1D integer tensors with negative values
    input1 = np.array([-1, -2, 3])
    input2 = np.array([-1, -2, 3])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 10: Unequal 1D integer tensors with different values
    input1 = np.array([-1, -2, 3])
    input2 = np.array([-1, 2, 3])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def isin_inputs():
    list_of_inputs = []
    elements = np.array([[1, 2], [3, 4]])
    test_elements = np.array([2, 3])
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    elements = np.array([1, 2, 2, 3, 4, 5])
    test_elements = np.array([2, 4])
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    elements = np.array([1, 2, 2, 3, 4, 5])
    test_elements = np.array([2, 4])
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": True,
        "invert": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    elements = np.array([[1.5, 2.5], [3.5, 4.5]])
    test_elements = np.array([2.5, 3.5])
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    elements = np.array([-1, -2, 0, 1, 2])
    test_elements = np.array([-2, 1])
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    elements = np.array([1, 2])
    test_elements = 2
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    elements = 1
    test_elements = np.array([1,2,3])
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    elements = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    test_elements = np.array([2, 3, 7])
    input_dict = {
        "elements": elements,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def set_autocast_enabled_inputs():
    list_of_inputs = []
    input_dict = {
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def frexp_inputs():
    list_of_inputs = []
    # Input 1: 1D float tensor
    input1 = torch.arange(1, 9, dtype=torch.float32).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D float tensor with negative values
    input2 = torch.randn(3, 4, dtype=torch.float64).numpy() * -1
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Scalar float tensor
    input3 = torch.tensor(3.14, dtype=torch.float32).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Empty tensor
    input4 = torch.empty(0, dtype=torch.float32).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 3D float tensor
    input5 = torch.randn(2, 3, 5, dtype=torch.float32).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: 1D tensor with zeros
    input6 = torch.tensor([0.0, 1.0, 2.0, 0.0, 4.0], dtype=torch.float32).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Large values
    input7 = torch.tensor([2**30, 2**60, -2**30], dtype=torch.float64).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def set_autocast_xla_enabled_inputs():
    list_of_inputs = []
    input_dict_1 = {
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    input_dict_2 = {
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    input_dict_3 = {
        "enabled": bool(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    input_dict_4 = {
        "enabled": bool(0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    input_dict_5 = {
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    input_dict_6 = {
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    return list_of_inputs
def matrix_rank_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensor
    A = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Float tensor
    A = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]], dtype=np.float32)
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Complex tensor, hermitian=True
    A = np.array([[1+0j, 2-1j], [2+1j, 3+0j]], dtype=np.complex64)
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Different tolerance
    A = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    input_dict = {"A": A, "tol": 0.5, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Rectangular matrix
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Zero matrix
    A = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Matrix with negative values
    A = np.array([[1.0, -2.0], [-2.0, 4.0]], dtype=np.float32)
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Single element matrix
    A = np.array([[5.0]], dtype=np.float32)
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def linalg_inv_inputs():
    list_of_inputs = []
    A = torch.randn(4, 4).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(2, 3, 4, 4).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(4, 4, dtype=torch.complex128).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(3, 3).double().numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = (torch.randn(2, 2) + 1j * torch.randn(2, 2)).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(2, 2, 2, 2).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.eye(3).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = (torch.randn(2, 3, 4, 4) + 1j * torch.randn(2, 3, 4, 4)).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def expit_inputs():
    list_of_inputs = []
    input1 = np.array([0.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[-1, 0], [1, 2]], dtype=np.int64)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([[-1.5, 0.5], [1.5, 2.5]], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([-5, -4, -3, -2, -1], dtype=np.int32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def set_deterministic_debug_mode_inputs():
    list_of_inputs = []
    input_dict = {
        "mode": "default"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "mode": "warn"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "mode": "error"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "mode": "off"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "mode": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def negative_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randint(-10, 10, (2, 2, 2)).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(1).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(5,).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.zeros(2, 3).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = torch.ones(4, 4).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = (torch.rand(2, 3) * 100).int().numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    input8 = torch.randn(2, 3, 4).numpy()
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    input9 = torch.randn(1, 1, 1, 1).numpy()
    input_dict9 = {"input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    input10 = torch.tensor([1.0, -2.0, 3.0, -4.0]).numpy()
    input_dict10 = {"input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    return list_of_inputs
def arcsin_inputs():
    generated_inputs = []
    # Input 1: Basic float tensor
    input1 = np.array([0.0, 0.5, -0.5])
    input_dict1 = {"input": input1}
    generated_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Multi-dimensional float tensor
    input2 = np.array([[0.2, 0.4], [-0.1, -0.3]])
    input_dict2 = {"input": input2}
    generated_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Single element float tensor
    input3 = np.array(0.7)
    input_dict3 = {"input": input3}
    generated_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Larger multi-dimensional tensor
    input4 = np.random.uniform(low=-1.0, high=1.0, size=(2, 3, 4))
    input_dict4 = {"input": input4}
    generated_inputs.append(copy.deepcopy(input_dict4))
    return generated_inputs
def aminmax_inputs():
    list_of_inputs = []
    # Test case 1: 1D integer tensor
    input1 = np.array([1, -3, 5, 0, -2], dtype=np.int32)
    input_dict1 = {"input": input1, "dim": None, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Test case 2: 2D float tensor with dim and keepdim
    input2 = np.array([[1.5, -2.5, 3.5], [4.5, 0.5, -1.5]], dtype=np.float32)
    input_dict2 = {"input": input2, "dim": 0, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Test case 3: 3D tensor without dim
    input3 = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict3 = {"input": input3, "dim": None, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Test case 4: 2D int tensor with dim=1
    input4 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    input_dict4 = {"input": input4, "dim": 1, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Test case 5: 1D tensor with NaN
    input5 = np.array([1.0, -3.0, np.nan, 5.0], dtype=np.float32)
    input_dict5 = {"input": input5, "dim": None, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Test case 7: 2D tensor with negative values and keepdim=True
    input7 = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int32)
    input_dict7 = {"input": input7, "dim": 0, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def index_select_inputs():
    list_of_inputs = []
    # Case 1: 2D float tensor, dim=0, integer index
    input_tensor = torch.randn(3, 4).numpy()
    index_tensor = torch.tensor([0, 2]).numpy()
    input_dict = {"input": input_tensor, "dim": 0, "index": index_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: 3D int tensor, dim=1, integer index with negative values
    input_tensor = torch.randint(-5, 5, (2, 5, 3)).numpy()
    index_tensor = torch.tensor([0, 2]).numpy()
    input_dict = {"input": input_tensor, "dim": 1, "index": index_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: 1D float tensor, dim=0, integer index
    input_tensor = torch.randn(6).numpy()
    index_tensor = torch.tensor([1, 3, 5]).numpy()
    input_dict = {"input": input_tensor, "dim": 0, "index": index_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: 4D complex tensor, dim=2, integer index
    input_tensor = torch.randn(2, 3, 4, 5, dtype=torch.complex64).numpy()
    index_tensor = torch.tensor([0, 2]).numpy()
    input_dict = {"input": input_tensor, "dim": 2, "index": index_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: 2D tensor, dim=1, index containing duplicate values
    input_tensor = torch.randn(4, 4).numpy()
    index_tensor = torch.tensor([1, 1, 3, 0]).numpy()
    input_dict = {"input": input_tensor, "dim": 1, "index": index_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 6: 5D tensor, dim = 4, mixed positive and negative indices
    input_tensor = torch.randn(2, 2, 2, 2, 2).numpy()
    index_tensor = torch.tensor([0]).numpy()
    input_dict = {"input": input_tensor, "dim": 4, "index": index_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 7: Long Tensor Input
    input_tensor = torch.arange(24).reshape(2, 3, 4).long().numpy()
    index_tensor = torch.tensor([0, 2]).numpy()
    input_dict = {"input": input_tensor, "dim": 1, "index": index_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def true_divide_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other1 = np.array([[0.5, 1.0], [1.5, 2.0]])
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Integer tensors
    input2 = np.array([[1, 2], [3, 4]])
    other2 = np.array([[2, 2], [2, 2]])
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Negative values
    input3 = np.array([[-1.0, 2.0], [-3.0, 4.0]])
    other3 = np.array([[0.5, -1.0], [1.5, -2.0]])
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Different shapes (broadcasting)
    input4 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    other4 = np.array([1.0, 2.0, 3.0])
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Scalar divisor
    input5 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other5 = np.array(2.0)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Multidimensional arrays
    input6 = np.random.rand(2, 3, 4)
    other6 = np.random.rand(2, 3, 4)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: Zero values in divisor
    input7 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other7 = np.array([[0.5, 0.0], [1.5, 0.0]])
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def hspmm_inputs():
    list_of_inputs = []
    rows = 10
    cols = 5
    mat1_indices = torch.randint(0, 2, (2, 15))
    mat1_values = torch.randn(15)
    mat1 = torch.sparse_coo_tensor(mat1_indices, mat1_values, (rows, cols))
    mat2 = torch.randn(cols, 8)
    input_dict = {
        "mat1": mat1.to_dense().numpy(),
        "mat2": mat2.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    rows = 5
    cols = 10
    mat1_indices = torch.randint(0, 2, (2, 20))
    mat1_values = torch.randn(20)
    mat1 = torch.sparse_coo_tensor(mat1_indices, mat1_values, (rows, cols))
    mat2 = torch.randn(cols, 12)
    input_dict = {
        "mat1": mat1.to_dense().numpy(),
        "mat2": mat2.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    rows = 8
    cols = 3
    mat1_indices = torch.randint(0, 2, (2, 10))
    mat1_values = torch.randn(10)
    mat1 = torch.sparse_coo_tensor(mat1_indices, mat1_values, (rows, cols))
    mat2 = torch.randn(cols, 5)
    input_dict = {
        "mat1": mat1.to_dense().numpy(),
        "mat2": mat2.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    rows = 4
    cols = 7
    mat1_indices = torch.randint(0, 2, (2, 8))
    mat1_values = torch.randn(8)
    mat1 = torch.sparse_coo_tensor(mat1_indices, mat1_values, (rows, cols))
    mat2 = torch.randn(cols, 2)
    input_dict = {
        "mat1": mat1.to_dense().numpy(),
        "mat2": mat2.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    rows = 6
    cols = 9
    mat1_indices = torch.randint(0, 2, (2, 18))
    mat1_values = torch.randn(18)
    mat1 = torch.sparse_coo_tensor(mat1_indices, mat1_values, (rows, cols))
    mat2 = torch.randn(cols, 10)
    
    input_dict = {
        "mat1": mat1.to_dense().numpy(),
        "mat2": mat2.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def slogdet_inputs():
    list_of_inputs = []
    A = torch.randn(3, 3).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(2, 2).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(5, 5).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def erfcx_inputs():
    list_of_inputs = []
    input1 = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([[-0.5, -1.5], [-2.5, -3.5]], dtype=np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([0.0], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def sqrt__inputs():
    list_of_inputs = []
    input1 = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[[1.0, 4.0], [9.0, 16.0]], [[25.0, 36.0], [49.0, 64.0]]], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([1, 4, 9, 16], dtype=np.int32)
    input_dict4 = {"input": input4.astype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[1, 4], [9, 16]], dtype=np.int64)
    input_dict5 = {"input": input5.astype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([1.0], dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def freeze_inputs():
    list_of_inputs = []
    class MyModule1(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.linear = nn.Linear(out_features, out_features)
        def forward(self, input):
            output = self.weight.mm(input)
            output = self.linear(output)
            return output
    scripted_module1 = torch.jit.script(MyModule1(2, 3).eval())
    input_dict1 = {
        "mod": scripted_module1,
        "preserved_attrs": None,
        "optimize_numerics": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    class MyModule2(nn.Module):
        def __init__(self):
            super().__init__()
            self.modified_tensor = torch.tensor(10)
            self.version = 1
        def forward(self, input):
            self.modified_tensor += 1
            return input + self.modified_tensor
    scripted_module2 = torch.jit.script(MyModule2().eval())
    input_dict2 = {
        "mod": scripted_module2,
        "preserved_attrs": ["version"],
        "optimize_numerics": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    class MyModule3(nn.Module):
        def __init__(self):
            super().__init__()
            self.submodule = nn.Linear(5, 5)
            self.submodule.version = 1
        def forward(self, input):
            return self.submodule(input)
    scripted_module3 = torch.jit.script(MyModule3().eval())
    input_dict3 = {
        "mod": scripted_module3,
        "preserved_attrs": ["submodule.version"],
        "optimize_numerics": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    class MyModule4(nn.Module):
        def __init__(self):
            super().__init__()
            self.my_list = [1, 2, 3]
        def forward(self, input):
            return input
    scripted_module4 = torch.jit.script(MyModule4().eval())
    input_dict4 = {
        "mod": scripted_module4,
        "preserved_attrs": ["my_list"],
        "optimize_numerics": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    class MyModule5(nn.Module):
        def __init__(self):
            super().__init__()
            self.const_tensor = torch.randn(3,3)
        def forward(self, input):
            return self.const_tensor @ input
    scripted_module5 = torch.jit.script(MyModule5().eval())
    input_dict5 = {
        "mod": scripted_module5,
        "preserved_attrs": [],
        "optimize_numerics": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def arcsin_inputs():
    list_of_inputs = []
    input1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[-1.0, -0.5], [0.0, 0.5], [1.0, 0.2]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([-0.8, -0.2, 0.3, 0.9], dtype=np.float16)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([[-0.9, 0.1], [0.5, 0.7]], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([[[ -0.6, 0.2], [0.4, 0.8]], [[-0.3, 0.1], [0.9, 0.5]]], dtype=np.float64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array(-0.4, dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array(0.6, dtype=np.float64)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def is_storage_inputs():
    list_of_inputs = []
    # Input 1: Float storage
    float_tensor = torch.randn(5, 5)
    float_storage = float_tensor.storage()
    input_dict = {"obj": float_storage}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Int storage
    int_tensor = torch.randint(0, 10, (3, 3), dtype=torch.int32)
    int_storage = int_tensor.storage()
    input_dict = {"obj": int_storage}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Double Storage
    double_tensor = torch.randn(2, 2, dtype=torch.float64)
    double_storage = double_tensor.storage()
    input_dict = {"obj": double_storage}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Byte Storage
    byte_tensor = torch.randint(0, 256, (4, 4), dtype=torch.uint8)
    byte_storage = byte_tensor.storage()
    input_dict = {"obj": byte_storage}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Bool Storage
    bool_tensor = torch.tensor([[True, False], [False, True]])
    bool_storage = bool_tensor.storage()
    input_dict = {"obj": bool_storage}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Empty Storage
    empty_tensor = torch.empty(0)
    empty_storage = empty_tensor.storage()
    input_dict = {"obj": empty_storage}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def cross_inputs():
    list_of_inputs = []
    # Example 1: Basic 2D tensors with dim=1
    a = torch.randn(4, 3).numpy()
    b = torch.randn(4, 3).numpy()
    input_dict = {"input": a, "other": b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 2: Different tensor sizes with dim=0
    a = torch.randn(3, 4).numpy()
    b = torch.randn(3, 4).numpy()
    input_dict = {"input": a, "other": b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 3: 3D tensors with explicit dim=2
    a = torch.randn(2, 3, 3).numpy()
    b = torch.randn(2, 3, 3).numpy()
    input_dict = {"input": a, "other": b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 4: Integer tensors with default dim
    a = torch.randint(-5, 5, (4, 3)).numpy()
    b = torch.randint(-5, 5, (4, 3)).numpy()
    input_dict = {"input": a, "other": b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 5: Float tensors with negative values
    a = (torch.randn(4, 3) * -1).numpy()
    b = (torch.randn(4, 3) * -1).numpy()
    input_dict = {"input": a, "other": b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 6: 1D tensors, should cause error as dimension must be > 1 and size = 3 at dim=-1/0
    a = torch.randn(3).numpy()
    b = torch.randn(3).numpy()
    input_dict = {"input": a, "other": b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def autocast_decrement_nesting_inputs():
    list_of_inputs = []
    input1 = {"input": np.array([1])}
    list_of_inputs.append(copy.deepcopy(input1))
    input2 = {"input": np.array([1.0])}
    list_of_inputs.append(copy.deepcopy(input2))
    input3 = {"input": np.array([1+1j])}
    list_of_inputs.append(copy.deepcopy(input3))
    
    input4 = {"input": np.array([[1,2],[3,4]])}
    list_of_inputs.append(copy.deepcopy(input4))
    input5 = {"input": np.array([[[1,2],[3,4]],[[5,6],[7,8]]])}
    list_of_inputs.append(copy.deepcopy(input5))
    return list_of_inputs
def parameter_inputs():
    generated_inputs = []
    # Input 1: Float tensor, requires_grad=True
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"data": data, "requires_grad": True}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Float tensor, requires_grad=False
    data = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    input_dict = {"data": data, "requires_grad": False}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Int tensor, requires_grad=True.  This is likely invalid.
    # data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    # input_dict = {"data": data, "requires_grad": True}
    # generated_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Negative float tensor, requires_grad=True
    data = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {"data": data, "requires_grad": True}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Multi-dimensional float tensor, requires_grad=False
    data = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"data": data, "requires_grad": False}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Zero tensor, requires_grad=True
    data = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"data": data, "requires_grad": True}
    generated_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single element tensor, requires_grad=True
    data = np.array(3.14, dtype=np.float32)
    input_dict = {"data": data, "requires_grad": True}
    generated_inputs.append(copy.deepcopy(input_dict))
    return generated_inputs
def celu_inputs():
    list_of_inputs = []
    input1 = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    alpha1 = 1.0
    input_dict1 = {"input": input1, "alpha": alpha1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[-1, -0.5], [0, 0.5]], dtype=np.float64)
    alpha2 = 0.5
    input_dict2 = {"input": input2, "alpha": alpha2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[-1, -0.5], [0, 0.5]], dtype=np.float32)
    alpha3 = 2.0
    input_dict3 = {"input": input3, "alpha": alpha3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([[-1, -0.5], [0, 0.5]], dtype=np.float16)
    alpha4 = 0.75
    input_dict4 = {"input": input4, "alpha": alpha4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    alpha5 = 1.5
    input_dict5 = {"input": input5, "alpha": alpha5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([[-2, -1, 0, 1, 2]], dtype=np.float32)
    alpha6 = 0.25
    input_dict6 = {"input": input6, "alpha": alpha6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([[-3, -2, -1], [0, 1, 2]], dtype=np.float64)
    alpha7 = 0.8
    input_dict7 = {"input": input7, "alpha": alpha7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    alpha8 = 0.9
    input_dict8 = {"input": input8, "alpha": alpha8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    return list_of_inputs
def is_tracing_inputs():
    list_of_inputs = []
    return list_of_inputs
def script_warning_inputs():
    list_of_inputs = []
    input1 = {"input": np.array(1)}
    list_of_inputs.append(copy.deepcopy(input1))
    input2 = {"input": np.array([1, 2, 3])}
    list_of_inputs.append(copy.deepcopy(input2))
    
    input3 = {"input": np.array([[1, 2], [3, 4]])}
    list_of_inputs.append(copy.deepcopy(input3))
    input4 = {"input": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])}
    list_of_inputs.append(copy.deepcopy(input4))
    input5 = {"input": np.array(1.0)}
    list_of_inputs.append(copy.deepcopy(input5))
    return list_of_inputs
def entr_inputs():
    list_of_inputs = []
    input1 = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[0.2, 0.3], [0.4, 0.1]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([0.0, 1.0, 0.5, 0.25], dtype=np.float16)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    return list_of_inputs
def conj_physical__inputs():
    list_of_inputs = []
    # Input 1: Complex float64 tensor
    input1 = np.array([[1 + 2j, 3 - 4j], [5 + 6j, 7 - 8j]], dtype=np.complex128)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Complex float32 tensor with different values
    input2 = np.array([[-1 - 2j, 3 + 4j], [-5 - 6j, 7 + 8j]], dtype=np.complex64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 3D Complex float64 tensor
    input3 = np.array([[[1 + 2j, 3 - 4j], [5 + 6j, 7 - 8j]],
                       [[9 - 10j, 11 + 12j], [13 - 14j, 15 + 16j]]], dtype=np.complex128)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 1D Complex float32 tensor
    input4 = np.array([1j, 2 - 1j, 3 + 2j, -4 - 3j], dtype=np.complex64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Complex float64 tensor with zero values
    input5 = np.array([[0 + 0j, 0 - 0j], [0 + 0j, 0 - 0j]], dtype=np.complex128)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: 4D complex float32 tensor
    input6 = np.random.rand(2, 2, 2, 2).astype(np.float32) + 1j * np.random.rand(2, 2, 2, 2).astype(np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def is_warn_always_enabled_inputs():
    list_of_inputs = []
    
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def atan_inputs():
    list_of_inputs = []
    input1 = np.array([0.5, -0.2, 1.0, -0.8], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[-1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([1, 2, 3, 4], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[[1.0, -1.0], [2.0, -2.0]], [[3.0, -3.0], [4.0, -4.0]]], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array(5).astype(np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([0.0]).astype(np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def fft2_inputs():
    list_of_inputs = []
    # Input 1: Basic complex tensor
    input1 = torch.randn(10, 10, dtype=torch.complex64).numpy()
    input_dict1 = {
        "input": input1,
        "s": None,
        "dim": (-2, -1),
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Real tensor with specified s and dim
    input2 = torch.randn(12, 12).numpy()
    input_dict2 = {
        "input": input2,
        "s": (8, 8),
        "dim": (0, 1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Real tensor with ortho norm
    input3 = torch.randn(16, 16).numpy()
    input_dict3 = {
        "input": input3,
        "s": None,
        "dim": (-2, -1),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Different dimensions
    input4 = torch.randn(5, 7, 9, 11).numpy()
    input_dict4 = {
        "input": input4,
        "s": (7,9),
        "dim": (1,2),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Integer tensor
    input5 = torch.randint(0, 10, (8, 8)).numpy()
    input_dict5 = {
        "input": input5,
        "s": None,
        "dim": (-2, -1),
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def ceil__inputs():
    list_of_inputs = []
    input_dict = {"input": np.array([1.2, 2.5, -3.1, -0.5])}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {"input": np.array([[1.2, 2.5], [-3.1, -0.5]])}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {"input": np.array([1, 2, 3], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {"input": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {"input": np.array([0.0, -0.0, 1e-8, -1e-8])}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {"input": np.array([[[1.2, 2.5], [-3.1, -0.5]], [[-1.2, -2.5], [3.1, 0.5]]])}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {"input": np.array([np.nan, np.inf, -np.inf, 1.5])}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def masked_scatter_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensor, boolean mask, float source
    input_tensor = torch.randn(3, 4).numpy()
    mask_tensor = (torch.rand(3, 4) > 0.5).numpy()
    source_tensor = torch.randn(mask_tensor.sum()).numpy()
    input_dict = {"input": input_tensor, "mask": mask_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Int tensor, boolean mask, int source
    input_tensor = torch.randint(-5, 5, (2, 2)).numpy()
    mask_tensor = (torch.rand(2, 2) > 0.5).numpy()
    source_tensor = torch.randint(-5, 5, (mask_tensor.sum(),)).numpy()
    input_dict = {"input": input_tensor, "mask": mask_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: 3D tensor, boolean mask, float source
    input_tensor = torch.randn(2, 3, 2).numpy()
    mask_tensor = (torch.rand(2, 3, 2) > 0.5).numpy()
    source_tensor = torch.randn(mask_tensor.sum()).numpy()
    input_dict = {"input": input_tensor, "mask": mask_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Negative values in input and source
    input_tensor = torch.randn(2, 2) * -1.0
    input_tensor = input_tensor.numpy()
    mask_tensor = (torch.rand(2, 2) > 0.5).numpy()
    source_tensor = (torch.randn(mask_tensor.sum()) * -1.0).numpy()
    input_dict = {"input": input_tensor, "mask": mask_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 1D tensors
    input_tensor = torch.randn(5).numpy()
    mask_tensor = (torch.rand(5) > 0.5).numpy()
    source_tensor = torch.randn(mask_tensor.sum()).numpy()
    input_dict = {"input": input_tensor, "mask": mask_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Complex tensor
    input_tensor = torch.randn(2, 2, dtype=torch.complex64).numpy()
    mask_tensor = (torch.rand(2, 2) > 0.5).numpy()
    source_tensor = torch.randn(mask_tensor.sum(), dtype=torch.complex64).numpy()
    input_dict = {"input": input_tensor, "mask": mask_tensor, "source": source_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def row_stack_inputs():
    list_of_inputs = []
    # Input 1: Basic 2D float tensors
    tensors = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: 2D int tensors
    tensors = [np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([[5, 6], [7, 8]], dtype=np.int32)]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Different shapes along dim 1
    tensors = [np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: 1D tensors
    tensors = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: 3D tensors
    tensors = [np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
               np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Negative values
    tensors = [np.array([[-1.0, 2.0], [3.0, -4.0]]), np.array([[5.0, -6.0], [-7.0, 8.0]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 7: Mix of int and float, upcasted
    tensors = [np.array([[1, 2], [3, 4]]), np.array([[5.0, 6.0], [7.0, 8.0]])]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def are_deterministic_algorithms_enabled_inputs():
    list_of_inputs = []
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def set_module_inputs():
    list_of_inputs = []
    class MyModule1(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
        def forward(self, x):
            return self.linear(x)
    module1 = MyModule1()
    input_dict1 = {
        "name": "my_module1",
        "module": module1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    class MyModule2(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3)
        def forward(self, x):
            return self.conv(x)
    module2 = MyModule2()
    input_dict2 = {
        "name": "my_module2",
        "module": module2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    class MyModule3(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.Linear(10, 5)
            )
        def forward(self, x):
            return self.seq(x)
    module3 = MyModule3()
    input_dict3 = {
        "name": "my_module3",
        "module": module3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    class MyModule4(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(10, 20)
        def forward(self, x):
            out, _ = self.lstm(x)
            return out
    
    module4 = MyModule4()
    input_dict4 = {
        "name": "my_module4",
        "module": module4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    class MyModule5(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 128)
        def forward(self, x):
            return self.embedding(x)
    module5 = MyModule5()
    input_dict5 = {
        "name": "my_module5",
        "module": module5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    class MyModule6(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(32)
        def forward(self, x):
            return self.bn(x)
    
    module6 = MyModule6()
    input_dict6 = {
        "name": "my_module6",
        "module": module6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    class MyModule7(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(p=0.5)
        def forward(self, x):
            return self.dropout(x)
    
    module7 = MyModule7()
    input_dict7 = {
        "name": "my_module7",
        "module": module7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def celu_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensor with alpha=1.0
    input_tensor = torch.randn(3, 4).numpy()
    input_dict = {"input": input_tensor, "alpha": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Float tensor with negative values and alpha=0.5
    input_tensor = torch.randn(2, 2) * -1.0
    input_tensor = input_tensor.numpy()
    input_dict = {"input": input_tensor, "alpha": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Float tensor with alpha=2.0
    input_tensor = torch.randn(5,).numpy()
    input_dict = {"input": input_tensor, "alpha": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: High-dimensional tensor with alpha=0.25
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {"input": input_tensor, "alpha": 0.25}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Scalar tensor (0-dimensional) with alpha=1.5
    input_tensor = torch.randn(1).item()
    input_tensor = np.array(input_tensor)
    input_dict = {"input": input_tensor, "alpha": 1.5}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def scatter_add_inputs():
    list_of_inputs = []
    # Case 1: Basic case with float tensor, 1D
    input1 = torch.zeros(5).float().numpy()
    index1 = torch.tensor([0, 1, 2, 0, 3]).long().numpy()
    src1 = torch.randn(5).float().numpy()
    input_dict1 = {"input": input1, "dim": 0, "index": index1, "src": src1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Case 2: 2D input, dim=0. Reduced index range to avoid out-of-bounds error.
    input2 = torch.zeros(3, 5).float().numpy()
    index2 = torch.tensor([[0, 1, 2, 0, 1], [1, 2, 0, 2, 0]]).long().numpy()
    src2 = torch.randn(2, 5).float().numpy()
    input_dict2 = {"input": input2, "dim": 0, "index": index2, "src": src2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Case 3: 2D input, dim=1. Reduced index range to avoid out-of-bounds error.
    input3 = torch.zeros(2, 4).float().numpy()
    index3 = torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]]).long().numpy()
    src3 = torch.randn(2, 4).float().numpy()
    input_dict3 = {"input": input3, "dim": 1, "index": index3, "src": src3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Case 4: Integer input. Reduced index range to avoid out-of-bounds error.
    input4 = torch.zeros(5).int().numpy()
    index4 = torch.tensor([0, 1, 2, 0, 3]).long().numpy()
    src4 = torch.randint(0, 10, (5,)).int().numpy()
    input_dict4 = {"input": input4, "dim": 0, "index": index4, "src": src4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Case 5: 3D input, dim=0. Reduced index range to avoid out-of-bounds error.
    input5 = torch.zeros(3, 2, 4).float().numpy()
    index5 = torch.tensor([[[0, 1, 2, 0], [1, 0, 2, 2]], [[1, 2, 0, 1], [0, 2, 1, 2]]]).long().numpy()
    src5 = torch.randn(2, 2, 4).float().numpy()
    input_dict5 = {"input": input5, "dim": 0, "index": index5, "src": src5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def relu_inputs():
    list_of_inputs = []
    # Input 1: Basic 2D float tensor with negative values
    input1 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 1D integer tensor
    input2 = np.array([-1, 0, 1, 2, -3], dtype=np.int32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 3D float tensor
    input3 = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: tensor with all zeros
    input4 = np.zeros((5, 5), dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: tensor with mixed positive and negative values, large and small
    input5 = np.array([[-100.0, 0.001], [1.0, -0.0001], [10.0, -1.0]], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Higher dimensional tensor (4D)
    input6 = np.random.randn(1, 2, 3, 4).astype(np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Empty tensor
    input7 = np.array([], dtype=np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def fmax_inputs():
    list_of_inputs = []
    # Case 1: Basic float tensors
    input1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input2 = torch.tensor([3.0, 1.0, 2.0]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Int tensors
    input1 = torch.tensor([1, 2, 3]).numpy()
    input2 = torch.tensor([3, 1, 2]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Negative values and different shapes (broadcasting)
    input1 = torch.tensor([-1.0, 2.0, -3.0]).numpy()
    input2 = torch.tensor([[-2.0], [1.0], [-4.0]]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: NaNs
    input1 = torch.tensor([1.0, float('nan'), 3.0]).numpy()
    input2 = torch.tensor([float('nan'), 2.0, float('nan')]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Both NaNs
    input1 = torch.tensor([float('nan'), float('nan')]).numpy()
    input2 = torch.tensor([float('nan'), float('nan')]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 6: Multi-dimensional tensors
    input1 = torch.randn(2, 3).numpy()
    input2 = torch.randn(2, 3).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 7: Broadcasting with multi-dimensional tensors
    input1 = torch.randn(2, 3).numpy()
    input2 = torch.randn(3).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 8: Integer and Float tensors with different shapes
    input1 = torch.tensor([[1, 2], [3, 4]]).numpy()
    input2 = torch.tensor([1.5, 3.5]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 9: Larger Tensors
    input1 = torch.randn(5, 5, 5).numpy()
    input2 = torch.randn(5, 5, 5).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def nan_to_num_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensor with NaN, inf, -inf
    input1 = np.array([float('nan'), float('inf'), float('-inf'), 1.0, 2.0], dtype=np.float32)
    input_dict1 = {
        "input": input1,
        "nan": 0.0,
        "posinf": 1.0,
        "neginf": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Multi-dimensional float tensor
    input3 = np.array([[float('nan'), 1.0], [2.0, float('inf')]], dtype=np.float64)
    input_dict3 = {
        "input": input3,
        "nan": -1.0,
        "posinf": 10.0,
        "neginf": -10.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 3: Tensor with different replacement values
    input4 = np.array([float('nan'), float('inf'), float('-inf'), 0.0], dtype=np.float32)
    input_dict4 = {
        "input": input4,
        "nan": 100.0,
        "posinf": 200.0,
        "neginf": 300.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 4: Tensor with zero replacement values
    input5 = np.array([float('nan'), float('inf'), float('-inf')], dtype=np.float32)
    input_dict5 = {
        "input": input5,
        "nan": 0.0,
        "posinf": 0.0,
        "neginf": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 5: 3D tensor
    input6 = np.array([[[float('nan'), 1.0], [2.0, float('inf')]], [[3.0, float('-inf')], [4.0, 5.0]]], dtype=np.float32)
    input_dict6 = {
        "input": input6,
        "nan": 0.5,
        "posinf": 10.5,
        "neginf": -10.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def randn_like_inputs():
    list_of_inputs = []
    # Input 1: Float tensor
    input1 = torch.randn(2, 3, 4).numpy()
    input_dict1 = {
        "input": input1,
        "dtype": None,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 4: Specify dtype
    input4 = torch.randn(4, 4).numpy()
    input_dict4 = {
        "input": input4,
        "dtype": torch.float64,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 1D tensor
    input5 = torch.randn(10).numpy()
    input_dict5 = {
        "input": input5,
        "dtype": None,
        "layout": None,
        "requires_grad": True,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Negative values
    input6 = torch.randn(3, 3) * -1.0
    input6 = input6.numpy()
    input_dict6 = {
        "input": input6,
        "dtype": None,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: Complex tensor
    input7 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    input_dict7 = {
        "input": input7,
        "dtype": None,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def cosine_similarity_inputs():
    list_of_inputs = []
    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x2 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    dim = 1
    eps = 1e-8
    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]])
    x2 = np.array([[7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]])
    dim = 1
    eps = 1e-8
    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([4.0, 5.0, 6.0])
    dim = 0
    eps = 1e-8
    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    x2 = np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]])
    dim = 2
    eps = 1e-8
    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    x2 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float64)
    dim = 1
    eps = 1e-8
    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def is_inference_mode_enabled_inputs():
    list_of_inputs = []
    input1 = None
    list_of_inputs.append(copy.deepcopy(input1))
    return list_of_inputs
def lazy_instance_norm2d_inputs():
    generated_inputs = []
    # Input 1: Basic example with default parameters
    input1 = torch.randn(2, 3, 32, 32).numpy()
    input_dict1 = {
        'eps': 1e-05,
        'momentum': 0.1,
        'affine': True,
        'track_running_stats': True,
        'dtype': None,
        'input': input1
    }
    generated_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Different eps and momentum
    input2 = torch.randn(1, 5, 16, 16).numpy()
    input_dict2 = {
        'eps': 1e-03,
        'momentum': 0.2,
        'affine': True,
        'track_running_stats': True,
        'dtype': None,
        'input': input2
    }
    generated_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: affine=False, track_running_stats=False
    input3 = torch.randn(4, 7, 64, 64).numpy()
    input_dict3 = {
        'eps': 1e-05,
        'momentum': 0.1,
        'affine': False,
        'track_running_stats': False,
        'dtype': None,
        'input': input3
    }
    generated_inputs.append(copy.deepcopy(input_dict3))
    # Input 5: (C, H, W) input - REMOVED: This input caused an error because the driver expected an NCHW input and not CHW
    # input5 = torch.randn(1, 24, 24).numpy()
    # input_dict5 = {
    #     'eps': 1e-05,
    #     'momentum': 0.1,
    #     'affine': True,
    #     'track_running_stats': True,
    #     'dtype': None,
    #     'input': input5
    # }
    # generated_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Small input size
    input6 = torch.randn(1, 3, 2, 2).numpy()
    input_dict6 = {
        'eps': 1e-05,
        'momentum': 0.1,
        'affine': True,
        'track_running_stats': True,
        'dtype': None,
        'input': input6
    }
    generated_inputs.append(copy.deepcopy(input_dict6))
    return generated_inputs
def vecdot_inputs():
    list_of_inputs = []
    # Case 1: Basic float tensors, default dim
    x = np.random.randn(3, 2).astype(np.float32)
    y = np.random.randn(3, 2).astype(np.float32)
    input_dict = {"x": x, "y": y, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Complex tensors, default dim
    x = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    y = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    input_dict = {"x": x, "y": y, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Broadcasting, specified dim
    x = np.random.randn(2, 1, 3).astype(np.float64)
    y = np.random.randn(2, 4, 3).astype(np.float64)
    input_dict = {"x": x, "y": y, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: Different shapes, negative dim
    x = np.random.randn(5, 4, 2).astype(np.float32)
    y = np.random.randn(5, 4, 2).astype(np.float32)
    input_dict = {"x": x, "y": y, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Scalar input, dimension 0
    x = np.random.randn(3).astype(np.float32)
    y = np.random.randn(3).astype(np.float32)
    input_dict = {"x": x, "y": y, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def gammaincc_inputs():
    list_of_inputs = []
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    a = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    a = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    x = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([1.0], dtype=np.float32)
    x = np.array([1.0], dtype=np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    a = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    x = np.array([[1.1, 1.2], [1.3, 1.4]], dtype=np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def circularpad1d_inputs():
    list_of_inputs = []
    padding = 2
    input_dict = {
        "padding": padding,
        "input": np.arange(4, dtype=np.float32).reshape(2, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    padding = (3, 1)
    input_dict = {
        "padding": padding,
        "input": np.arange(6, dtype=np.float32).reshape(2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = 0
    input_dict = {
        "padding": padding,
        "input": np.arange(4, dtype=np.float32).reshape(1, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    padding = (0, 0)
    input_dict = {
        "padding": padding,
        "input": np.arange(4, dtype=np.int32).reshape(1, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    padding = 1
    input_dict = {
        "padding": padding,
        "input": np.arange(4, dtype=np.float64).reshape(1, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    padding = (1, 1)
    input_dict = {
        "padding": padding,
        "input": np.arange(4, dtype=np.int64).reshape(1, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    padding = 2
    input_dict = {
        "padding": padding,
        "input": np.arange(6, dtype=np.float32).reshape(2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def cosine_similarity_inputs():
    list_of_inputs = []
    input1 = np.random.randn(100, 128).astype(np.float32)
    input2 = np.random.randn(100, 128).astype(np.float32)
    input_dict = {
        "input1": input1,
        "input2": input2,
        "dim": 1,
        "eps": 1e-08
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.random.randn(50, 64).astype(np.float32)
    input2 = np.random.randn(50, 64).astype(np.float32)
    input_dict = {
        "input1": input1,
        "input2": input2,
        "dim": 0,
        "eps": 1e-06
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.random.randn(20, 30, 40).astype(np.float32)
    input2 = np.random.randn(20, 30, 40).astype(np.float32)
    input_dict = {
        "input1": input1,
        "input2": input2,
        "dim": -1,
        "eps": 1e-04
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.random.randn(10, 20, 30).astype(np.float32)
    input2 = np.random.randn(10, 20, 30).astype(np.float32)
    input_dict = {
        "input1": input1,
        "input2": input2,
        "dim": 2,
        "eps": 1e-02
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.random.randn(5, 10).astype(np.float32)
    input2 = np.random.randn(5, 10).astype(np.float32)
    input_dict = {
        "input1": input1,
        "input2": input2,
        "dim": 1,
        "eps": 1e-12
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def hsplit_inputs():
    list_of_inputs = []
    # Case 1: 2D tensor, integer sections
    t1 = torch.arange(16.0).reshape(4, 4).numpy()
    input_dict1 = {"input": t1, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Case 2: 2D tensor, list of indices
    t2 = torch.arange(16.0).reshape(4, 4).numpy()
    input_dict2 = {"input": t2, "indices_or_sections": [1, 3]}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Case 3: 1D tensor, integer sections
    t3 = torch.arange(12.0).numpy()
    input_dict3 = {"input": t3, "indices_or_sections": 3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Case 4: 1D tensor, list of indices
    t4 = torch.arange(12.0).numpy()
    input_dict4 = {"input": t4, "indices_or_sections": [2, 5, 8]}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Case 5: 3D tensor, integer sections
    t5 = torch.arange(24.0).reshape(2, 4, 3).numpy()
    input_dict5 = {"input": t5, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Case 6: 3D tensor, list of indices
    t6 = torch.arange(24.0).reshape(2, 3, 4).numpy()
    input_dict6 = {"input": t6, "indices_or_sections": [1, 2]}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Case 7: 2D int tensor
    t7 = torch.arange(16).reshape(4, 4).numpy()
    input_dict7 = {"input": t7, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def multiply_inputs():
    list_of_inputs = []
    input1 = np.random.randn(3, 4).astype(np.float32)
    other1 = np.random.randn(3, 4).astype(np.float32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.random.randint(-5, 5, size=(2, 2), dtype=np.int32)
    other2 = np.random.randint(-5, 5, size=(2, 2), dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([1, 2, 3], dtype=np.float64)
    other3 = np.array([4, 5, 6], dtype=np.float64)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.random.randn(1, 5, 5).astype(np.float32)
    other4 = np.random.randn(1, 5, 5).astype(np.float32)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array(-2.5)
    other5 = np.array(3.0)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([1+1j, 2+2j])
    other6 = np.array([3+3j, 4+4j])
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([[-1, -2], [-3, -4]], dtype=np.int8)
    other7 = np.array([[5, 6], [7, 8]], dtype=np.int8)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def addmv_inputs():
    list_of_inputs = []
    input_tensor = torch.randn(3).numpy()
    mat = torch.randn(3, 2).numpy()
    vec = torch.randn(2).numpy()
    beta = 1.0
    alpha = 1.0
    
    input_dict = {
        "input": input_tensor,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(5).numpy()
    mat = torch.randn(5, 4).numpy()
    vec = torch.randn(4).numpy()
    beta = 0.5
    alpha = 2.0
    
    input_dict = {
        "input": input_tensor,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.zeros(4).numpy()
    mat = torch.ones(4, 3).numpy()
    vec = torch.ones(3).numpy()
    beta = 0.0
    alpha = 1.0
    
    input_dict = {
        "input": input_tensor,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(2, dtype=torch.float64).numpy()
    mat = torch.randn(2, 3, dtype=torch.float64).numpy()
    vec = torch.randn(3, dtype=torch.float64).numpy()
    beta = 1.0
    alpha = 1.0
    
    input_dict = {
        "input": input_tensor,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(3).numpy()
    mat = torch.randn(3, 2).numpy()
    vec = torch.randn(2).numpy()
    beta = -0.5
    alpha = -1.0
    
    input_dict = {
        "input": input_tensor,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def get_num_threads_inputs():
    list_of_inputs = []
    
    list_of_inputs.append({})
    list_of_inputs.append({})
    list_of_inputs.append({})
    list_of_inputs.append({})
    list_of_inputs.append({})
    return list_of_inputs
def fftshift_inputs():
    generated_inputs = []
    # Test case 1: 1D float tensor
    input_1 = torch.arange(-5, 5, dtype=torch.float32).numpy()
    input_dict_1 = {"input": input_1, "dim": None}
    generated_inputs.append(copy.deepcopy(input_dict_1))
    # Test case 2: 2D int tensor with specified dim
    input_2 = torch.arange(16).reshape(4, 4).to(torch.int32).numpy()
    input_dict_2 = {"input": input_2, "dim": (0,)}
    generated_inputs.append(copy.deepcopy(input_dict_2))
    # Test case 3: 3D complex tensor
    input_3 = torch.randn(2, 3, 4, dtype=torch.complex64).numpy()
    input_dict_3 = {"input": input_3, "dim": None}
    generated_inputs.append(copy.deepcopy(input_dict_3))
    # Test case 4: 2D float tensor with different dim
    input_4 = torch.randn(5, 5).numpy()
    input_dict_4 = {"input": input_4, "dim": (1,)}
    generated_inputs.append(copy.deepcopy(input_dict_4))
    # Test case 5: 1D tensor with only one element
    input_5 = torch.tensor([1.0]).numpy()
    input_dict_5 = {"input": input_5, "dim": None}
    generated_inputs.append(copy.deepcopy(input_dict_5))
    # Test case 6: 4D tensor with specified dimensions
    input_6 = torch.randn(2, 3, 4, 5).numpy()
    input_dict_6 = {"input": input_6, "dim": (0, 2)}
    generated_inputs.append(copy.deepcopy(input_dict_6))
    
    # Test case 7: 2D int tensor with negative values
    input_7 = torch.arange(-8, 8).reshape(4, 4).to(torch.int32).numpy()
    input_dict_7 = {"input": input_7, "dim": (0,1)}
    generated_inputs.append(copy.deepcopy(input_dict_7))
    return generated_inputs
def parameters_to_vector_inputs():
    list_of_inputs = []
    # Input 1: List of float tensors with different shapes
    params1 = [torch.randn(2, 3).numpy(), torch.randn(5).numpy(), torch.randn(1, 1, 4).numpy()]
    input_dict1 = {"parameters": params1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: List of int tensors with different shapes
    params2 = [torch.randint(0, 10, (2, 2)).numpy(), torch.randint(-5, 5, (3,)).numpy()]
    input_dict2 = {"parameters": params2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: List of mixed float and int tensors
    params3 = [torch.randn(3, 1).numpy(), torch.randint(0, 5, (2,)).numpy(), torch.randn(1).numpy()]
    input_dict3 = {"parameters": params3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: List containing a single tensor
    params4 = [torch.randn(4, 4).numpy()]
    input_dict4 = {"parameters": params4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: List with empty tensors
    params5 = [torch.randn(0).numpy(), torch.randn(2, 0).numpy(), torch.randn(0, 0, 0).numpy()]
    input_dict5 = {"parameters": params5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: List of tensors with negative values
    params6 = [torch.randn(2, 3) * -1.0, torch.randn(5) * -2.0, torch.randn(1, 1, 4) * -0.5]
    input_dict6 = {"parameters": [p.numpy() for p in params6]}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: List with complex tensors
    params7 = [torch.randn(2, 3, dtype=torch.complex64).numpy(), torch.randn(5, dtype=torch.complex64).numpy()]
    input_dict7 = {"parameters": params7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def set_default_device_inputs():
    list_of_inputs = []
    input1 = {
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input1))
    if torch.cuda.is_available():
        input2 = {
            "device": "cuda"
        }
        list_of_inputs.append(copy.deepcopy(input2))
        input3 = {
            "device": "cuda:0"
        }
        list_of_inputs.append(copy.deepcopy(input3))
    
    if torch.backends.mps.is_available():
        input4 = {
            "device": "mps"
        }
        list_of_inputs.append(copy.deepcopy(input4))
    input5 = {
        "device": "meta"
    }
    list_of_inputs.append(copy.deepcopy(input5))
    return list_of_inputs
def alias_copy_inputs():
    list_of_inputs = []
    # Input 1: Float tensor, 1D
    input1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Int tensor, 2D
    input2 = torch.randint(-10, 10, (3, 4)).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Complex tensor, 3D
    input3 = torch.randn(2, 3, 2, dtype=torch.complex64).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Bool tensor, 2D
    input4 = torch.randint(0, 2, (4, 5)).bool().numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Long tensor, 4D
    input5 = torch.randint(-5, 15, (1, 2, 3, 4), dtype=torch.int64).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Double tensor, 2D
    input6 = torch.randn(2, 2, dtype=torch.float64).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: Byte tensor, 3D
    input7 = torch.randint(0, 256, (3, 2, 3), dtype=torch.uint8).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def solve_triangular_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensor, upper triangular
    A = np.array([[2.0, 1.0], [0.0, 3.0]], dtype=np.float32)
    b = np.array([[8.0], [9.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "b": b,
        "upper": True,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Lower triangular, float tensor
    A = np.array([[2.0, 0.0], [1.0, 3.0]], dtype=np.float32)
    b = np.array([[8.0], [9.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "b": b,
        "upper": False,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Unitriangular, transposed
    A = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    b = np.array([[8.0], [9.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "b": b,
        "upper": True,
        "transpose": True,
        "unitriangular": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Multi-dimensional input
    A = np.array([[[2.0, 1.0], [0.0, 3.0]], [[1.0, 2.0], [0.0, 4.0]]], dtype=np.float32)
    b = np.array([[[8.0], [9.0]], [[5.0], [6.0]]], dtype=np.float32)
    input_dict = {
        "A": A,
        "b": b,
        "upper": True,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Complex tensor
    A = np.array([[2.0 + 1j, 1.0], [0.0, 3.0 - 2j]], dtype=np.complex64)
    b = np.array([[8.0 + 2j], [9.0 - 1j]], dtype=np.complex64)
    input_dict = {
        "A": A,
        "b": b,
        "upper": True,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Another multi-dimensional example, lower triangular
    A = np.array([[[1, 0], [2, 3]], [[4, 0], [5, 6]]], dtype=np.float32)
    b = np.array([[[7], [8]], [[9], [10]]], dtype=np.float32)
    input_dict = {
        "A": A,
        "b": b,
        "upper": False,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def lu_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensor, pivot=True
    A = torch.randn(3, 2).numpy()
    input_dict = {"A": A, "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Float tensor, pivot=True
    A = torch.randn(2, 5).numpy()
    input_dict = {"A": A, "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Double tensor, pivot=True
    A = torch.randn(4, 4).double().numpy()
    input_dict = {"A": A, "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Complex tensor, pivot=True
    A = (torch.randn(2, 3) + 1j * torch.randn(2, 3)).numpy()
    input_dict = {"A": A, "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Complex tensor, pivot=True
    A = (torch.randn(3, 3) + 1j * torch.randn(3, 3)).numpy()
    input_dict = {"A": A, "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Batch of matrices, pivot=True
    A = torch.randn(2, 3, 4).numpy()
    input_dict = {"A": A, "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 7: Batch of matrices, pivot=True
    A = torch.randn(3, 2, 2).numpy()
    input_dict = {"A": A, "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Rectangular matrix with negative values, pivot = True
    A = torch.randn(2, 4) * -1.0
    input_dict = {"A": A.numpy(), "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 9: Rectangular matrix with negative values, pivot = True
    A = torch.randn(5, 3) * -1.0
    input_dict = {"A": A.numpy(), "pivot": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def vsplit_inputs():
    list_of_inputs = []
    # Input 1: Splitting into equal sections
    t = torch.arange(16.0).reshape(4, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Splitting at specified indices
    t = torch.arange(16.0).reshape(4, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [1, 3]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: 3D tensor splitting
    t = torch.arange(24.0).reshape(2, 3, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [1]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Uneven splitting with zero-sized tensor
    t = torch.arange(16.0).reshape(4, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [3, 6]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Float tensor
    t = torch.randn(6, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Integer Tensor
    t = torch.randint(0, 10, (5, 5)).numpy()
    input_dict = {"input": t, "indices_or_sections": [2, 4]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 7: 2D tensor with single split
    t = torch.arange(9.0).reshape(3, 3).numpy()
    input_dict = {"input": t, "indices_or_sections": [2]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def ZeroPad2d_inputs():
    list_of_inputs = []
    input_dict_1 = {
        "padding": 2,
        "input": np.random.randn(1, 1, 3, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    input_dict_2 = {
        "padding": (1, 1, 2, 0),
        "input": np.random.randn(1, 1, 3, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    input_dict_3 = {
        "padding": (0, 0, 0, 0),
        "input": np.random.randn(1, 1, 3, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    input_dict_4 = {
        "padding": (1, 2, 3, 4),
        "input": np.random.randn(1, 1, 3, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    input_dict_5 = {
        "padding": (5, 4, 3, 2),
        "input": np.random.randn(1, 1, 3, 3).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    return list_of_inputs
def KLDivLoss_inputs():
    generated_inputs = []
    # Case 1: Basic case with default parameters
    input1 = torch.randn(3, 5).log_softmax(dim=1).numpy()
    target1 = torch.rand(3, 5).softmax(dim=1).numpy()
    input_dict1 = {
        "size_average": None,
        "reduce": None,
        "reduction": "mean",
        "log_target": False
    }
    generated_inputs.append({"input": input1, "target": target1, **input_dict1})
    # Case 2: Different reduction method "batchmean"
    input2 = torch.randn(2, 4, 6).log_softmax(dim=1).numpy()
    target2 = torch.rand(2, 4, 6).softmax(dim=1).numpy()
    input_dict2 = {
        "size_average": None,
        "reduce": None,
        "reduction": "batchmean",
        "log_target": False
    }
    generated_inputs.append({"input": input2, "target": target2, **input_dict2})
    # Case 3: Different reduction method "sum"
    input3 = torch.randn(1, 3, 3, 3).log_softmax(dim=1).numpy()
    target3 = torch.rand(1, 3, 3, 3).softmax(dim=1).numpy()
    input_dict3 = {
        "size_average": None,
        "reduce": None,
        "reduction": "sum",
        "log_target": False
    }
    generated_inputs.append({"input": input3, "target": target3, **input_dict3})
    # Case 4: log_target = True
    input4 = torch.randn(4, 2).log_softmax(dim=1).numpy()
    target4 = torch.rand(4, 2).log_softmax(dim=1).numpy()
    input_dict4 = {
        "size_average": None,
        "reduce": None,
        "reduction": "mean",
        "log_target": True
    }
    generated_inputs.append({"input": input4, "target": target4, **input_dict4})
    # Case 5: reduction = "none"
    input5 = torch.randn(2, 2).log_softmax(dim=1).numpy()
    target5 = torch.rand(2, 2).softmax(dim=1).numpy()
    input_dict5 = {
        "size_average": None,
        "reduce": None,
        "reduction": "none",
        "log_target": False
    }
    generated_inputs.append({"input": input5, "target": target5, **input_dict5})
    
    # Case 6: 1D tensor
    input6 = torch.randn(5).log_softmax(dim=0).numpy()
    target6 = torch.rand(5).softmax(dim=0).numpy()
    input_dict6 = {
        "size_average": None,
        "reduce": None,
        "reduction": "mean",
        "log_target": False
    }
    generated_inputs.append({"input": input6, "target": target6, **input_dict6})
    # Case 7: Different shape
    input7 = torch.randn(1, 2, 3).log_softmax(dim=1).numpy()
    target7 = torch.rand(1, 2, 3).softmax(dim=1).numpy()
    input_dict7 = {
        "size_average": None,
        "reduce": None,
        "reduction": "mean",
        "log_target": False
    }
    generated_inputs.append({"input": input7, "target": target7, **input_dict7})
    return generated_inputs
def set_autocast_ipu_enabled_inputs():
    list_of_inputs = []
    input_dict = {
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def det_inputs():
    list_of_inputs = []
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = np.array([[-1.0, 2.0], [3.0, -4.0]])
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = np.eye(3)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = np.array([[1, 2], [3, 4]], dtype=np.int64)
    A = A.astype(np.float64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = np.random.rand(2, 2, 2)
    input_dict = {"A": A[0]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def triplet_margin_loss_inputs():
    list_of_inputs = []
    # Input 1: Basic case with default parameters
    input_dict = {
        "anchor": np.random.randn(100, 128),
        "positive": np.random.randn(100, 128),
        "negative": np.random.randn(100, 128),
        "margin": 1.0,
        "p": 2,
        "eps": 1e-06,
        "swap": False,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Different margin and p
    input_dict = {
        "anchor": np.random.randn(50, 64),
        "positive": np.random.randn(50, 64),
        "negative": np.random.randn(50, 64),
        "margin": 0.5,
        "p": 1,
        "eps": 1e-06,
        "swap": False,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: With swap enabled
    input_dict = {
        "anchor": np.random.randn(20, 32),
        "positive": np.random.randn(20, 32),
        "negative": np.random.randn(20, 32),
        "margin": 1.0,
        "p": 2,
        "eps": 1e-06,
        "swap": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Different reduction method
    input_dict = {
        "anchor": np.random.randn(80, 16),
        "positive": np.random.randn(80, 16),
        "negative": np.random.randn(80, 16),
        "margin": 1.0,
        "p": 2,
        "eps": 1e-06,
        "swap": False,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 'none' reduction method
    input_dict = {
        "anchor": np.random.randn(120, 256),
        "positive": np.random.randn(120, 256),
        "negative": np.random.randn(120, 256),
        "margin": 1.0,
        "p": 2,
        "eps": 1e-06,
        "swap": False,
        "reduction": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def typename_inputs():
    list_of_inputs = []
    # Input 1: Float tensor
    input_tensor = torch.randn(3, 4).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Int tensor
    input_tensor = torch.randint(0, 10, (2, 2)).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Bool tensor
    input_tensor = torch.tensor([[True, False], [False, True]]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Complex tensor
    input_tensor = torch.complex(torch.randn(2, 3), torch.randn(2, 3)).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Long tensor
    input_tensor = torch.randint(0, 100, (5,)).long().numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Double tensor
    input_tensor = torch.randn(2, 3, dtype=torch.float64).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Zero dimensional tensor (scalar)
    input_tensor = torch.tensor(5.0).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def torch_tile_inputs():
    list_of_inputs = []
    # Example 1: 1D tensor, simple repetition
    input1 = np.array([1, 2, 3])
    dims1 = (2,)
    input_dict1 = {"input": input1, "dims": dims1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Example 2: 2D tensor, repetition in both dimensions
    input2 = np.array([[1, 2], [3, 4]])
    dims2 = (2, 2)
    input_dict2 = {"input": input2, "dims": dims2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Example 3: 3D tensor, repetition in only some dimensions
    input3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dims3 = (1, 2, 1)
    input_dict3 = {"input": input3, "dims": dims3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Example 4: Integer tensor
    input4 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    dims4 = (2, 1)
    input_dict4 = {"input": input4, "dims": dims4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Example 5: Float tensor
    input5 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    dims5 = (1, 2)
    input_dict5 = {"input": input5, "dims": dims5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Example 6: Tensor with negative values
    input6 = np.array([[-1, 2], [3, -4]])
    dims6 = (2, 2)
    input_dict6 = {"input": input6, "dims": dims6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Example 7: Different dimensions
    input7 = np.array([1, 2, 3, 4])
    dims7 = (4,)
    input_dict7 = {"input": input7, "dims": dims7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Example 8: Different dimensions
    input8 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dims8 = (2, 1, 3)
    input_dict8 = {"input": input8, "dims": dims8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Example 9: Different dtype
    input9 = np.array([1, 2, 3], dtype=np.int64)
    dims9 = (3,)
    input_dict9 = {"input": input9, "dims": dims9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    return list_of_inputs
def flatten_inputs():
    list_of_inputs = []
    # Test case 1: Basic 2D tensor
    input1 = torch.randn(2, 3).numpy()
    input_dict1 = {"input": input1, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Test case 2: 3D tensor, flatten from dim 1
    input2 = torch.randn(2, 3, 4).numpy()
    input_dict2 = {"input": input2, "start_dim": 1, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Test case 3: 4D tensor, flatten a middle range of dimensions
    input3 = torch.randn(2, 3, 4, 5).numpy()
    input_dict3 = {"input": input3, "start_dim": 1, "end_dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Test case 4: 1D tensor
    input4 = torch.randn(5).numpy()
    input_dict4 = {"input": input4, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Test case 5: 5D tensor, negative end_dim
    input5 = torch.randn(1, 2, 3, 4, 5).numpy()
    input_dict5 = {"input": input5, "start_dim": 2, "end_dim": -2}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Test case 6: Float tensor
    input6 = torch.randn(2, 3).float().numpy()
    input_dict6 = {"input": input6, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Test case 7: Int tensor
    input7 = torch.randint(0, 10, (2, 3)).int().numpy()
    input_dict7 = {"input": input7, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    # Test case 8: 0D tensor
    input8 = torch.tensor(5).numpy()
    input_dict8 = {"input": input8, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Test case 9: start_dim == end_dim
    input9 = torch.randn(2, 3, 4).numpy()
    input_dict9 = {"input": input9, "start_dim": 1, "end_dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    return list_of_inputs
def get_default_device_inputs():
    list_of_inputs = []
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def arccos__inputs():
    generated_inputs = []
    # Input 1: Basic float tensor within [-1, 1]
    input1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    generated_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Multi-dimensional float tensor
    input2 = np.array([[-0.8, 0.2], [0.7, -0.3]], dtype=np.float64)
    input_dict2 = {"input": input2}
    generated_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Float tensor with values outside [-1, 1] (will result in NaN)
    input3 = np.array([-2.0, 1.5, 0.0], dtype=np.float32)
    input_dict3 = {"input": input3}
    generated_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Float tensor with a single large dimension
    input4 = np.random.uniform(low=-1.0, high=1.0, size=(1, 1000)).astype(np.float32)
    input_dict4 = {"input": input4}
    generated_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Scalar input
    input5 = np.array(0.6, dtype=np.float32)
    input_dict5 = {"input": input5}
    generated_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Negative values close to -1
    input6 = np.array([-0.99, -0.999, -0.9999], dtype=np.float32)
    input_dict6 = {"input": input6}
    generated_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Positive values close to 1
    input7 = np.array([0.99, 0.999, 0.9999], dtype=np.float32)
    input_dict7 = {"input": input7}
    generated_inputs.append(copy.deepcopy(input_dict7))
    
    return generated_inputs
def concatenate_inputs():
    list_of_inputs = []
    # Test case 1: Basic 2D tensors, axis=0
    tensors1 = [np.random.rand(2, 3).astype(np.float32), np.random.rand(3, 3).astype(np.float32)]
    input_dict1 = {"tensors": tensors1, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Test case 2: Basic 2D tensors, axis=1
    tensors2 = [np.random.rand(2, 3).astype(np.float32), np.random.rand(2, 4).astype(np.float32)]
    input_dict2 = {"tensors": tensors2, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Test case 3: 3D tensors, axis=0
    tensors3 = [np.random.rand(2, 3, 4).astype(np.float32), np.random.rand(1, 3, 4).astype(np.float32)]
    input_dict3 = {"tensors": tensors3, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Test case 4: 3D tensors, axis=1
    tensors4 = [np.random.rand(2, 3, 4).astype(np.float32), np.random.rand(2, 2, 4).astype(np.float32)]
    input_dict4 = {"tensors": tensors4, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Test case 5: 3D tensors, axis=2
    tensors5 = [np.random.rand(2, 3, 4).astype(np.float32), np.random.rand(2, 3, 5).astype(np.float32)]
    input_dict5 = {"tensors": tensors5, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Test case 6: Different data types (int32)
    tensors6 = [np.random.randint(0, 10, size=(2, 3), dtype=np.int32), np.random.randint(0, 10, size=(3, 3), dtype=np.int32)]
    input_dict6 = {"tensors": tensors6, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Test case 7: Different data types (int64)
    tensors7 = [np.random.randint(0, 10, size=(2, 3), dtype=np.int64), np.random.randint(0, 10, size=(3, 3), dtype=np.int64)]
    input_dict7 = {"tensors": tensors7, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    # Test case 8: Different shapes and dtypes (float64 and int32)
    tensors8 = [np.random.rand(2, 3).astype(np.float64), np.random.randint(0, 10, size=(2, 3), dtype=np.int32).astype(np.float64)]
    input_dict8 = {"tensors": tensors8, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    # Test case 9:  1D tensors
    tensors9 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    input_dict9 = {"tensors": tensors9, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    return list_of_inputs
def embedding_bag_inputs():
    list_of_inputs = []
    # Example 1: Basic case with sum mode
    input_dict = {
        'input': np.array([1, 2, 4, 5, 4, 3, 0], dtype=np.int64),
        'weight': np.random.rand(7, 3).astype(np.float32),
        'offsets': np.array([0, 4], dtype=np.int64),
        'max_norm': None,
        'norm_type': 2,
        'scale_grad_by_freq': False,
        'mode': 'sum',
        'sparse': False,
        'per_sample_weights': None,
        'include_last_offset': False,
        'padding_idx': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 2: With per_sample_weights and sum mode
    input_dict = {
        'input': np.array([0, 1, 2, 3, 0], dtype=np.int64),
        'weight': np.random.rand(4, 5).astype(np.float32),
        'offsets': np.array([0, 2], dtype=np.int64),
        'max_norm': None,
        'norm_type': 2,
        'scale_grad_by_freq': False,
        'mode': 'sum',
        'sparse': False,
        'per_sample_weights': np.array([0.5, 0.5, 0.5, 0.5, 0.5]).astype(np.float32),
        'include_last_offset': False,
        'padding_idx': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 3: With max mode and max_norm
    input_dict = {
        'input': np.array([0, 1, 2, 3], dtype=np.int64),
        'weight': np.random.rand(4, 4).astype(np.float32),
        'offsets': np.array([0], dtype=np.int64),
        'max_norm': 1.0,
        'norm_type': 2,
        'scale_grad_by_freq': False,
        'mode': 'max',
        'sparse': False,
        'per_sample_weights': None,
        'include_last_offset': False,
        'padding_idx': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 4: With scale_grad_by_freq and padding_idx
    input_dict = {
        'input': np.array([0, 1, 2, 0], dtype=np.int64),
        'weight': np.random.rand(3, 2).astype(np.float32),
        'offsets': np.array([0, 2], dtype=np.int64),
        'max_norm': None,
        'norm_type': 2,
        'scale_grad_by_freq': True,
        'mode': 'sum',
        'sparse': False,
        'per_sample_weights': None,
        'include_last_offset': False,
        'padding_idx': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 5: With include_last_offset
    input_dict = {
        'input': np.array([0, 1, 2, 3, 4, 0], dtype=np.int64),
        'weight': np.random.rand(5, 3).astype(np.float32),
        'offsets': np.array([0, 3, 6], dtype=np.int64),
        'max_norm': None,
        'norm_type': 2,
        'scale_grad_by_freq': False,
        'mode': 'sum',
        'sparse': False,
        'per_sample_weights': None,
        'include_last_offset': True,
        'padding_idx': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def permute_inputs():
    list_of_inputs = []
    input1 = torch.randn(2, 3, 5).numpy()
    dims1 = (2, 0, 1)
    input_dict1 = {"input": input1, "dims": dims1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randint(0, 10, (4, 2, 6, 3)).numpy()
    dims2 = (3, 1, 0, 2)
    input_dict2 = {"input": input2, "dims": dims2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(1, 5, 7, 2, 4).numpy()
    dims3 = (0, 4, 2, 3, 1)
    input_dict3 = {"input": input3, "dims": dims3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(3, 4).numpy()
    dims4 = (1, 0)
    input_dict4 = {"input": input4, "dims": dims4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(10).numpy()
    dims5 = (0,)
    input_dict5 = {"input": input5, "dims": dims5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = torch.randn(2, 2, 2).numpy()
    dims6 = (0, 1, 2)
    input_dict6 = {"input": input6, "dims": dims6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = torch.randn(2, 3, 4, 5).numpy()
    dims7 = (3, 2, 1, 0)
    input_dict7 = {"input": input7, "dims": dims7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.randn(5,).numpy()
    dims8 = (0,)
    input_dict8 = {"input": input8, "dims": dims8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    return list_of_inputs
def softplus_inputs():
    generated_inputs = []
    # Case 1: Default beta and threshold, with input
    input_val = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"input": input_val}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Custom beta, with input
    input_val = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"beta": 2, "input": input_val}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Custom threshold, with input
    input_val = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"threshold": 10, "input": input_val}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Case 4: Custom beta and threshold, with input
    input_val = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"beta": 0.5, "threshold": 30, "input": input_val}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Beta is zero, with input
    input_val = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"beta": 0, "input": input_val}
    generated_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: Negative Beta, with input
    input_val = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"beta": -1, "input": input_val}
    generated_inputs.append(copy.deepcopy(input_dict))
    return generated_inputs
def is_autocast_cpu_enabled_inputs():
    list_of_inputs = []
    list_of_inputs.append({})
    return list_of_inputs
def ifftshift_inputs():
    list_of_inputs = []
    # Case 1: 1D float tensor, default dim
    input1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Case 2: 2D int tensor, dim=0
    input2 = torch.randint(-5, 5, (4, 4)).numpy()
    input_dict2 = {"input": input2, "dim": (0,)}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Case 3: 3D complex tensor, dim=(1,2)
    input3 = torch.randn(2, 3, 4, dtype=torch.complex64).numpy()
    input_dict3 = {"input": input3, "dim": (1, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Case 4: 4D float tensor with negative values, dim=2
    input4 = torch.randn(1, 2, 3, 4).numpy()
    input_dict4 = {"input": input4, "dim": (2,)}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Case 5: 2D float tensor, dim=(0, 1)
    input5 = torch.randn(6, 6).numpy()
    input_dict5 = {"input": input5, "dim": (0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Case 6: 1D int tensor with different size
    input6 = torch.randint(-10, 10, (10,)).numpy()
    input_dict6 = {"input": input6, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Case 7: 3D float tensor, dim=0
    input7 = torch.randn(3, 5, 7).numpy()
    input_dict7 = {"input": input7, "dim": (0,)}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    # Case 8: 2D complex tensor, dim=1
    input8 = torch.randn(4, 5, dtype=torch.complex128).numpy()
    input_dict8 = {"input": input8, "dim": (1,)}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    return list_of_inputs
def fake_quantize_per_tensor_affine_inputs():
    list_of_inputs = []
    # Example 1: Basic float tensor
    input_tensor = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    scale = 0.5
    zero_point = 0
    quant_min = -128
    quant_max = 127
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 2: Int tensor
    input_tensor = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    scale = 0.25
    zero_point = 0
    quant_min = 0
    quant_max = 255
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 3: 2D float tensor
    input_tensor = np.array([[-1.0, -0.5], [0.0, 0.5], [1.0, 1.5]], dtype=np.float32)
    scale = 0.1
    zero_point = 10
    quant_min = 0
    quant_max = 255
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 4: 3D float tensor
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32)
    scale = 0.05
    zero_point = 128
    quant_min = 0
    quant_max = 255
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 5: Negative zero_point and min/max values
    input_tensor = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    scale = 0.2
    zero_point = -5
    quant_min = -100
    quant_max = 100
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 6: Small scale
    input_tensor = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    scale = 0.001
    zero_point = 0
    quant_min = -128
    quant_max = 127
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 7: zero point close to max
    input_tensor = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    scale = 0.1
    zero_point = 250
    quant_min = 0
    quant_max = 255
    input_dict = {"input": input_tensor, "scale": scale, "zero_point": zero_point, "quant_min": quant_min, "quant_max": quant_max}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def set_num_threads_inputs():
    list_of_inputs = []
    input_dict = {
        "num_threads": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "num_threads": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "num_threads": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "num_threads": 8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "num_threads": 16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def local_response_norm_inputs():
    list_of_inputs = []
    size = 3
    alpha = 0.0001
    beta = 0.75
    k = 1.0
    input_data = torch.randn(1, 3, 24, 24).numpy()
    list_of_inputs.append({"size": size, "alpha": alpha, "beta": beta, "k": k, "input": input_data})
    size = 5
    alpha = 0.0002
    beta = 0.5
    k = 2.0
    input_data = torch.randn(1, 5, 12, 12).numpy()
    list_of_inputs.append({"size": size, "alpha": alpha, "beta": beta, "k": k, "input": input_data})
    size = 1
    alpha = 0.00005
    beta = 0.9
    k = 0.5
    input_data = torch.randn(1, 1, 32, 32).numpy()
    list_of_inputs.append({"size": size, "alpha": alpha, "beta": beta, "k": k, "input": input_data})
    size = 7
    alpha = 0.00015
    beta = 0.6
    k = 1.5
    input_data = torch.randn(1, 7, 8, 8).numpy()
    list_of_inputs.append({"size": size, "alpha": alpha, "beta": beta, "k": k, "input": input_data})
    size = 9
    alpha = 0.00025
    beta = 0.4
    k = 2.5
    input_data = torch.randn(1, 9, 16, 16).numpy()
    list_of_inputs.append({"size": size, "alpha": alpha, "beta": beta, "k": k, "input": input_data})
    return list_of_inputs
def column_stack_inputs():
    list_of_inputs = []
    # Input 1: Basic 1D tensors
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Different data types (int and float)
    a = np.array([1, 2, 3])
    b = np.array([4.0, 5.0, 6.0])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: 2D tensors
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Mix of 1D and 2D tensors (valid if broadcastable)
    a = np.array([1, 2])
    b = np.array([[3, 4], [5, 6]])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: 3D tensors with compatible shapes for column stacking
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Tensors with negative values
    a = np.array([-1, -2, -3])
    b = np.array([4, 5, 6])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: More complex 2D arrays
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7, 8, 9], [10, 11, 12]])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def subtract_inputs():
    list_of_inputs = []
    # Case 1: Basic float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = np.array([[0.5, 1.0], [1.5, 2.0]])
    alpha1 = 1.0
    input_dict1 = {"input": input1, "other": input2, "alpha": alpha1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Case 2: Integer tensors with alpha=1
    input3 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input4 = np.array([[0, 1], [1, 2]], dtype=np.int32)
    alpha2 = 1
    input_dict2 = {"input": input3, "other": input4, "alpha": alpha2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Case 3: Negative values and different shapes (broadcasting)
    input5 = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    input6 = np.array([1.0, 2.0])
    alpha3 = 1.0
    input_dict3 = {"input": input5, "other": input6, "alpha": alpha3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Case 4: Higher dimensions (float)
    input7 = np.random.rand(2, 3, 4)
    input8 = np.random.rand(2, 3, 4)
    alpha4 = 0.5
    input_dict4 = {"input": input7, "other": input8, "alpha": alpha4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Case 5: Scalar other (float)
    input9 = np.array([[5.0, 6.0], [7.0, 8.0]])
    input10 = np.array(2.0)
    alpha5 = 1.0
    input_dict5 = {"input": input9, "other": input10, "alpha": alpha5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def is_autocast_ipu_enabled_inputs():
    list_of_inputs = []
    
    list_of_inputs.append({})
    
    return list_of_inputs
def rsqrt_inputs():
    list_of_inputs = []
    # Input 1: 1D float tensor
    input1 = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D float tensor with some small values
    input2 = np.array([[0.25, 1.0], [4.0, 16.0]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 3D float tensor
    input3 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 1D float tensor with a zero
    input4 = np.array([0.0, 1.0, 4.0], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 2D float tensor with a large value
    input5 = np.array([[1e9, 1.0], [4.0, 16.0]], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: 1D float tensor with very small values
    input6 = np.array([1e-6, 1e-3, 1.0], dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs
def reciprocal__inputs():
    list_of_inputs = []
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input5 = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def fix_inputs():
    list_of_inputs = []
    input_float = np.array([1.2, 2.7, -3.1, -4.8]).astype(np.float32)
    input_dict = {"input": input_float}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_int = np.array([1, 2, -3, -4]).astype(np.int32)
    input_dict = {"input": input_int}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_2d = np.array([[1.2, 2.7], [-3.1, -4.8]]).astype(np.float64)
    input_dict = {"input": input_2d}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_3d = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"input": input_3d}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_large = np.array([1000.2, 2000.7, -3000.1, -4000.8]).astype(np.float32)
    input_dict = {"input": input_large}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_zeros = np.array([0.0, 0.0, 0.0]).astype(np.float32)
    input_dict = {"input": input_zeros}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def arctan_inputs():
    list_of_inputs = []
    input1 = np.array([0.0, 1.0, -1.0, 10.0, -10.0])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[0.0, 1.0], [-1.0, 2.0]])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[[0.0, 1.0], [-1.0, 2.0]], [[3.0, -4.0], [5.0, 6.0]]])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([0.5, 1.5, -2.5, 3.5], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([-0.2, 0.7, -1.2, 2.3], dtype=np.float64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([0, 1, -1, 2], dtype=np.int32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([0, 1, -1, 2], dtype=np.int64)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def rsub_inputs():
    list_of_inputs = []
    # Test case 1: Basic float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    other1 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    alpha1 = 1.0
    input_dict1 = {"input": input1, "other": other1, "alpha": alpha1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Test case 2: Integer tensors
    input2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    other2 = np.array([[5, 6], [7, 8]], dtype=np.int32)
    alpha2 = 2
    input_dict2 = {"input": input2, "other": other2, "alpha": alpha2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Test case 3: Negative values
    input3 = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    other3 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    alpha3 = 1.0
    input_dict3 = {"input": input3, "other": other3, "alpha": alpha3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Test case 4: Different shapes (1D tensors)
    input4 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other4 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    alpha4 = 0.5
    input_dict4 = {"input": input4, "other": other4, "alpha": alpha4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Test case 5: Different shapes (3D tensors)
    input5 = np.random.rand(2, 3, 4).astype(np.float32)
    other5 = np.random.rand(2, 3, 4).astype(np.float32)
    alpha5 = 1.5
    input_dict5 = {"input": input5, "other": other5, "alpha": alpha5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Test case 6: Scalar other
    input6 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    other6 = np.array(5.0, dtype=np.float32)
    alpha6 = 1.0
    input_dict6 = {"input": input6, "other": other6, "alpha": alpha6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Test case 7: Scalar input, tensor other
    input7 = np.array(2.0, dtype=np.float32)
    other7 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    alpha7 = 1.0
    input_dict7 = {"input": input7, "other": other7, "alpha": alpha7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    # Test case 8: Complex tensors
    input8 = np.array([[1.0 + 1j, 2.0 - 2j], [3.0, 4.0 + 1j]], dtype=np.complex64)
    other8 = np.array([[5.0, 6.0], [7.0 - 1j, 8.0]], dtype=np.complex64)
    alpha8 = 1.0
    input_dict8 = {"input": input8, "other": other8, "alpha": alpha8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Test case 9: Integer tensors with alpha as integer
    input9 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    other9 = np.array([[5, 6], [7, 8]], dtype=np.int64)
    alpha9 = 2
    input_dict9 = {"input": input9, "other": other9, "alpha": alpha9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    return list_of_inputs
def CosineEmbeddingLoss_inputs():
    generated_inputs = []
    input1 = np.random.randn(3, 5).astype(np.float32)
    input2 = np.random.randn(3, 5).astype(np.float32)
    target = np.array([1, -1, 1]).astype(np.int8)
    input_dict = {'margin': 0.0, 'size_average': None, 'reduce': None, 'reduction': 'mean', 'input1':input1, 'input2':input2, 'target':target}
    generated_inputs.append(copy.deepcopy(input_dict))
    input1 = np.random.randn(2, 4).astype(np.float64)
    input2 = np.random.randn(2, 4).astype(np.float64)
    target = np.array([-1, -1]).astype(np.int8)
    input_dict = {'margin': 0.5, 'size_average': True, 'reduce': True, 'reduction': 'sum', 'input1':input1, 'input2':input2, 'target':target}
    generated_inputs.append(copy.deepcopy(input_dict))
    input1 = np.random.randn(5, 2).astype(np.float32)
    input2 = np.random.randn(5, 2).astype(np.float32)
    target = np.array([1, -1, 1, -1, 1]).astype(np.int8)
    input_dict = {'margin': 0.2, 'size_average': False, 'reduce': True, 'reduction': 'mean', 'input1':input1, 'input2':input2, 'target':target}
    generated_inputs.append(copy.deepcopy(input_dict))
    input1 = np.random.randn(10, 1).astype(np.float64)
    input2 = np.random.randn(10, 1).astype(np.float64)
    target = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1]).astype(np.int8)
    input_dict = {'margin': 0.8, 'size_average': None, 'reduce': False, 'reduction': 'none', 'input1':input1, 'input2':input2, 'target':target}
    generated_inputs.append(copy.deepcopy(input_dict))
    input1 = np.random.randn(1, 7).astype(np.float32)
    input2 = np.random.randn(1, 7).astype(np.float32)
    target = np.array([1]).astype(np.int8)
    input_dict = {'margin': 0.1, 'size_average': True, 'reduce': False, 'reduction': 'sum', 'input1':input1, 'input2':input2, 'target':target}
    generated_inputs.append(copy.deepcopy(input_dict))
    return generated_inputs
def less_equal_inputs():
    list_of_inputs = []
    # Test case 1: Basic float tensors
    input1 = np.array([1.0, 2.0, 3.0])
    other1 = np.array([2.0, 2.0, 1.0])
    list_of_inputs.append({"input": input1, "other": other1})
    # Test case 2: Integer tensors
    input2 = np.array([1, 2, 3], dtype=np.int32)
    other2 = np.array([2, 1, 3], dtype=np.int32)
    list_of_inputs.append({"input": input2, "other": other2})
    # Test case 3: Mixed types (float and int)
    input3 = np.array([1.0, 2.0, 3.0])
    other3 = np.array([2, 1, 3], dtype=np.int32)
    list_of_inputs.append({"input": input3, "other": other3})
    # Test case 4: Negative values
    input4 = np.array([-1.0, -2.0, 3.0])
    other4 = np.array([-2.0, 0.0, 3.0])
    list_of_inputs.append({"input": input4, "other": other4})
    # Test case 5: Multidimensional arrays
    input5 = np.array([[1, 2], [3, 4]])
    other5 = np.array([[2, 1], [4, 3]])
    list_of_inputs.append({"input": input5, "other": other5})
    # Test case 6: Scalar
    input6 = np.array([1, 2, 3])
    other6 = np.array(2)
    list_of_inputs.append({"input": input6, "other": other6})
    # Test case 7: Different shapes (broadcasting) - numpy automatically broadcasts
    input7 = np.array([[1, 2, 3], [4, 5, 6]])
    other7 = np.array([2, 4, 5])
    list_of_inputs.append({"input": input7, "other": other7})
    # Test case 8: Zero values
    input8 = np.array([0.0, 2.0, 0.0])
    other8 = np.array([0.0, 1.0, 1.0])
    list_of_inputs.append({"input": input8, "other": other8})
    
    return list_of_inputs
def torch_select_inputs():
    list_of_inputs = []
    # Input 1: 2D float tensor, dim=0, index=0
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "dim": 0, "index": 0}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 3D int tensor, dim=1, index=1
    input2 = torch.randint(0, 10, (2, 5, 3)).numpy()
    input_dict2 = {"input": input2, "dim": 1, "index": 1}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 4D complex tensor, dim=2, index=-1
    input3 = torch.randn(2, 3, 4, 5, dtype=torch.complex64).numpy()
    input_dict3 = {"input": input3, "dim": 2, "index": -1}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 1D tensor, dim=0, index=0
    input4 = torch.arange(5).numpy()
    input_dict4 = {"input": input4, "dim": 0, "index": 0}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 3D float tensor with negative index
    input5 = torch.randn(4, 6, 8).numpy()
    input_dict5 = {"input": input5, "dim": 1, "index": -2}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: 2D bool tensor
    input6 = torch.randint(0, 2, (3, 4)).bool().numpy()
    input_dict6 = {"input": input6, "dim": 0, "index": 1}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: 5D float tensor
    input7 = torch.randn(2, 3, 4, 5, 6).numpy()
    input_dict7 = {"input": input7, "dim": 3, "index": 2}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def ones_like_inputs():
    list_of_inputs = []
    input1 = np.array([[1, 2], [3, 4]])
    input_dict1 = {
        "input": input1,
        "dtype": None,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
    input_dict2 = {
        "input": input2,
        "dtype": torch.float64,
        "layout": None,
        "requires_grad": True,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([1, 2, 3], dtype=np.int64)
    input_dict3 = {
        "input": input3,
        "dtype": torch.int32,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict4 = {
        "input": input4,
        "dtype": None,
        "layout": None,
        "requires_grad": True,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([1+1j, 2+2j], dtype=np.complex64)
    input_dict5 = {
        "input": input5,
        "dtype": torch.complex128,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([False, True, False])
    input_dict6 = {
        "input": input6,
        "dtype": torch.bool,
        "layout": None,
        "requires_grad": True,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float16)
    input_dict7 = {
        "input": input7,
        "dtype": torch.float16,
        "layout": None,
        "requires_grad": False,
        "memory_format": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def cholesky_inputs():
    list_of_inputs = []
    # Input 1: Basic symmetric positive-definite matrix
    a = np.random.rand(3, 3)
    a = a @ a.T + np.eye(3) * 1e-3
    input_dict = {"input": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Batch of symmetric positive-definite matrices
    a = np.random.rand(2, 2, 2)
    a = a @ np.transpose(a, (0, 2, 1)) + np.eye(2) * 1e-3
    input_dict = {"input": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Larger symmetric positive-definite matrix
    a = np.random.rand(5, 5)
    a = a @ a.T + np.eye(5) * 1e-3
    input_dict = {"input": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Batch of larger matrices
    a = np.random.rand(3, 4, 4)
    a = a @ np.transpose(a, (0, 2, 1)) + np.eye(4) * 1e-3
    input_dict = {"input": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Another valid case
    a = np.random.rand(2, 3, 3)
    a = a @ np.transpose(a, (0, 2, 1)) + np.eye(3) * 1e-3
    input_dict = {"input": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Single element matrix
    a = np.array([[1.0]])
    input_dict = {"input": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def optimize_for_inference_inputs():
    list_of_inputs = []
    # Example 1: Simple float tensor
    mod = torch.jit.script(torch.nn.Linear(10, 5))
    mod = torch.jit.freeze(mod.eval())
    list_of_inputs.append({"mod": mod})
    # Example 2: Int tensor
    class IntModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return x + 1
    mod = torch.jit.script(IntModel())
    mod = torch.jit.freeze(mod.eval())
    list_of_inputs.append({"mod": mod})
    # Example 3: More complex model
    class ComplexModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(5, 10)
            self.linear2 = torch.nn.Linear(10, 5)
        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = torch.sigmoid(self.linear2(x))
            return x
    mod = torch.jit.script(ComplexModel())
    mod = torch.jit.freeze(mod.eval())
    list_of_inputs.append({"mod": mod})
    # Example 4: Module with dropout
    class DropoutModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)
            self.dropout = torch.nn.Dropout(p=0.5)
        def forward(self, x):
            x = self.linear(x)
            x = self.dropout(x)
            return x
    mod = torch.jit.script(DropoutModel())
    mod = torch.jit.freeze(mod.eval())
    list_of_inputs.append({"mod": mod})
    # Example 5: Conv2d module
    class ConvModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, kernel_size=3)
            self.relu = torch.nn.ReLU()
            self.pool = torch.nn.MaxPool2d(2, 2)
        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.pool(x)
            return x
    mod = torch.jit.script(ConvModel())
    mod = torch.jit.freeze(mod.eval())
    list_of_inputs.append({"mod": mod})
    # Example 6: Script module with if-else statement
    @torch.jit.script
    def scripted_fn(x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return x * 2
        else:
            return x / 2
    class IfElseModel(torch.nn.Module):
        def forward(self, x):
            return scripted_fn(x)
    mod = torch.jit.script(IfElseModel())
    mod = torch.jit.freeze(mod.eval())
    list_of_inputs.append({"mod": mod})
    return list_of_inputs
def clone_inputs():
    list_of_inputs = []
    # Example 1: 1D Float Tensor
    input_1 = torch.randn(5).numpy()
    input_dict_1 = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    # Example 2: 2D Int Tensor
    input_2 = torch.randint(-10, 10, (3, 4)).numpy()
    input_dict_2 = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    # Example 3: 3D Complex Tensor
    input_3 = (torch.randn(2, 3, 2) + 1j * torch.randn(2, 3, 2)).numpy()
    input_dict_3 = {"input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    # Example 4: 4D Float Tensor with negative values
    input_4 = torch.randn(1, 3, 2, 2) * -1.0
    input_4 = input_4.numpy()
    input_dict_4 = {"input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Example 5: Scalar Tensor
    input_5 = torch.tensor(5.0).numpy()
    input_dict_5 = {"input": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    # Example 6: Empty Tensor
    input_6 = torch.empty(0).numpy()
    input_dict_6 = {"input": input_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    # Example 7: Bool Tensor
    input_7 = torch.tensor([True, False, True]).numpy()
    input_dict_7 = {"input": input_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    return list_of_inputs
def ZeroPad1d_inputs():
    list_of_inputs = []
    # Test case 1: Integer padding
    input_dict = {"padding": 2, "input": np.random.randn(1, 2, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 2: Tuple padding (different left and right)
    input_dict = {"padding": (3, 1), "input": np.random.randn(1, 2, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 3: Zero padding
    input_dict = {"padding": 0, "input": np.random.randn(1, 2, 5)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 4: Large padding values
    input_dict = {"padding": (10, 5), "input": np.random.randn(1, 2, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 5: Same padding on both sides using a tuple
    input_dict = {"padding": (4, 4), "input": np.random.randn(1, 2, 6)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def ReflectionPad3d_inputs():
    generated_inputs = []
    # Test case 1: int padding
    input1 = torch.arange(8, dtype=torch.float).reshape(1, 1, 2, 2, 2).numpy()
    padding1 = 1
    input_dict1 = {"padding": padding1, "input": input1}
    generated_inputs.append(copy.deepcopy(input_dict1))
    # Test case 2: tuple padding (different on each side)
    input2 = torch.randn(1, 3, 5, 5, 5).numpy()
    padding2 = (1, 2, 0, 1, 2, 0)
    input_dict2 = {"padding": padding2, "input": input2}
    generated_inputs.append(copy.deepcopy(input_dict2))
    # Test case 3: tuple padding (uniform)
    input3 = torch.randn(2, 4, 3, 3, 3).numpy()
    padding3 = (1, 1, 1, 1, 1, 1)
    input_dict3 = {"padding": padding3, "input": input3}
    generated_inputs.append(copy.deepcopy(input_dict3))
    # Test case 4: zero padding
    input4 = torch.randn(1, 1, 4, 4, 4).numpy()
    padding4 = 0
    input_dict4 = {"padding": padding4, "input": input4}
    generated_inputs.append(copy.deepcopy(input_dict4))
    # Test case 5: smaller padding
    input5 = torch.randn(1, 2, 3, 3, 3).numpy()
    padding5 = 1
    input_dict5 = {"padding": padding5, "input": input5}
    generated_inputs.append(copy.deepcopy(input_dict5))
    
    return generated_inputs
def strict_fusion_inputs():
    list_of_inputs = []
    
    return list_of_inputs
def bartlett_window_inputs():
    list_of_inputs = []
    input_dict = {
        "window_length": np.int32(5),
        "periodic": True,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "window_length": np.int64(10),
        "periodic": False,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "window_length": np.int32(7),
        "periodic": True,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "window_length": np.int64(12),
        "periodic": False,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "window_length": np.int32(3),
        "periodic": True,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "window_length": np.int64(8),
        "periodic": False,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def abs__inputs():
    list_of_inputs = []
    # Input 1: 1D float tensor with negative values
    input1 = np.array([-1.0, 2.0, -3.0, 4.0, -5.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D int tensor
    input2 = np.array([[-1, 2], [-3, 4]], dtype=np.int32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 3D float tensor with mixed positive and negative values
    input3 = np.array([[[1.0, -2.0], [3.0, -4.0]], [[-5.0, 6.0], [-7.0, 8.0]]], dtype=np.float64)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 2D tensor with zero values
    input5 = np.array([[-1, 0, 1], [0, -2, 0]], dtype=np.int64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 5: 1D tensor with different ranges of values
    input6 = np.array([-100, 200, -300, 400, -500], dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def fftn_inputs():
    list_of_inputs = []
    # Input 1: Basic example with complex input
    input1 = torch.randn(10, 10, dtype=torch.complex64).numpy()
    input_dict1 = {
        "input": input1,
        "s": None,
        "dim": None,
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Real input with specified dimensions and size
    input2 = torch.randn(5, 5, 5).numpy()
    input_dict2 = {
        "input": input2,
        "s": (10, 10, 5),
        "dim": (0, 1, 2),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Integer input with 'ortho' normalization
    input3 = torch.randint(0, 10, (8, 8)).float().numpy()
    input_dict3 = {
        "input": input3,
        "s": None,
        "dim": None,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 3D input with padding and specific dimensions
    input4 = torch.randn(4, 4, 4).numpy()
    input_dict4 = {
        "input": input4,
        "s": (8, 4, 8),
        "dim": (0, 1, 2),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Complex input with specified size and dim
    input5 = (torch.randn(3, 3) + 1j * torch.randn(3, 3)).numpy()
    input_dict5 = {
        "input": input5,
        "s": (6, 6),
        "dim": (0, 1),
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Float input with normalization
    input6 = torch.randn(2, 2).numpy()
    input_dict6 = {
        "input": input6,
        "s": None,
        "dim": None,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def arcsinh_inputs():
    generated_inputs = []
    # Input 1: Basic float tensor
    input1 = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    generated_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Negative values
    input2 = np.array([-1.0, -2.0, 0.5, -0.5], dtype=np.float64)
    input_dict2 = {"input": input2}
    generated_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Multi-dimensional tensor
    input3 = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    input_dict3 = {"input": input3}
    generated_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Larger values
    input4 = np.array([10.0, 20.0, -10.0, -20.0], dtype=np.float64)
    input_dict4 = {"input": input4}
    generated_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Zero tensor
    input5 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict5 = {"input": input5}
    generated_inputs.append(copy.deepcopy(input_dict5))
    # Input 6:  Different dimension
    input6 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict6 = {"input": input6}
    generated_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: Smaller values
    input7 = np.array([0.1, 0.2, -0.1, -0.2], dtype=np.float64)
    input_dict7 = {"input": input7}
    generated_inputs.append(copy.deepcopy(input_dict7))
    return generated_inputs
def positive_inputs():
    list_of_inputs = []
    # Float tensor
    input_float = torch.randn(5).numpy()
    input_dict = {"input": input_float}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Int tensor
    input_int = torch.randint(-10, 10, (3, 3)).numpy()
    input_dict = {"input": input_int}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Complex tensor
    input_complex = torch.randn(2, 2, dtype=torch.complex64).numpy()
    input_dict = {"input": input_complex}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Multi-dimensional tensor
    input_multidim = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_multidim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Tensor with negative values
    input_negative = torch.randn(4, 4) * -1.0
    input_negative = input_negative.numpy()
    input_dict = {"input": input_negative}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Scalar tensor
    input_scalar = torch.tensor(5.0).numpy()
    input_dict = {"input": input_scalar}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Zero tensor
    input_zero = torch.zeros(2, 2).numpy()
    input_dict = {"input": input_zero}
    list_of_inputs.append(copy.deepcopy(input_dict))
    #Tensor with a mix of positive and negative values
    input_mixed = torch.tensor([-1.0, 2.0, -3.0, 4.0]).numpy()
    input_dict = {"input": input_mixed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def arctan_inputs():
    list_of_inputs = []
    # Input 1: 1D float tensor with positive and negative values
    input1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D float tensor with different values
    input2 = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: scalar value
    input4 = np.array(2.5, dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 4: Large values
    input5 = np.array([100.0, -100.0, 1000.0, -1000.0], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 5: All zeros
    input6 = np.zeros((3, 3), dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def acosh_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensor
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Negative values (should still work as acosh is defined for x >= 1)
    input2 = np.array([1.0, 2.0, 3.0, 10.0], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    # Input 3: 2D float tensor
    input3 = np.array([[1.0, 1.5, 2.0], [2.5, 3.0, 3.5]], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Larger values
    input4 = np.array([100.0, 1000.0, 10000.0], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Scalar input
    input5 = np.array(1.5, dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Zeros (should raise an error but need to include to test edge cases)
    input6 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs
def torch_linalg_eigh_inputs():
    generated_inputs = []
    # Input 1: Real symmetric matrix, UPLO='L' (default)
    A = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    input_dict = {"A": A, "UPLO": 'L'}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Real symmetric matrix, UPLO='U'
    A = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    input_dict = {"A": A, "UPLO": 'U'}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Complex Hermitian matrix, UPLO='L'
    A = np.array([[2.0 + 0.0j, 1.0 - 1.0j], [1.0 + 1.0j, 3.0 + 0.0j]], dtype=np.complex128)
    input_dict = {"A": A, "UPLO": 'L'}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Complex Hermitian matrix, UPLO='U'
    A = np.array([[2.0 + 0.0j, 1.0 - 1.0j], [1.0 + 1.0j, 3.0 + 0.0j]], dtype=np.complex64)
    input_dict = {"A": A, "UPLO": 'U'}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Batch of real symmetric matrices
    A = np.array([[[2.0, 1.0], [1.0, 3.0]], [[4.0, 2.0], [2.0, 5.0]]], dtype=np.float64)
    input_dict = {"A": A, "UPLO": 'L'}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Larger real symmetric matrix
    A = np.array([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]], dtype=np.float64)
    input_dict = {"A": A, "UPLO": 'L'}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Input 7: Batch of complex hermitian matrices with different UPLO
    A = np.array([[[2.0 + 0.0j, 1.0 - 1.0j], [1.0 + 1.0j, 3.0 + 0.0j]], [[4.0 + 0.0j, 2.0 - 2.0j], [2.0 + 2.0j, 5.0 + 0.0j]]], dtype=np.complex128)
    input_dict = {"A": A, "UPLO": 'U'}
    generated_inputs.append(copy.deepcopy(input_dict))
    
    return generated_inputs
def modulelist_inputs():
    list_of_inputs = []
    # Case 1: Empty ModuleList
    input_dict = {
        "modules": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: ModuleList with Linear layers
    input_dict = {
        "modules": [nn.Linear(10, 20), nn.Linear(20, 30)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: ModuleList with Conv2d and ReLU
    input_dict = {
        "modules": [nn.Conv2d(3, 16, kernel_size=3), nn.ReLU()]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: ModuleList with BatchNorm1d and Dropout
    input_dict = {
        "modules": [nn.BatchNorm1d(100), nn.Dropout(0.5)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: ModuleList with a mix of different module types
    input_dict = {
        "modules": [nn.Linear(5, 10), nn.ReLU(), nn.Conv1d(1, 3, kernel_size=3)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 6: ModuleList with Sequential
    input_dict = {
        "modules": [nn.Sequential(nn.Linear(10, 5), nn.ReLU())]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 7: ModuleList with nested ModuleList
    input_dict = {
        "modules": [nn.ModuleList([nn.Linear(5,5), nn.ReLU()]), nn.Linear(10, 10)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def count_nonzero_inputs():
    list_of_inputs = []
    input_np = np.array([
        [1, 0, 2, 0],
        [0, 3, 0, 4]
    ])
    input_dict = {"input": input_np, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_np = np.array([
        [0, 0, 0],
        [0, 0, 0]
    ])
    input_dict = {"input": input_np, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_np = np.array([
        [1.0, 0.0, 2.5],
        [0.0, -3.2, 0.0]
    ])
    input_dict = {"input": input_np, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_np = np.array([
        [[1, 0], [0, 2]],
        [[3, 0], [0, 4]]
    ])
    input_dict = {"input": input_np, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_np = np.array([
        1, 2, 3, 4, 5
    ])
    input_dict = {"input": input_np, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_np = np.array([
        [1, 0, 2, 0],
        [0, 3, 0, 4]
    ])
    input_dict = {"input": input_np, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_np = np.array([
        [1, 0, 2, 0],
        [0, 3, 0, 4]
    ])
    input_dict = {"input": input_np, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_np = np.array([
        [[1, 0], [0, 2]],
        [[3, 0], [0, 4]]
    ])
    input_dict = {"input": input_np, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_np = np.array([
        [[1, 0], [0, 2]],
        [[3, 0], [0, 4]]
    ])
    input_dict = {"input": input_np, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_np = np.array([
        [[1, 0], [0, 2]],
        [[3, 0], [0, 4]]
    ])
    input_dict = {"input": input_np, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def hamming_window_inputs():
    generated_inputs = []
    input_dict = {
        "window_length": 5,
        "periodic": False,
        "alpha": 0.5,
        "beta": 0.5,
        "dtype": torch.float32
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "window_length": 10,
        "periodic": True,
        "alpha": 0.5,
        "beta": 0.5,
        "dtype": torch.float64
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "window_length": 7,
        "periodic": False,
        "alpha": 0.0,
        "beta": 1.0,
        "dtype": torch.float32
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "window_length": 12,
        "periodic": True,
        "alpha": 1.0,
        "beta": 0.0,
        "dtype": torch.float64
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "window_length": 3,
        "periodic": False,
        "alpha": 0.25,
        "beta": 0.75,
        "dtype": torch.float32
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "window_length": 8,
        "periodic": True,
        "alpha": 0.75,
        "beta": 0.25,
        "dtype": torch.float64
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "window_length": 15,
        "periodic": False,
        "alpha": 0.6,
        "beta": 0.4,
        "dtype": torch.float32
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    
    return generated_inputs
def set_num_interop_threads_inputs():
    list_of_inputs = []
    input_dict = {
        "num_threads": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def arctanh_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensor
    input1 = np.array([0.1, 0.5, -0.2, 0.8]).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Multi-dimensional float tensor with values near 1 and -1 (but not equal)
    input2 = np.array([[0.9, -0.95], [0.7, -0.65]]).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Float tensor with a wider range of values (still within -1 and 1)
    input3 = np.linspace(-0.99, 0.99, 10).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Higher dimensional float tensor
    input4 = np.random.uniform(-0.9, 0.9, size=(2, 3, 4)).astype(np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Single element float tensor
    input5 = np.array([0.5]).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Another set of different float values
    input6 = np.array([-0.3, 0.6, -0.8, 0.2]).astype(np.float64)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: A 3D tensor with more diverse values
    input7 = np.random.uniform(-0.7, 0.7, size=(3, 2, 2)).astype(np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def multigammaln_inputs():
    list_of_inputs = []
    input1 = np.random.rand(5).astype(np.float32)
    p1 = 3
    safe1 = True
    input_dict1 = {"input": input1, "p": p1, "safe": safe1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.random.rand(2, 3).astype(np.float64)
    p2 = 2
    safe2 = False
    input_dict2 = {"input": input2, "p": p2, "safe": safe2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = (np.random.rand(3, 2) + 1j*np.random.rand(3, 2)).astype(np.complex64)
    p3 = 1
    safe3 = True
    input_dict3 = {"input": input3, "p": p3, "safe": safe3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = (np.random.rand(4) + 1j*np.random.rand(4)).astype(np.complex128)
    p4 = 4
    safe4 = False
    input_dict4 = {"input": input4, "p": p4, "safe": safe4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([1.5, 2.5, 3.5, 4.5]).astype(np.float32)
    p5 = 2
    safe5 = True
    input_dict5 = {"input": input5, "p": p5, "safe": safe5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.random.rand(2, 2, 2).astype(np.float64)
    p6 = 3
    safe6 = False
    input_dict6 = {"input": input6, "p": p6, "safe": safe6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([[2.0, 3.0], [4.0, 5.0]]).astype(np.float32)
    p7 = 1
    safe7 = True
    input_dict7 = {"input": input7, "p": p7, "safe": safe7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def view_as_real_inputs():
    list_of_inputs = []
    # Input 1: Simple 2D float tensor
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 3D float tensor
    input2 = torch.randn(2, 3, 5).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Complex 2D tensor
    input3 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Complex 3D tensor
    input4 = torch.randn(3, 1, 2, dtype=torch.complex128).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Larger float tensor with different dimensions
    input5 = torch.randn(4, 5, 2, 3).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Float Tensor with size 1
    input6 = torch.randn(1, 1, 1).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: Float Tensor with zeros
    input7 = torch.zeros(2, 3).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def is_grad_enabled_inputs():
    list_of_inputs = []
    list_of_inputs.append({})
    
    return list_of_inputs
def select_copy_inputs():
    list_of_inputs = []
    # Case 1: Basic 2D float tensor
    input_tensor = torch.randn(3, 4).numpy()
    dim = 0
    index = 1
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: 3D int tensor, negative index
    input_tensor = torch.randint(-5, 5, (2, 3, 5)).numpy()
    dim = 1
    index = -1
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: 4D complex tensor
    input_tensor = torch.randn(2, 2, 2, 2, dtype=torch.complex64).numpy()
    dim = 2
    index = 0
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: 1D tensor
    input_tensor = torch.arange(5).float().numpy()
    dim = 0
    index = 2
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Large 3D tensor
    input_tensor = torch.randn(10, 20, 30).numpy()
    dim = 0
    index = 5
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 6: Boolean tensor
    input_tensor = torch.randint(0, 2, (2, 3), dtype=torch.bool).numpy()
    dim = 1
    index = 0
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 7: Double tensor
    input_tensor = torch.randn(3, 3, dtype=torch.float64).numpy()
    dim = 0
    index = 1
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def autocast_increment_nesting_inputs():
    list_of_inputs = []
    
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def native_dropout_inputs():
    list_of_inputs = []
    # Test case 1: Simple 2D float tensor
    input1 = torch.randn(5, 5).numpy()
    input_dict1 = {
        "input": input1,
        "p": 0.5,
        "train": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Test case 2: 3D integer tensor with dropout probability 0 (converted to float)
    input2 = torch.randint(0, 10, (3, 4, 5)).float().numpy()
    input_dict2 = {
        "input": input2,
        "p": 0.0,
        "train": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Test case 3: 1D float tensor with dropout probability 1
    input3 = torch.randn(10).numpy()
    input_dict3 = {
        "input": input3,
        "p": 1.0,
        "train": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Test case 4: 4D float tensor with dropout probability 0.2, train = False
    input4 = torch.randn(2, 3, 4, 5).numpy()
    input_dict4 = {
        "input": input4,
        "p": 0.2,
        "train": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Test case 5: 2D complex tensor (converted to float)
    input5 = (torch.randn(5, 5) + 1j * torch.randn(5, 5)).abs().numpy()
    input_dict5 = {
        "input": input5,
        "p": 0.3,
        "train": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def floor_inputs():
    list_of_inputs = []
    input1 = np.array([1.2, 2.7, -3.4, -0.5])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[1.5, 2.3], [3.8, -4.1]])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[[1.9, 2.1], [3.5, 4.6]], [[5.2, 6.8], [-7.4, 8.3]]])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([5])
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([-5.0])
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([0.0])
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def cross_entropy_loss_inputs():
    generated_inputs = []
    # Case 1: Class indices, no weight, default reduction, no label smoothing
    input_tensor = torch.randn(3, 5).numpy()
    target_tensor = torch.randint(0, 5, (3,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor, "weight": None, "ignore_index": -100, "reduction": "mean", "label_smoothing": 0.0}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Class probabilities, no weight, sum reduction, label smoothing
    input_tensor = torch.randn(3, 5).numpy()
    target_tensor = torch.randn(3, 5).softmax(dim=1).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor, "weight": None, "ignore_index": -100, "reduction": "sum", "label_smoothing": 0.1}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Class indices, with weight, none reduction, ignore_index
    input_tensor = torch.randn(2, 4).numpy()
    target_tensor = torch.randint(0, 4, (2,)).numpy()
    weight_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4]).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor, "weight": weight_tensor, "ignore_index": 1, "reduction": "none", "label_smoothing": 0.0}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Case 4: K-dimensional input, class indices, mean reduction
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    target_tensor = torch.randint(0, 3, (2, 4, 5)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor, "weight": None, "ignore_index": -100, "reduction": "mean", "label_smoothing": 0.0}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Unbatched input, class indices
    input_tensor = torch.randn(5).numpy()
    target_tensor = torch.randint(0, 5, ()).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor, "weight": None, "ignore_index": -100, "reduction": "mean", "label_smoothing": 0.0}
    generated_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: Class probabilities, with weight, none reduction, label smoothing
    input_tensor = torch.randn(2, 3).numpy()
    target_tensor = torch.randn(2, 3).softmax(dim=1).numpy()
    weight_tensor = torch.tensor([0.1, 0.2, 0.3]).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor, "weight": weight_tensor, "ignore_index": -100, "reduction": "none", "label_smoothing": 0.2}
    generated_inputs.append(copy.deepcopy(input_dict))
    return generated_inputs
def prepare_for_loss(input_dict):
    new_input_dict = {}
    new_input_dict['input'] = torch.tensor(input_dict['input'])
    new_input_dict['target'] = torch.tensor(input_dict['target'])
    if input_dict['weight'] is not None:
        new_input_dict['weight'] = torch.tensor(input_dict['weight'])
    else:
        new_input_dict['weight'] = None
    new_input_dict['ignore_index'] = input_dict['ignore_index']
    new_input_dict['reduction'] = input_dict['reduction']
    new_input_dict['label_smoothing'] = input_dict['label_smoothing']
    return new_input_dict
    
def log_softmax_inputs():
    list_of_inputs = []
    # Input 1: 2D tensor, dim=1
    input1 = np.random.randn(2, 3).astype(np.float32)
    input_dict1 = {"dim": 1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 3D tensor, dim=0
    input2 = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict2 = {"dim": 0, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 2D tensor with negative values, dim=0
    input3 = np.random.randn(5, 5).astype(np.float32) * -1
    input_dict3 = {"dim": 0, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 1D tensor, dim=0
    input4 = np.random.randn(10).astype(np.float32)
    input_dict4 = {"dim": 0, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 4D tensor, dim=2
    input5 = np.random.randn(2, 3, 4, 5).astype(np.float64)
    input_dict5 = {"dim": 2, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: 3D tensor, dim=1
    input6 = np.random.randn(3, 5, 2).astype(np.float32)
    input_dict6 = {"dim": 1, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def rfft_inputs():
    generated_inputs = []
    # Test case 1: Basic test with a 1D tensor
    input1 = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict1 = {"input": input1, "n": None, "dim": -1, "norm": None}
    generated_inputs.append(copy.deepcopy(input_dict1))
    # Test case 2: 2D tensor with specified dim
    input2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    input_dict2 = {"input": input2, "n": None, "dim": 0, "norm": None}
    generated_inputs.append(copy.deepcopy(input_dict2))
    # Test case 3: Specify n (signal length) - padding
    input3 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict3 = {"input": input3, "n": 5, "dim": -1, "norm": None}
    generated_inputs.append(copy.deepcopy(input_dict3))
    # Test case 4: Specify n (signal length) - trimming
    input4 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    input_dict4 = {"input": input4, "n": 3, "dim": -1, "norm": None}
    generated_inputs.append(copy.deepcopy(input_dict4))
    # Test case 5: With normalization
    input5 = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict5 = {"input": input5, "n": None, "dim": -1, "norm": "forward"}
    generated_inputs.append(copy.deepcopy(input_dict5))
    # Test case 6: With different normalization
    input6 = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict6 = {"input": input6, "n": None, "dim": -1, "norm": "ortho"}
    generated_inputs.append(copy.deepcopy(input_dict6))
    
    # Test case 7: With negative values
    input7 = torch.tensor([-1.0, 2.0, -3.0, 4.0]).numpy()
    input_dict7 = {"input": input7, "n": None, "dim": -1, "norm": None}
    generated_inputs.append(copy.deepcopy(input_dict7))
    # Test case 8: 3D tensor
    input8 = torch.randn(2, 3, 4).numpy()
    input_dict8 = {"input": input8, "n": None, "dim": 1, "norm": None}
    generated_inputs.append(copy.deepcopy(input_dict8))
    
    # Test case 9: float64 tensor
    input9 = torch.randn(4, dtype=torch.float64).numpy()
    input_dict9 = {"input": input9, "n": None, "dim": -1, "norm": None}
    generated_inputs.append(copy.deepcopy(input_dict9))
    
    # Test case 10: int64 tensor
    input10 = torch.randint(0, 10, (4,), dtype=torch.int64).numpy()
    input_dict10 = {"input": input10, "n": None, "dim": -1, "norm": None}
    generated_inputs.append(copy.deepcopy(input_dict10))
    return generated_inputs
def vitals_enabled_inputs():
    list_of_inputs = []
    
    # Input is not actually used, but driver expects a dictionary with a valid 'input' key.
    input1 = {"input": np.array([1.0])}
    list_of_inputs.append(copy.deepcopy(input1))
    
    return list_of_inputs
def is_anomaly_check_nan_enabled_inputs():
    list_of_inputs = []
    
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def negative_inputs():
    list_of_inputs = []
    input1 = np.array([1, 2, 3], dtype=np.int32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[1, -2], [-3, 4]], dtype=np.int64)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([-1, 0, 1], dtype=np.int8)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([1, 2, 3], dtype=np.uint8)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float16)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    input8 = np.array([1 + 1j, 2 - 2j], dtype=np.complex64)
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([1 + 1j, 2 - 2j], dtype=np.complex128)
    input_dict9 = {"input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    return list_of_inputs
def torch_stft_inputs():
    list_of_inputs = []
    # Input 1: Basic example with real input
    input1 = torch.randn(1000).numpy()
    n_fft1 = 256
    hop_length1 = 64
    win_length1 = 256
    window1 = torch.hann_window(win_length1).numpy()
    return_complex1 = True
    input_dict1 = {
        "input": input1,
        "n_fft": n_fft1,
        "hop_length": hop_length1,
        "win_length": win_length1,
        "window": window1,
        "center": True,
        "pad_mode": "reflect",
        "normalized": False,
        "onesided": True,
        "return_complex": return_complex1,
        "align_to_window": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Complex input
    input2 = torch.randn(512, dtype=torch.complex64).numpy()
    n_fft2 = 128
    hop_length2 = 32
    win_length2 = 128
    window2 = torch.hamming_window(win_length2).numpy()
    return_complex2 = True
    input_dict2 = {
        "input": input2,
        "n_fft": n_fft2,
        "hop_length": hop_length2,
        "win_length": win_length2,
        "window": window2,
        "center": False,
        "pad_mode": "constant",
        "normalized": True,
        "onesided": False,
        "return_complex": return_complex2,
        "align_to_window": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Batch input (2D)
    input3 = torch.randn(2, 768).numpy()
    n_fft3 = 128
    hop_length3 = 64
    win_length3 = 128
    window3 = torch.bartlett_window(win_length3).numpy()
    return_complex3 = True
    input_dict3 = {
        "input": input3,
        "n_fft": n_fft3,
        "hop_length": hop_length3,
        "win_length": win_length3,
        "window": window3,
        "center": True,
        "pad_mode": "replicate",
        "normalized": False,
        "onesided": True,
        "return_complex": return_complex3,
        "align_to_window": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: No window
    input4 = torch.randn(2048).numpy()
    n_fft4 = 512
    hop_length4 = 128
    win_length4 = 512
    return_complex4 = True
    input_dict4 = {
        "input": input4,
        "n_fft": n_fft4,
        "hop_length": hop_length4,
        "win_length": win_length4,
        "window": None,
        "center": False,
        "pad_mode": "circular",
        "normalized": True,
        "onesided": False,
        "return_complex": return_complex4,
        "align_to_window": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Different parameters
    input5 = torch.randn(1, 500).numpy()
    n_fft5 = 256
    hop_length5 = 128
    win_length5 = 128
    window5 = torch.blackman_window(win_length5).numpy()
    return_complex5 = True
    input_dict5 = {
        "input": input5,
        "n_fft": n_fft5,
        "hop_length": hop_length5,
        "win_length": win_length5,
        "window": window5,
        "center": True,
        "pad_mode": "reflect",
        "normalized": False,
        "onesided": True,
        "return_complex": return_complex5,
        "align_to_window": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Negative values in input
    input6 = (torch.rand(500) - 0.5).numpy()
    n_fft6 = 128
    hop_length6 = 64
    win_length6 = 128
    window6 = torch.hann_window(win_length6).numpy()
    return_complex6 = True
    input_dict6 = {
        "input": input6,
        "n_fft": n_fft6,
        "hop_length": hop_length6,
        "win_length": win_length6,
        "window": window6,
        "center": True,
        "pad_mode": "reflect",
        "normalized": False,
        "onesided": True,
        "return_complex": return_complex6,
        "align_to_window": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def absolute_inputs():
    list_of_inputs = []
    input1 = np.array([-1, -2, 3, 4, -5], dtype=np.int32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([-1.5, -2.5, 3.5, 4.5, -5.5], dtype=np.float32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[-1, 2], [-3, 4]], dtype=np.int64)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([[-1.0 + 1j, 2.0 - 2j], [-3.0 + 3j, 4.0 - 4j]], dtype=np.complex128)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([[[1, -2], [3, -4]], [[-5, 6], [-7, 8]]], dtype=np.int16)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def index_copy_inputs():
    list_of_inputs = []
    # Case 1: Basic 1D tensor
    input_tensor = torch.randn(5).numpy()
    index_tensor = torch.tensor([0, 2, 4]).numpy()
    source_tensor = torch.randn(3).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: 2D tensor
    input_tensor = torch.randn(3, 4).numpy()
    index_tensor = torch.tensor([0, 2]).numpy()
    source_tensor = torch.randn(2, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Different dim to copy along
    input_tensor = torch.randn(3, 4).numpy()
    index_tensor = torch.tensor([1, 3]).numpy()
    source_tensor = torch.randn(3, 2).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: 3D tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    index_tensor = torch.tensor([0, 1]).numpy()
    source_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Integer input tensor
    input_tensor = torch.randint(0, 10, (3, 4)).numpy()
    index_tensor = torch.tensor([0, 1]).numpy()
    source_tensor = torch.randint(0, 10, (2, 4)).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 6: Float input tensor, dim=1, different source shape to test exceptions
    input_tensor = torch.randn(5, 5).numpy()
    index_tensor = torch.tensor([0, 2, 4]).numpy()
    source_tensor = torch.randn(5, 3).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "index": index_tensor,
        "source": source_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def greater_inputs():
    list_of_inputs = []
    input1 = np.array([1, 2, 3], dtype=np.int32)
    other1 = np.array([0, 2, 4], dtype=np.int32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    other2 = np.array([[2.0, 1.0], [4.0, 3.0]], dtype=np.float32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([-1, -2, -3], dtype=np.int64)
    other3 = np.array([0, -2, -4], dtype=np.int64)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([1, 2, 3, 4]).reshape(2, 2).astype(np.float64)
    other4 = 2.0
    input_dict4 = {"input": input4, "other": np.array(other4)}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    other5 = np.array([5, 4, 3, 2, 1], dtype=np.int16)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([5], dtype=np.uint8)
    other6 = np.array([3], dtype=np.uint8)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([1, 2, 3, 4, 5, 6]).reshape(1, 2, 3).astype(np.int8)
    other7 = np.array([6, 5, 4, 3, 2, 1]).reshape(1, 2, 3).astype(np.int8)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    input8 = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    other8 = np.array([0.3, 0.2, 0.1], dtype=np.float16)
    input_dict8 = {"input": input8, "other": other8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs
def as_strided_copy_inputs():
    list_of_inputs = []
    # Case 1: Basic 2D float tensor
    input1 = torch.randn(5, 5).numpy()
    size1 = (3, 3)
    stride1 = (1, 1)
    input_dict1 = {"input": input1, "size": size1, "stride": stride1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Case 2: 3D int tensor
    input2 = torch.randint(0, 10, (4, 4, 4)).numpy()
    size2 = (2, 2, 2)
    stride2 = (1, 1, 2)
    input_dict2 = {"input": input2, "size": size2, "stride": stride2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Case 3: 1D tensor
    input3 = torch.arange(10).float().numpy()
    size3 = (5,)
    stride3 = (2,)
    input_dict3 = {"input": input3, "size": size3, "stride": stride3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Case 4: Different size and stride combination for 2D tensor
    input4 = torch.randn(7, 7).numpy()
    size4 = (4, 4)
    stride4 = (2, 1)
    input_dict4 = {"input": input4, "size": size4, "stride": stride4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Case 5: Larger tensor and size with complex strides
    input5 = torch.randn(10, 10, 10).numpy()
    size5 = (5, 5, 5)
    stride5 = (3, 2, 1)
    input_dict5 = {"input": input5, "size": size5, "stride": stride5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def get_total_norm_inputs():
    list_of_inputs = []
    # Input 1: List of float tensors, norm_type=2.0
    tensor_list_1 = [torch.randn(3, 4).numpy(), torch.randn(5, 2).numpy()]
    input_dict_1 = {"parameters": tensor_list_1, "norm_type": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    # Input 2: List of float tensors, norm_type=1.0
    tensor_list_2 = [torch.randn(2, 2).numpy(), torch.randn(3,).numpy()]
    input_dict_2 = {"parameters": tensor_list_2, "norm_type": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    # Input 3: List of tensors with different shapes, norm_type=float('inf')
    tensor_list_3 = [torch.randn(1, 2, 3).numpy(), torch.randn(4).numpy(), torch.randn(2, 1).numpy()]
    input_dict_3 = {"parameters": tensor_list_3, "norm_type": float('inf')}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    # Input 4: List containing single tensor, norm_type=0.5
    tensor_list_4 = [torch.randn(10).numpy()]
    input_dict_4 = {"parameters": tensor_list_4, "norm_type": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    # Input 5: List of tensors with negative values, norm_type=3.0
    tensor_list_5 = [torch.randn(2, 3).numpy() * -1, torch.randn(4).numpy() * -1]
    input_dict_5 = {"parameters": tensor_list_5, "norm_type": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    # Input 6: Empty list of tensors, norm_type=2.0
    tensor_list_6 = []
    input_dict_6 = {"parameters": tensor_list_6, "norm_type": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    return list_of_inputs
def generate_divide_inputs():
    generated_inputs = []
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other1 = np.array([0.5, 2.0, 1.5], dtype=np.float32)
    input_dict1 = {"input": input1, "other": other1, "rounding_mode": None}
    generated_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    other2 = np.array([[2, 1], [4, 2]], dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2, "rounding_mode": "floor"}
    generated_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    other3 = np.array([0.5, -1.0, 1.5], dtype=np.float64)
    input_dict3 = {"input": input3, "other": other3, "rounding_mode": "trunc"}
    generated_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([1, 2, 3], dtype=np.int64)
    other4 = 2
    input_dict4 = {"input": input4, "other": other4, "rounding_mode": None}
    generated_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    other5 = 1.2
    input_dict5 = {"input": input5, "other": other5, "rounding_mode": "floor"}
    generated_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([1, 2, 3], dtype=np.int32)
    other6 = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    input_dict6 = {"input": input6, "other": other6, "rounding_mode": None}
    generated_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([1, 2, 3], dtype=np.int32)
    other7 = 2
    input_dict7 = {"input": input7, "other": other7, "rounding_mode": "trunc"}
    generated_inputs.append(copy.deepcopy(input_dict7))
    return generated_inputs
import io
def torch_save_inputs():
    list_of_inputs = []
    # Input 1: Simple float tensor
    x = torch.tensor([0.1, 1.2, 2.3, 3.4, 4.5]).numpy()
    f = "tensor1.pt"
    input_dict = {"obj": x, "f": f}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Integer tensor with negative values
    x = torch.tensor([-1, 0, 1, 2, -3]).numpy()
    f = "tensor2.pt"
    input_dict = {"obj": x, "f": f}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: 2D tensor
    x = torch.randn(2, 3).numpy()
    f = "tensor3.pt"
    input_dict = {"obj": x, "f": f}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: 3D tensor
    x = torch.randn(2, 3, 4).numpy()
    f = "tensor4.pt"
    input_dict = {"obj": x, "f": f}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Complex tensor
    x = torch.randn(2, 2, dtype=torch.complex64).numpy()
    f = "tensor5.pt"
    input_dict = {"obj": x, "f": f}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Bool tensor
    x = torch.tensor([True, False, True]).numpy()
    f = "tensor6.pt"
    input_dict = {"obj": x, "f": f}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 7: Zero dimensional tensor (scalar)
    x = torch.tensor(5.0).numpy()
    f = "tensor7.pt"
    input_dict = {"obj": x, "f": f}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def view_as_complex_copy_inputs():
    list_of_inputs = []
    # Input 1: Basic 2D float tensor
    input1 = torch.randn(2, 2, 2).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Basic 3D float tensor with different dimensions
    input2 = torch.randn(3, 4, 2).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 1D float tensor
    input3 = torch.randn(4,2).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Larger tensor
    input4 = torch.randn(5, 2, 5, 2).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Tensor with negative values
    input5 = (torch.randn(2, 2, 2) * -1).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Tensor with small values
    input6 = (torch.rand(2, 2, 2) * 0.1).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def gaussian_nllloss_inputs():
    list_of_inputs = []
    # Case 1: Basic case with mean reduction
    input1 = torch.randn(5, 2).numpy()
    target1 = torch.randn(5, 2).numpy()
    var1 = torch.ones(5, 2).numpy()
    input_dict1 = {
        "input": input1,
        "target": target1,
        "var": var1,
        "full": False,
        "eps": 1e-6,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Case 2: Sum reduction with different eps
    input2 = torch.randn(3, 4).numpy()
    target2 = torch.randn(3, 4).numpy()
    var2 = torch.rand(3, 4).numpy()
    input_dict2 = {
        "input": input2,
        "target": target2,
        "var": var2,
        "full": True,
        "eps": 1e-4,
        "reduction": "sum"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Case 3: No reduction
    input3 = torch.randn(2, 2, 2).numpy()
    target3 = torch.randn(2, 2, 2).numpy()
    var3 = torch.rand(2, 2, 2).numpy()
    input_dict3 = {
        "input": input3,
        "target": target3,
        "var": var3,
        "full": False,
        "eps": 1e-6,
        "reduction": "none"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Case 4: Broadcasting var
    input4 = torch.randn(4, 5).numpy()
    target4 = torch.randn(4, 5).numpy()
    var4 = torch.ones(4, 1).numpy()
    input_dict4 = {
        "input": input4,
        "target": target4,
        "var": var4,
        "full": True,
        "eps": 1e-5,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Case 5: Scalar var
    input5 = torch.randn(10).numpy()
    target5 = torch.randn(10).numpy()
    var5 = np.array(2.0)
    input_dict5 = {
        "input": input5,
        "target": target5,
        "var": var5,
        "full": False,
        "eps": 1e-7,
        "reduction": "sum"
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def vstack_inputs():
    list_of_inputs = []
    # Test case 1: Two 1D tensors
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 2: Two 2D tensors
    a = np.array([[1], [2], [3]])
    b = np.array([[4], [5], [6]])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 3: Multiple tensors with the same shape (after atleast_2d)
    a = np.array([1, 2])
    b = np.array([3, 4])
    c = np.array([5, 6])
    input_dict = {"tensors": [a, b, c]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 4: Three 2D tensors with the same number of columns
    a = np.array([[1, 2, 3]])
    b = np.array([[4, 5, 6]])
    c = np.array([[7, 8, 9]])
    input_dict = {"tensors": [a, b, c]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 5: Tensors with different data types (mixed int and float) - make sure they have the same shape
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 6:  Tensors with negative values
    a = np.array([-1, -2, -3])
    b = np.array([4, 5, 6])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Test case 7: Three dimensional tensors with matching shapes
    a = np.array([[[1, 2], [3, 4]]])
    b = np.array([[[5, 6], [7, 8]]])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def CompilationUnit_inputs():
    list_of_inputs = []
    # Input 1: Simple function definition
    input1 = """
    def foo(x):
        return x + 1
    """
    list_of_inputs.append({"source": input1, "input": torch.randn(1).numpy()})
    # Input 2: Function with type hints
    input2 = """
    def bar(x: int) -> int:
        return x * 2
    """
    list_of_inputs.append({"source": input2, "input": torch.randint(0, 10, (1,)).numpy()})
    # Input 3: Class definition with a method
    input3 = """
    class MyClass:
        def __init__(self, value: float):
            self.value = value
        def get_value(self) -> float:
            return self.value
    """
    list_of_inputs.append({"source": input3, "input": torch.randn(1, 1).numpy()})
    # Input 4: Multiple functions and classes
    input4 = """
    def add(x: int, y: int) -> int:
        return x + y
    class Point:
        def __init__(self, x: float, y: float):
            self.x = x
            self.y = y
        def distance_from_origin(self) -> float:
            return (self.x**2 + self.y**2)**0.5
    """
    list_of_inputs.append({"source": input4, "input": torch.randn(2, 2).numpy()})
    # Input 5: A function that uses torch
    input5 = """
    
    def create_tensor(size: int) -> torch.Tensor:
        return torch.randn(size)
    """
    list_of_inputs.append({"source": input5, "input": torch.randn(3, 3).numpy()})
    return list_of_inputs
def argwhere_inputs():
    list_of_inputs = []
    # Input 1: 1D integer tensor
    input1 = np.array([1, 0, 2, 0, 3])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D float tensor
    input2 = np.array([[1.0, 0.0, 2.5], [0.0, -1.0, 3.2]])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 3D integer tensor with negative values
    input3 = np.array([[[1, 0, -1], [0, 2, 0]], [[-2, 0, 3], [0, -1, 0]]])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 1D boolean tensor
    input4 = np.array([True, False, True, False, True])
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 2D complex tensor
    input5 = np.array([[1+1j, 0, 2-2j], [0, -1+0j, 3+1j]])
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Empty array
    input6 = np.array([])
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: All zeros
    input7 = np.zeros((2,3))
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: 4D array
    input8 = np.random.randint(-5, 5, size=(2, 2, 2, 2))
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    return list_of_inputs
def Sigmoid_inputs():
    generated_inputs = []
    # Input 1: Scalar float
    input1 = np.array(0.5, dtype=np.float32)
    generated_inputs.append({"input": input1})
    # Input 2: 1D tensor with negative values
    input2 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    generated_inputs.append({"input": input2})
    # Input 3: 2D tensor
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    generated_inputs.append({"input": input3})
    # Input 4: 3D tensor with mixed positive and negative values
    input4 = np.array([[[ -1, 2], [3, -4]], [[5, -6], [-7, 8]]], dtype=np.float32)
    generated_inputs.append({"input": input4})
    # Input 5: Larger tensor
    input5 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    generated_inputs.append({"input": input5})
    return generated_inputs
def crow_indices_copy_inputs():
    generated_inputs = []
    dense_tensor1 = torch.tensor([[0, 1, 0], [2, 0, 3]])
    sparse_tensor1 = dense_tensor1.to_sparse_csr()
    input1 = sparse_tensor1.crow_indices().numpy()
    generated_inputs.append({"input": input1})
    dense_tensor2 = torch.tensor([[0, 1, 0], [2, 0, 3]], dtype=torch.int64)
    sparse_tensor2 = dense_tensor2.to_sparse_csr()
    input2 = sparse_tensor2.crow_indices().numpy()
    generated_inputs.append({"input": input2})
    dense_tensor3 = torch.tensor([[0, 1, 0], [2, 0, 3]], dtype=torch.float32)
    sparse_tensor3 = dense_tensor3.to_sparse_csr()
    input3 = sparse_tensor3.crow_indices().numpy()
    generated_inputs.append({"input": input3})
    
    dense_tensor6 = torch.tensor([[1, 2], [3, 4]])
    sparse_tensor6 = dense_tensor6.to_sparse_csr()
    input6 = sparse_tensor6.crow_indices().numpy()
    generated_inputs.append({"input": input6})
    return generated_inputs
def tanh_inputs():
    generated_inputs = []
    # Input 1: 1D tensor of floats
    input1 = np.random.randn(5).astype(np.float32)
    input_dict1 = {"input": input1}
    generated_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D tensor of floats with negative values
    input2 = np.random.randn(3, 4).astype(np.float64)
    input_dict2 = {"input": input2}
    generated_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 3D tensor of integers
    input3 = np.random.randint(-5, 5, size=(2, 3, 2)).astype(np.int32)
    input_dict3 = {"input": input3}
    generated_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Scalar float
    input4 = np.array(3.14).astype(np.float32)
    input_dict4 = {"input": input4}
    generated_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 4D tensor of complex numbers
    input5 = (np.random.randn(2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2)).astype(np.complex64)
    input_dict5 = {"input": input5}
    generated_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Empty tensor
    input6 = np.array([]).astype(np.float32)
    input_dict6 = {"input": input6}
    generated_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: 1D tensor of floats with large values
    input7 = np.array([-1000.0, 0.0, 1000.0]).astype(np.float32)
    input_dict7 = {"input": input7}
    generated_inputs.append(copy.deepcopy(input_dict7))
    return generated_inputs
def cos__inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(2, 3, 4).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(1).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(5, 5, 5, 5).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(2, 2, 2, 2, 2).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = (torch.rand(3, 4) * 100).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = (torch.rand(3, 4) * -100).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    input8 = torch.zeros(3, 4).numpy()
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    input9 = torch.ones(3, 4).numpy()
    input_dict9 = {"input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = (torch.randn(3, 4) + 1j * torch.randn(3,4)).numpy()
    input_dict10 = {"input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    return list_of_inputs
def less_inputs():
    generated_inputs = []
    # Test case 1: Basic float tensors
    input1 = torch.randn(2, 3).numpy()
    other1 = torch.randn(2, 3).numpy()
    generated_inputs.append({"input": input1, "other": other1})
    # Test case 2: Integer tensors
    input2 = torch.randint(-5, 5, (4, 4)).numpy()
    other2 = torch.randint(-5, 5, (4, 4)).numpy()
    generated_inputs.append({"input": input2, "other": other2})
    # Test case 3: Different shapes (but broadcastable)
    input3 = torch.randn(1, 5).numpy()
    other3 = torch.randn(5).numpy()
    generated_inputs.append({"input": input3, "other": other3})
    # Test case 4: Scalar comparison
    input4 = torch.randn(3, 3).numpy()
    other4 = np.float64(0.5)
    generated_inputs.append({"input": input4, "other": other4})
    # Test case 5: Scalar comparison with integer tensor
    input5 = torch.randint(-10, 10, (2, 2)).numpy()
    other5 = np.int32(3)
    generated_inputs.append({"input": input5, "other": other5})
    
    # Test case 6: Higher dimensions
    input6 = torch.randn(2, 3, 4).numpy()
    other6 = torch.randn(2, 3, 4).numpy()
    generated_inputs.append({"input": input6, "other": other6})
    # Test case 7: Negative values
    input7 = torch.randn(5, 5) - 2.0
    other7 = torch.randn(5, 5) - 1.0
    generated_inputs.append({"input": input7.numpy(), "other": other7.numpy()})
    # Test case 8: zero values
    input8 = torch.zeros(2, 2).numpy()
    other8 = torch.ones(2, 2).numpy()
    generated_inputs.append({"input": input8, "other": other8})
    # Test case 9: same values
    input9 = torch.ones(3, 3).numpy()
    other9 = torch.ones(3, 3).numpy()
    generated_inputs.append({"input": input9, "other": other9})
    
    return generated_inputs
def nanmean_inputs():
    list_of_inputs = []
    # Example 1: Basic 2D tensor with NaNs, dim=0
    input_tensor = torch.tensor([[float('nan'), 1.0, 2.0], [3.0, float('nan'), 5.0]]).numpy()
    input_dict = {"input": input_tensor, "dim": (0,), "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 2: 1D tensor with NaNs, no dim specified
    input_tensor = torch.tensor([float('nan'), 1.0, 2.0, float('nan')]).numpy()
    input_dict = {"input": input_tensor, "dim": None, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 3: 3D tensor with NaNs, dim=(0, 2), keepdim=True
    input_tensor = torch.randn(2, 3, 4).float()
    input_tensor[0, 1, 2] = float('nan')
    input_tensor[1, 0, 0] = float('nan')
    input_tensor = input_tensor.numpy()
    input_dict = {"input": input_tensor, "dim": (0, 2), "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: Tensor with all NaNs in a dimension
    input_tensor = torch.tensor([[float('nan'), float('nan')], [1.0, 2.0]]).numpy()
    input_dict = {"input": input_tensor, "dim": (0,), "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 6: Negative values and NaNs
    input_tensor = torch.tensor([[-1.0, float('nan'), -2.0], [float('nan'), -3.0, 4.0]]).numpy()
    input_dict = {"input": input_tensor, "dim": (1,), "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 7: 4D tensor
    input_tensor = torch.randn(2, 2, 2, 2).float()
    input_tensor[0, 0, 0, 0] = float('nan')
    input_tensor[1, 1, 1, 1] = float('nan')
    input_tensor = input_tensor.numpy()
    input_dict = {"input": input_tensor, "dim": (0, 2), "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def vander_inputs():
    list_of_inputs = []
    # Input 1: Basic case with default N and increasing
    x = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"x": x, "N": None, "increasing": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: Specify N
    x = torch.tensor([1, 2, 3, 4]).numpy()
    input_dict = {"x": x, "N": 2, "increasing": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: Specify increasing=True
    x = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"x": x, "N": None, "increasing": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: Float tensor with N and increasing
    x = torch.tensor([1.5, 2.5, 3.5]).numpy()
    input_dict = {"x": x, "N": 4, "increasing": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: Negative values
    x = torch.tensor([-1, 0, 1]).numpy()
    input_dict = {"x": x, "N": None, "increasing": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: Larger N than len(x)
    x = torch.tensor([1, 2]).numpy()
    input_dict = {"x": x, "N": 5, "increasing": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: N = 1
    x = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"x": x, "N": 1, "increasing": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def fft_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensor
    input1 = torch.randn(4).numpy()
    input_dict1 = {
        "input": input1,
        "n": None,
        "dim": -1,
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Int tensor with specified n and dim
    input2 = torch.randint(0, 10, (8,)).numpy()
    input_dict2 = {
        "input": input2,
        "n": 4,
        "dim": 0,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Complex tensor with norm
    input3 = torch.randn(5, dtype=torch.complex64).numpy()
    input_dict3 = {
        "input": input3,
        "n": 10,
        "dim": -1,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Multidimensional tensor
    input4 = torch.randn(2, 3, 4).numpy()
    input_dict4 = {
        "input": input4,
        "n": None,
        "dim": 1,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Tensor with negative values and specified length
    input5 = torch.randint(-5, 5, (7,)).float().numpy()
    input_dict5 = {
        "input": input5,
        "n": 16,
        "dim": 0,
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Different dim
    input6 = torch.randn(2, 3, 4).numpy()
    input_dict6 = {
        "input": input6,
        "n": 2,
        "dim": 2,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: Larger tensor
    input7 = torch.randn(10).numpy()
    input_dict7 = {
        "input": input7,
        "n": 5,
        "dim": 0,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: Complex tensor with different dim
    input8 = torch.randn(2, 4, dtype=torch.complex64).numpy()
    input_dict8 = {
        "input": input8,
        "n": None,
        "dim": 1,
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    return list_of_inputs
def dsplit_inputs():
    list_of_inputs = []
    # Input 1: Basic 3D tensor with integer sections
    t = torch.arange(24.0).reshape(2, 3, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [2]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: 3D tensor with integer sections that evenly divide
    t = torch.arange(16.0).reshape(2, 2, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: 4D tensor with list of indices
    t = torch.arange(48.0).reshape(2, 2, 3, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [1, 3]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: 3D tensor with float values and tuple of indices
    t = torch.randn(2, 3, 5).numpy()
    input_dict = {"input": t, "indices_or_sections": (2, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: 3D tensor with a split resulting in an empty tensor
    t = torch.arange(24.0).reshape(2, 3, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [5]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: 3D tensor with complex numbers
    t = torch.randn(2, 2, 4, dtype=torch.complex64).numpy()
    input_dict = {"input": t, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 7: 5D tensor
    t = torch.arange(32.0).reshape(1, 2, 2, 2, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 8: 3D tensor, uneven split using list
    t = torch.arange(24.0).reshape(2, 3, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [1,2]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def hardsigmoid_inputs():
    list_of_inputs = []
    input1 = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.float32)
    input_dict1 = {
        "inplace": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[-4, -3, -2], [-1, 0, 1], [2, 3, 4]], dtype=np.float64)
    input_dict2 = {
        "inplace": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict3 = {
        "inplace": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([[-5.5, -3.2, 0.1], [2.8, 3.1, 6.2]], dtype=np.float16)
    input_dict4 = {
        "inplace": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([-10, 0, 10], dtype=np.float32)
    input_dict5 = {
        "inplace": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict6 = {
        "inplace": True,
        "input": input6.astype(np.float32) 
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs
def vdot_inputs():
    list_of_inputs = []
    input1 = torch.tensor([2, 3]).numpy()
    input2 = torch.tensor([2, 1]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input2 = torch.tensor([4.0, 5.0, 6.0]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.tensor([-1, -2, -3]).numpy()
    input2 = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = torch.tensor([1 + 2j, 3 - 1j]).numpy()
    input2 = torch.tensor([2 + 1j, 4 - 0j]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = torch.tensor([2 + 1j, 4 - 0j]).numpy()
    input2 = torch.tensor([1 + 2j, 3 - 1j]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.tensor([1, 2, 3], dtype=torch.float32).numpy()
    input2 = torch.tensor([4, 5, 6], dtype=torch.float32).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.tensor([1, 2, 3], dtype=torch.int64).numpy()
    input2 = torch.tensor([4, 5, 6], dtype=torch.int64).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.tensor([1 + 1j, 2 - 2j, 3 + 0j]).numpy()
    input2 = torch.tensor([4 - 1j, 5 + 2j, 6 - 0j]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def log_softmax_inputs():
    list_of_inputs = []
    # Input 1: 1D array, dim=0
    input1 = np.array([1.0, 2.0, 3.0])
    input_dict1 = {"input": input1, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D array, dim=1
    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input_dict2 = {"input": input2, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 2D array with negative values, dim=0
    input3 = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
    input_dict3 = {"input": input3, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 3D array, dim=2
    input4 = np.random.rand(2, 3, 4)
    input_dict4 = {"input": input4, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 4D array, dim=3
    input5 = np.random.rand(2, 3, 4, 5)
    input_dict5 = {"input": input5, "dim": 3}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: 2D integer array, dim=1
    input6 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict6 = {"input": input6, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: 1D array with a different dim
    input7 = np.array([1.0, 2.0, 3.0])
    input_dict7 = {"input": input7, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def igammac_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 4).numpy()
    other1 = torch.rand(3, 4).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randint(1, 10, (2, 2)).float().numpy()
    other2 = torch.randint(1, 5, (2, 2)).float().numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(5).numpy()
    other3 = torch.rand(5).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(1, 5, 5).numpy()
    other4 = torch.rand(1, 5, 5).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(2, 3, 4, 5).numpy()
    other5 = torch.rand(2, 3, 4, 5).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = torch.randn(3, 4).numpy()
    other6 = (torch.rand(3, 4) + 1e-6).numpy()
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = (torch.rand(3, 4) + 1e-6).numpy()
    other7 = torch.randn(3, 4).numpy()
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def constant_pad1d_inputs():
    list_of_inputs = []
    # Example 1: Integer padding, float value
    input_dict = {
        "input": torch.randn(1, 2, 4).numpy(),
        "padding": 2,
        "value": 3.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 2: Tuple padding, integer value
    input_dict = {
        "input": torch.randn(1, 2, 3).numpy(),
        "padding": (3, 1),
        "value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 3: Zero padding, negative value
    input_dict = {
        "input": torch.randn(1, 1, 5).numpy(),
        "padding": 0,
        "value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 4: Large padding values, large value
    input_dict = {
        "input": torch.randn(1, 3, 2).numpy(),
        "padding": (10, 5),
        "value": 100.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 5: Same padding on both sides, negative value
    input_dict = {
        "input": torch.randn(2, 2, 1).numpy(),
        "padding": 5,
        "value": -2.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def cond_inputs():
    list_of_inputs = []
    def true_fn1(x):
        return x + 1
    def false_fn1(x):
        return x - 1
    x1 = torch.tensor([2.0]).numpy()
    input_dict1 = {
        "pred": True,
        "true_fn": true_fn1,
        "false_fn": false_fn1,
        "operands": (x1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    def true_fn2(x):
        return x * 2
    def false_fn2(x):
        return x / 2
    x2 = torch.tensor([4.0]).numpy()
    input_dict2 = {
        "pred": False,
        "true_fn": true_fn2,
        "false_fn": false_fn2,
        "operands": (x2,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    def true_fn3(x, y):
        return x + y
    def false_fn3(x, y):
        return x - y
    x3 = torch.tensor([5.0]).numpy()
    y3 = torch.tensor([2.0]).numpy()
    input_dict3 = {
        "pred": torch.tensor([True]).bool().numpy().item(),
        "true_fn": true_fn3,
        "false_fn": false_fn3,
        "operands": (x3, y3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    def true_fn4():
        return torch.tensor([10.0]).numpy()
    def false_fn4():
        return torch.tensor([5.0]).numpy()
    
    input_dict4 = {
        "pred": torch.tensor([False]).bool().numpy().item(),
        "true_fn": true_fn4,
        "false_fn": false_fn4,
        "operands": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    def true_fn5(x):
        return torch.sin(x)
    def false_fn5(x):
        return torch.cos(x)
    x5 = torch.tensor([np.pi / 2]).numpy()
    input_dict5 = {
        "pred": x5 < 5,
        "true_fn": true_fn5,
        "false_fn": false_fn5,
        "operands": (x5,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    def true_fn6(x):
        return x * 3
    def false_fn6(x):
        return x / 3
    
    x6 = torch.tensor([-9.0]).numpy()
    input_dict6 = {
        "pred": torch.tensor([True]).bool().numpy(),
        "true_fn": true_fn6,
        "false_fn": false_fn6,
        "operands": (x6,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def diagonal_inputs():
    list_of_inputs = []
    # Example 1: Basic 2D tensor
    input1 = torch.randn(3, 3).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Example 2: 2D tensor with offset
    input2 = torch.randn(4, 4).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Example 3: 2D tensor with offset > 0
    input3 = torch.randn(5, 5).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Example 4: 2D tensor with offset < 0
    input4 = torch.randn(6, 6).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Example 5: 3D tensor with specified dims
    input5 = torch.randn(2, 5, 4).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Example 6: 4D tensor
    input6 = torch.randn(2, 3, 4, 5).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Example 7: Rectangular matrix
    input7 = torch.randn(2, 5).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    # Example 8: Integer tensor
    input8 = torch.randint(0, 10, (3, 3)).numpy()
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    # Example 9: Float64 tensor
    input9 = torch.randn(3, 3, dtype=torch.float64).numpy()
    input_dict9 = {"input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    # Example 10: Complex tensor
    input10 = torch.randn(3, 3, dtype=torch.complex64).numpy()
    input_dict10 = {"input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    return list_of_inputs
def erf__inputs():
    list_of_inputs = []
    input1 = np.random.randn(5).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.random.randn(2, 3).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.random.randn(1, 4, 4).astype(np.float16)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = (np.random.rand(3, 2) * 10 - 5).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.random.randn(2, 2, 2).astype(np.float64)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs
def bitwise_left_shift_inputs():
    list_of_inputs = []
    # Input 1: Basic case with positive integers
    input1 = np.array([1, 2, 3, 4], dtype=np.int32)
    other1 = np.array([1, 2, 0, 3], dtype=np.int32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Broadcasting with a scalar
    input2 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    other2 = np.array(2, dtype=np.int64)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Multi-dimensional arrays
    input3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    other3 = np.array([[[0, 1], [2, 0]], [[1, 0], [0, 2]]], dtype=np.int32)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Different integer types
    input4 = np.array([1, 2, 3], dtype=np.int16)
    other4 = np.array([1, 2, 1], dtype=np.int16)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Larger shift values
    input5 = np.array([1, 2, 3], dtype=np.int32)
    other5 = np.array([10, 5, 2], dtype=np.int32)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def conj_physical_inputs():
    list_of_inputs = []
    # Input 1: Complex tensor
    input1 = np.array([1 + 1j, 2 + 2j, 3 + 3j])
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Multidimensional complex tensor
    input2 = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Complex tensor with negative values
    input3 = np.array([-1 - 1j, -2 - 2j, -3 - 3j])
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Larger complex tensor
    input4 = np.array([
        [1 + 1j, 2 + 2j, 3 + 3j],
        [4 + 4j, 5 + 5j, 6 + 6j],
        [7 + 7j, 8 + 8j, 9 + 9j]
    ])
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Complex tensor with mixed positive and negative values
    input5 = np.array([1 - 1j, -2 + 2j, 3 - 3j, -4 + 4j])
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: 3D complex tensor
    input6 = np.array([
        [[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]],
        [[5 + 5j, 6 + 6j], [7 + 7j, 8 + 8j]]
    ])
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Complex tensor with zero values
    input7 = np.array([0 + 0j, 1 + 1j, 0 - 1j, -1 + 0j])
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def is_autocast_xla_enabled_inputs():
    list_of_inputs = []
    
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def threshold_inputs():
    list_of_inputs = []
    input_1 = np.array([-1, 0, 1, 2], dtype=np.float32)
    threshold_1 = 1.0
    value_1 = 0.0
    input_dict_1 = {"input": input_1, "threshold": threshold_1, "value": value_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    input_2 = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.float64)
    threshold_2 = 2.0
    value_2 = -1.0
    input_dict_2 = {"input": input_2, "threshold": threshold_2, "value": value_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    input_3 = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.int64)
    threshold_3 = 2
    value_3 = -1
    input_dict_3 = {"input": input_3, "threshold": float(threshold_3), "value": float(value_3)}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    input_4 = np.array([[-1.5, 0.5, 1.5], [2.5, 3.5, 4.5]], dtype=np.float32)
    threshold_4 = 2.0
    value_4 = 0.0
    input_dict_4 = {"input": input_4, "threshold": threshold_4, "value": value_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    input_5 = np.array([[-1, 0, 1], [2, 3, 4]], dtype=np.int32)
    threshold_5 = 1
    value_5 = 5
    input_dict_5 = {"input": input_5, "threshold": float(threshold_5), "value": float(value_5)}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    input_6 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=np.float32)
    threshold_6 = 4.0
    value_6 = 10.0
    input_dict_6 = {"input": input_6, "threshold": threshold_6, "value": value_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    input_7 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    threshold_7 = 3.0
    value_7 = -2.0
    input_dict_7 = {"input": input_7, "threshold": threshold_7, "value": value_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    return list_of_inputs
def miopen_batch_norm_inputs():
    list_of_inputs = []
    # Input 1: Basic case, 4D float32
    input1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    weight1 = np.random.randn(3).astype(np.float32)
    bias1 = np.random.randn(3).astype(np.float32)
    running_mean1 = np.random.randn(3).astype(np.float32)
    running_var1 = np.random.rand(3).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "weight": weight1,
        "bias": bias1,
        "running_mean": running_mean1,
        "running_var": running_var1,
        "training": True,
        "exponential_average_factor": 0.1,
        "epsilon": 1e-5
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 3D float64, different training params
    input2 = np.random.randn(2, 3, 4).astype(np.float64)
    weight2 = np.random.randn(3).astype(np.float64)
    bias2 = np.random.randn(3).astype(np.float64)
    running_mean2 = np.random.randn(3).astype(np.float64)
    running_var2 = np.random.rand(3).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "weight": weight2,
        "bias": bias2,
        "running_mean": running_mean2,
        "running_var": running_var2,
        "training": False,
        "exponential_average_factor": 0.2,
        "epsilon": 1e-4
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 5D float32, no training
    input3 = np.random.randn(2, 3, 4, 5, 6).astype(np.float32)
    weight3 = np.random.randn(3).astype(np.float32)
    bias3 = np.random.randn(3).astype(np.float32)
    running_mean3 = np.random.randn(3).astype(np.float32)
    running_var3 = np.random.rand(3).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "weight": weight3,
        "bias": bias3,
        "running_mean": running_mean3,
        "running_var": running_var3,
        "training": False,
        "exponential_average_factor": 0.3,
        "epsilon": 1e-3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 4D float16 (if supported), training true, different epsilon
    input4 = np.random.randn(2, 3, 4, 5).astype(np.float32) # keep float32, float16 causes problems with miopen
    weight4 = np.random.randn(3).astype(np.float32)
    bias4 = np.random.randn(3).astype(np.float32)
    running_mean4 = np.random.randn(3).astype(np.float32)
    running_var4 = np.random.rand(3).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "weight": weight4,
        "bias": bias4,
        "running_mean": running_mean4,
        "running_var": running_var4,
        "training": True,
        "exponential_average_factor": 0.4,
        "epsilon": 1e-2
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: 2D, training false
    input5 = np.random.randn(2, 3).astype(np.float32)
    weight5 = np.random.randn(3).astype(np.float32)
    bias5 = np.random.randn(3).astype(np.float32)
    running_mean5 = np.random.randn(3).astype(np.float32)
    running_var5 = np.random.rand(3).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "weight": weight5,
        "bias": bias5,
        "running_mean": running_mean5,
        "running_var": running_var5,
        "training": False,
        "exponential_average_factor": 0.5,
        "epsilon": 1e-1
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Different shapes
    input6 = np.random.randn(1, 5, 7, 9).astype(np.float32)
    weight6 = np.random.randn(5).astype(np.float32)
    bias6 = np.random.randn(5).astype(np.float32)
    running_mean6 = np.random.randn(5).astype(np.float32)
    running_var6 = np.random.rand(5).astype(np.float32)
    input_dict6 = {
        "input": input6,
        "weight": weight6,
        "bias": bias6,
        "running_mean": running_mean6,
        "running_var": running_var6,
        "training": True,
        "exponential_average_factor": 0.6,
        "epsilon": 1e-6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: Small values
    input7 = np.random.randn(2, 3, 4, 5).astype(np.float32) * 0.01
    weight7 = np.random.randn(3).astype(np.float32) * 0.01
    bias7 = np.random.randn(3).astype(np.float32) * 0.01
    running_mean7 = np.random.randn(3).astype(np.float32) * 0.01
    running_var7 = np.random.rand(3).astype(np.float32) * 0.01
    input_dict7 = {
        "input": input7,
        "weight": weight7,
        "bias": bias7,
        "running_mean": running_mean7,
        "running_var": running_var7,
        "training": False,
        "exponential_average_factor": 0.7,
        "epsilon": 1e-7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def greater_equal_inputs():
    list_of_inputs = []
    # Test case 1: Float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other1 = np.array([[2.0, 1.0], [3.0, 5.0]])
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Test case 2: Integer tensors
    input2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    other2 = np.array([[2, 1], [3, 5]], dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Test case 3: Scalar comparison
    input3 = np.array([[1, 2], [3, 4]])
    other3 = 3
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Test case 4: Multi-dimensional tensors
    input4 = np.random.rand(2, 3, 4)
    other4 = np.random.rand(2, 3, 4)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Test case 5: Negative values
    input5 = np.array([[-1, -2], [-3, -4]])
    other5 = np.array([[0, -1], [-3, -5]])
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Test case 6: Different shapes (broadcasting)
    input6 = np.array([[1, 2, 3]])
    other6 = np.array([2, 1, 4])
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Test case 7: Boolean tensor
    input7 = np.array([[True, False], [True, True]])
    other7 = np.array([[False, True], [True, False]])
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Test case 8: Zero dimension tensor
    input8 = np.array(5)
    other8 = np.array(3)
    input_dict8 = {"input": input8, "other": other8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    return list_of_inputs
def vector_norm_inputs():
    list_of_inputs = []
    # Input 1: Basic 1D tensor with default parameters
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {
        "input": input1,
        "ord": None,
        "dim": None,
        "keepdim": False,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D tensor with specified ord and dim
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict2 = {
        "input": input2,
        "ord": 1.0,
        "dim": 1,
        "keepdim": True,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 3D tensor with negative values and specified dtype
    input3 = np.array([[[1.0, -2.0], [3.0, -4.0]], [[-5.0, 6.0], [-7.0, 8.0]]], dtype=np.float64)
    input_dict3 = {
        "input": input3,
        "ord": 2.0,
        "dim": (1, 2),
        "keepdim": False,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 5: Matrix norm - REMOVED DUE TO ERROR - string not supported
    # Input 6: Negative Ord
    input6 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict6 = {
        "input": input6,
        "ord": -1.0,
        "dim": None,
        "keepdim": False,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: 4D tensor
    input7 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_dict7 = {
        "input": input7,
        "ord": 2.0,
        "dim": (1,2),
        "keepdim": True,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    input8 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict8 = {
        "input": input8,
        "ord": np.inf,
        "dim": None,
        "keepdim": False,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs
def is_anomaly_enabled_inputs():
    list_of_inputs = []
    input1 = {}
    list_of_inputs.append(copy.deepcopy(input1))
    
    return list_of_inputs
def view_as_complex_inputs():
    list_of_inputs = []
    # Input 1: Basic 2D float tensor
    input1 = np.random.randn(2, 2, 2).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 3D float tensor
    input2 = np.random.randn(2, 3, 2, 2).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    # Input 3: 1D float tensor
    input3 = np.random.randn(4,2).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Different sized 2D tensor
    input4 = np.random.randn(5, 2, 2).astype(np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Larger tensor with different data type
    input5 = np.random.randn(3, 4, 2, 2).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Another shape
    input6 = np.random.randn(2, 5, 2).astype(np.float64)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def get_autocast_cpu_dtype_inputs():
    list_of_inputs = []
    list_of_inputs.append({})
    
    return list_of_inputs
def get_deterministic_debug_mode_inputs():
    list_of_inputs = []
    
    list_of_inputs.append({})
    list_of_inputs.append({})
    list_of_inputs.append({})
    list_of_inputs.append({})
    
    list_of_inputs.append({})
    
    list_of_inputs.append({})
    
    return list_of_inputs
def gcd__inputs():
    list_of_inputs = []
    input1 = torch.randint(1, 100, (5,)).numpy()
    other1 = torch.randint(1, 100, (5,)).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randint(-100, -1, (3, 4)).numpy()
    other2 = torch.randint(1, 100, (3, 4)).numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randint(1, 100, (2, 2, 2)).numpy()
    other3 = torch.randint(1, 100, (2, 2, 2)).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randint(1, 100, (1,)).numpy()
    other4 = torch.randint(1, 100, (1,)).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randint(-100, -1, (2, 3, 4)).numpy()
    other5 = torch.randint(-100, -1, (2, 3, 4)).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = torch.randint(1, 100, (2,)).numpy()
    other6 = torch.tensor([0, 0]).numpy()
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.tensor([0, 0]).numpy()
    other7 = torch.randint(1, 100, (2,)).numpy()
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def qr_inputs():
    list_of_inputs = []
    # Input 1: Basic square matrix
    input1 = np.array([[12., -51, 4], [6, 167, -68], [-4, 24, -41]], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Rectangular matrix (m > n)
    input2 = np.random.rand(5, 3).astype(np.float32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Rectangular matrix (m < n)
    input3 = np.random.rand(3, 5).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Batch of matrices
    input4 = np.random.rand(2, 4, 4).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Matrix with negative values
    input5 = np.array([[-1, 2], [3, -4]], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def swapaxes_inputs():
    list_of_inputs = []
    # Example 1: 3D tensor, swapping axes 0 and 1
    x = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]).numpy()
    input_dict = {"input": x, "axis0": 0, "axis1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 2: 3D tensor, swapping axes 0 and 2
    x = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]).numpy()
    input_dict = {"input": x, "axis0": 0, "axis1": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 3: 2D tensor, swapping axes 0 and 1 (same as transpose)
    x = torch.randn(2, 3).numpy()
    input_dict = {"input": x, "axis0": 0, "axis1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 4: 4D tensor, swapping axes 1 and 3
    x = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {"input": x, "axis0": 1, "axis1": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 5: 1D tensor, swapping axes 0 and 0 (no change)
    x = torch.arange(5).numpy()
    input_dict = {"input": x, "axis0": 0, "axis1": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 6: 3D integer tensor
    x = torch.randint(0, 10, (2, 3, 4)).numpy()
    input_dict = {"input": x, "axis0": 0, "axis1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 7: 4D complex tensor
    x = torch.randn(2, 3, 4, 5, dtype=torch.complex64).numpy()
    input_dict = {"input": x, "axis0": 2, "axis1": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def rfft2_inputs():
    list_of_inputs = []
    input1 = torch.randn(10, 10).numpy()
    input_dict1 = {
        "input": input1,
        "s": None,
        "dim": (-2, -1),
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(5, 5, 5).numpy()
    input_dict2 = {
        "input": input2,
        "s": (10, 10),
        "dim": (-2, -1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(8, 8).numpy()
    input_dict3 = {
        "input": input3,
        "s": (4, 4),
        "dim": (0, 1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(12, 12).numpy()
    input_dict4 = {
        "input": input4,
        "s": (16, 16),
        "dim": (-2, -1),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(4, 4, 4, 4).numpy()
    input_dict5 = {
        "input": input5,
        "s": (8, 8),
        "dim": (-2, -1),
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = torch.randn(3, 5).numpy()
    input_dict6 = {
        "input": input6,
        "s": (6, 10),
        "dim": (0, 1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = torch.randn(2, 7, 9).numpy()
    input_dict7 = {
        "input": input7,
        "s": (4, 10),
        "dim": (1, 2),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.randn(4, 6).numpy()
    input_dict8 = {
        "input": input8,
        "s": None,
        "dim": (0, 1),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    return list_of_inputs
def manual_seed_inputs():
    list_of_inputs = []
    seed1 = np.array(0)
    input_dict1 = {"seed": seed1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    seed2 = np.array(1)
    input_dict2 = {"seed": seed2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    seed3 = np.array(2147483647)
    input_dict3 = {"seed": seed3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    seed4 = np.array(-2147483648)
    input_dict4 = {"seed": seed4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    seed5 = np.array(12345)
    input_dict5 = {"seed": seed5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    seed6 = np.array(99999)
    input_dict6 = {"seed": seed6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def swapdims_inputs():
    list_of_inputs = []
    # Input 1: 3D float tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "dim0": 0, "dim1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2: 4D int tensor
    input_tensor = torch.randint(0, 10, (2, 2, 3, 3)).numpy()
    input_dict = {"input": input_tensor, "dim0": 1, "dim1": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3: 2D complex tensor
    input_tensor = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict = {"input": input_tensor, "dim0": 0, "dim1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4: 5D tensor, negative dimension
    input_tensor = torch.randn(2, 3, 4, 5, 6).numpy()
    input_dict = {"input": input_tensor, "dim0": 0, "dim1": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5: 1D tensor
    input_tensor = torch.randn(5).numpy()
    input_dict = {"input": input_tensor, "dim0": 0, "dim1": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 6: 3D tensor, same dimensions
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "dim0": 1, "dim1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 7: 4D tensor, different datatypes
    input_tensor = torch.rand(2, 2, 3, 3).double().numpy()
    input_dict = {"input": input_tensor, "dim0": 1, "dim1": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def set_autocast_cache_enabled_inputs():
    list_of_inputs = []
    
    input_dict = {
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def trunc__inputs():
    list_of_inputs = []
    input1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(2, 3).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randint(-5, 5, (3, 4), dtype=torch.int32).float().numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(1, 2, 3).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(2, 2, 2, 2).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = (torch.rand(2,2) * 10 - 5).numpy() #Ensure negative values
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = torch.arange(-5, 5, 0.5).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def hann_window_inputs():
    generated_inputs = []
    input_dict = {
        "window_length": 5,
        "periodic": False,
        "dtype": torch.float32,
        "layout": "strided",
        "requires_grad": False
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "window_length": 10,
        "periodic": True,
        "dtype": torch.float64,
        "layout": "strided",
        "requires_grad": True
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "window_length": 7,
        "periodic": False,
        "dtype": torch.float16,
        "layout": "strided",
        "requires_grad": False
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "window_length": 12,
        "periodic": True,
        "dtype": torch.bfloat16,
        "layout": "strided",
        "requires_grad": True
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "window_length": 3,
        "periodic": False,
        "dtype": torch.float32,
        "layout": "strided",
        "requires_grad": False
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "window_length": 15,
        "periodic": True,
        "dtype": torch.float64,
        "layout": "strided",
        "requires_grad": True
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    
    return generated_inputs
def layer_norm_inputs():
    list_of_inputs = []
    # Input 1: Basic example with float32 and normalized_shape as a list
    input1 = np.random.randn(2, 3, 4).astype(np.float32)
    normalized_shape1 = [4]
    eps1 = 1e-5
    elementwise_affine1 = True
    input_dict1 = {
        "input": input1,
        "normalized_shape": normalized_shape1,
        "eps": eps1,
        "elementwise_affine": elementwise_affine1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Different normalized_shape and input dimensions, elementwise_affine=False
    input2 = np.random.randn(1, 5, 6, 7).astype(np.float32)
    normalized_shape2 = [6, 7]
    eps2 = 1e-8
    elementwise_affine2 = False
    input_dict2 = {
        "input": input2,
        "normalized_shape": normalized_shape2,
        "eps": eps2,
        "elementwise_affine": elementwise_affine2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    # Input 3:  Using a different data type (float64) and negative values
    input3 = (np.random.randn(3, 2, 5) - 0.5).astype(np.float64)  
    normalized_shape3 = [5]
    eps3 = 1e-6
    elementwise_affine3 = True
    input_dict3 = {
        "input": input3,
        "normalized_shape": normalized_shape3,
        "eps": eps3,
        "elementwise_affine": elementwise_affine3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 1D Input
    input4 = np.random.randn(10).astype(np.float32)
    normalized_shape4 = [10]
    eps4 = 1e-5
    elementwise_affine4 = True
    input_dict4 = {
        "input": input4,
        "normalized_shape": normalized_shape4,
        "eps": eps4,
        "elementwise_affine": elementwise_affine4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Smaller epsilon
    input5 = np.random.randn(2, 3, 4).astype(np.float32)
    normalized_shape5 = [4]
    eps5 = 1e-12
    elementwise_affine5 = True
    input_dict5 = {
        "input": input5,
        "normalized_shape": normalized_shape5,
        "eps": eps5,
        "elementwise_affine": elementwise_affine5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def corrcoef_inputs():
    generated_inputs = []
    # Example 1: Basic 2D tensor
    x = np.array([[0, 1, 2], [2, 1, 0]])
    input_dict = {"input": x}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Example 2: Random 2D tensor with float values
    x = np.random.randn(2, 4)
    input_dict = {"input": x}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Example 3: 1D tensor
    x = np.random.randn(5)
    input_dict = {"input": x}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Example 4: Scalar tensor
    x = np.array(5)
    input_dict = {"input": x}
    generated_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: Tensor with negative values
    x = np.array([[-1, 2, -3], [4, -5, 6]])
    input_dict = {"input": x}
    generated_inputs.append(copy.deepcopy(input_dict))
    # Example 6: Larger 2D tensor
    x = np.random.rand(10, 20)
    input_dict = {"input": x}
    generated_inputs.append(copy.deepcopy(input_dict))
    
    # Example 7: Integer tensor
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict = {"input": x}
    generated_inputs.append(copy.deepcopy(input_dict))
    return generated_inputs
def atanh_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensor within the valid range (-1, 1)
    input1 = np.array([-0.5, 0, 0.5]).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Float tensor with negative values, close to -1 and 1
    input2 = np.array([-0.99, -0.75, -0.25, 0.25, 0.75, 0.99]).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 2D float tensor
    input3 = np.array([[-0.8, 0.2], [0.4, -0.6]]).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 3D float tensor
    input4 = np.random.uniform(low=-0.9, high=0.9, size=(2, 2, 2)).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Float16 tensor
    input5 = np.array([-0.1, 0.3, -0.5, 0.7]).astype(np.float16)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Scalar tensor
    input6 = np.array(0.2).astype(np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Array containing near-zero values.
    input7 = np.array([-0.0001, 0, 0.0001]).astype(np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def HuberLoss_inputs():
    list_of_inputs = []
    # Case 1: Default values
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.5, 2.5, 3.5]),
        "reduction": 'mean',
        "delta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Different reduction
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.5, 2.5, 3.5]),
        "reduction": 'sum',
        "delta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: No reduction
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.5, 2.5, 3.5]),
        "reduction": 'none',
        "delta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: Different delta
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.5, 2.5, 3.5]),
        "reduction": 'mean',
        "delta": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Larger delta
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.5, 2.5, 3.5]),
        "reduction": 'mean',
        "delta": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def take_inputs():
    list_of_inputs = []
    src = torch.tensor([[4, 3, 5], [6, 7, 8]])
    index = torch.tensor([0, 2, 5])
    input_dict = {"input": src.numpy(), "index": index.long().numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))
    src = torch.tensor([[1.1, 2.2], [3.3, 4.4]])
    index = torch.tensor([0, 1, 2, 3])
    input_dict = {"input": src.numpy(), "index": index.long().numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))
    src = torch.arange(12).reshape(3, 4)
    index = torch.tensor([0, 5, 11, 7])
    input_dict = {"input": src.numpy(), "index": index.long().numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))
    src = torch.randn(2, 3, 4)
    index = torch.randint(0, 24, (5,))
    input_dict = {"input": src.numpy(), "index": index.long().numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))
    src = torch.randint(-10, 10, (5, 5))
    index = torch.tensor([0, -1, 5, -5])
    input_dict = {"input": src.numpy(), "index": index.long().numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))
    src = torch.tensor([1, 2, 3, 4, 5])
    index = torch.tensor([0, 0, 1, 2, 2, 3])
    input_dict = {"input": src.numpy(), "index": index.long().numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    src = torch.tensor([1+1j, 2+2j, 3+3j, 4+4j, 5+5j])
    index = torch.tensor([0, 2, 4])
    input_dict = {"input": src.numpy(), "index": index.long().numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def asin__inputs():
    list_of_inputs = []
    # Input 1: Basic float tensor within [-1, 1]
    input1 = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: 2D float tensor within [-1, 1]
    input2 = np.array([[0.2, 0.4, -0.1], [0.8, -0.9, 0.3]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 3D float tensor within [-1, 1]
    input3 = np.random.uniform(low=-1.0, high=1.0, size=(2, 3, 4)).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Float tensor with a single value
    input4 = np.array([0.6], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Larger tensor
    input5 = np.random.uniform(low=-1.0, high=1.0, size=(5, 5)).astype(np.float64)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: tensor with some edge values
    input6 = np.array([-1.0, -0.9, 0.0, 0.9, 1.0], dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    # Input 7: Large random tensor
    input7 = np.random.uniform(-1, 1, (10, 10, 10)).astype(np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs
def set_warn_always_inputs():
    list_of_inputs = []
    input_dict = {
        "warn_always": np.array(True)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "warn_always": np.array(False)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "warn_always": np.array([True])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "warn_always": np.array([False])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "warn_always": np.array([[True]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "warn_always": np.array([[False]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "warn_always": np.array([True, False, True])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "warn_always": np.array([[True, False], [False, True]])
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def get_rng_state_inputs():
    list_of_inputs = []
    
    input_dict = {}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def fftfreq_inputs():
    list_of_inputs = []
    input_dict = {
        "n": 5,
        "d": 1.0,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "n": 4,
        "d": 0.5,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "n": 10,
        "d": 2.0,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "n": 7,
        "d": 0.1,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "n": 16,
        "d": 1.5,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "n": 3,
        "d": 1.0,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "n": 8,
        "d": 0.25,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def soft_margin_loss_inputs():
    list_of_inputs = []
    input_dict1 = {
        'input': np.array([0.5, -0.2, 0.8]),
        'target': np.array([1, -1, 1]),
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input_dict2 = {
        'input': np.array([[0.5, -0.2], [0.8, -0.1]]),
        'target': np.array([[1, -1], [1, -1]]),
        'reduction': 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input_dict3 = {
        'input': np.array([0.2, -0.9, 0.5]),
        'target': np.array([-1, 1, -1]),
        'reduction': 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input_dict4 = {
        'input': np.array([1.0, -0.5, 0.0]),
        'target': np.array([1, -1, 1]),
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input_dict5 = {
        'input': np.array([-0.3, 0.7, -0.1]),
        'target': np.array([-1, 1, -1]),
        'reduction': 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def clip_inputs():
    list_of_inputs = []
    # Input 1: Basic float tensor
    input1 = np.random.randn(3, 4).astype(np.float32)
    min_val1 = -1.0
    max_val1 = 1.0
    input_dict1 = {"input": input1, "min": min_val1, "max": max_val1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Integer tensor
    input2 = np.random.randint(-5, 5, size=(2, 2), dtype=np.int32)
    min_val2 = 0.0
    max_val2 = 3.0
    input_dict2 = {"input": input2, "min": min_val2, "max": max_val2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 1D tensor
    input3 = np.random.rand(5).astype(np.float64)
    min_val3 = 0.2
    max_val3 = 0.8
    input_dict3 = {"input": input3, "min": min_val3, "max": max_val3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: 3D tensor with negative values
    input4 = np.random.randn(2, 3, 2).astype(np.float32) * 5 - 2.5
    min_val4 = -1.5
    max_val4 = 2.0
    input_dict4 = {"input": input4, "min": min_val4, "max": max_val4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: Tensor with all same values
    input5 = np.full((4, 4), 2.0, dtype=np.float32)
    min_val5 = 1.0
    max_val5 = 3.0
    input_dict5 = {"input": input5, "min": min_val5, "max": max_val5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Mixed positive and negative boundaries
    input6 = np.random.randn(2, 2).astype(np.float32)
    min_val6 = -0.5
    max_val6 = 0.5
    input_dict6 = {"input": input6, "min": min_val6, "max": max_val6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Double tensor
    input7 = np.random.randn(3, 4).astype(np.float64)
    min_val7 = -1.0
    max_val7 = 1.0
    input_dict7 = {"input": input7, "min": min_val7, "max": max_val7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
import io
def jit_load_inputs():
    list_of_inputs = []
    # Create a dummy ScriptModule for testing
    class DummyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)
        def forward(self, x):
            return self.linear(x)
    module = torch.jit.script(DummyModule())
    # Save the module to a buffer
    buffer = io.BytesIO()
    torch.jit.save(module, buffer)
    buffer.seek(0)
    # Input 1: Load from buffer with default parameters
    input_dict = {
        'f': buffer,
        'map_location': None,
        '_extra_files': None,
        '_restore_shapes': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Reset buffer for subsequent uses
    buffer.seek(0)
    # Input 2: Load from buffer, map to CPU
    input_dict = {
        'f': buffer,
        'map_location': 'cpu',
        '_extra_files': None,
        '_restore_shapes': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Reset buffer for subsequent uses
    buffer.seek(0)
    # Input 3: Load from buffer, map to CUDA (if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dict = {
        'f': buffer,
        'map_location': device,
        '_extra_files': None,
        '_restore_shapes': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Reset buffer for subsequent uses
    buffer.seek(0)
    # Input 4: Load from buffer with extra files
    extra_files = {'test.txt': 'This is a test file.'}
    input_dict = {
        'f': buffer,
        'map_location': None,
        '_extra_files': extra_files,
        '_restore_shapes': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Reset buffer for subsequent uses
    buffer.seek(0)
    
    # Input 5: Load from buffer, restore shapes
    input_dict = {
        'f': buffer,
        'map_location': None,
        '_extra_files': None,
        '_restore_shapes': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Reset buffer for subsequent uses
    buffer.seek(0)
    # Input 6: Load from string filename
    temp_file = "temp_scriptmodule.pt"
    torch.jit.save(module, temp_file)
    input_dict = {
        'f': temp_file,
        'map_location': None,
        '_extra_files': None,
        '_restore_shapes': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def not_equal_inputs():
    list_of_inputs = []
    # Case 1: Basic integer tensors
    input1 = np.array([1, 2, 3, 4])
    input2 = np.array([1, 3, 2, 4])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Float tensors with different shapes
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = np.array([1.0, 3.0])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Negative numbers and different dtypes
    input1 = np.array([-1, -2, 0, 1], dtype=np.int64)
    input2 = np.array([0, -2, 1, 2], dtype=np.int32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: Multidimensional tensors
    input1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input2 = np.array([[[1, 3], [3, 5]], [[5, 7], [7, 9]]])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Zero-dimensional tensors (scalars)
    input1 = np.array(5)
    input2 = np.array(6)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: Complex numbers
    input1 = np.array([1 + 1j, 2 + 2j, 3 + 3j])
    input2 = np.array([1 + 1j, 3 + 2j, 4 + 3j])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 7: Boolean tensors
    input1 = np.array([True, False, True])
    input2 = np.array([False, False, True])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def logit__inputs():
    list_of_inputs = []
    input1 = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    eps1 = 1e-6
    input_dict1 = {"input": input1, "eps": eps1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[0.2, 0.6], [0.4, 0.8]], dtype=np.float64)
    eps2 = 1e-8
    input_dict2 = {"input": input2, "eps": eps2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([0.01, 0.99, 0.5, 0.2, 0.8], dtype=np.float32)
    eps3 = 1e-5
    input_dict3 = {"input": input3, "eps": eps3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([[[0.3, 0.7], [0.1, 0.9]], [[0.6, 0.4], [0.8, 0.2]]], dtype=np.float64)
    eps4 = 1e-7
    input_dict4 = {"input": input4, "eps": eps4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    eps5 = 1e-6
    input_dict5 = {"input": input5, "eps": eps5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([[0.25, 0.75], [0.33, 0.67]], dtype=np.float64)
    eps6 = 1e-9
    input_dict6 = {"input": input6, "eps": eps6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([0.123, 0.456, 0.789], dtype=np.float32)
    eps7 = 1e-4
    input_dict7 = {"input": input7, "eps": eps7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def sigmoid__inputs():
    list_of_inputs = []
    # Input 1: Float tensor, 1D
    input1 = np.random.randn(5).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Float tensor, 2D
    input2 = np.random.randn(3, 4).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: Float tensor, 3D, with negative values
    input3 = (np.random.randn(2, 3, 2) * 10 - 5).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 5: Float Tensor, 4D
    input5 = np.random.randn(1, 2, 3, 4).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: Float Tensor, Scalar
    input6 = np.array(-2.5).astype(np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def log_inputs():
    list_of_inputs = []
    # Input 1: Positive float tensor
    input1 = np.random.rand(3, 4).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Input 2: Positive double tensor
    input2 = np.random.rand(2, 2, 2).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Input 3: 1D tensor
    input3 = np.random.rand(5).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Input 4: Scalar tensor
    input4 = np.array(0.5).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    # Input 5: tensor with value 1
    input5 = np.ones((2, 3)).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    # Input 6: tensor with value close to zero but positive to avoid log(0) issues
    input6 = np.random.rand(2, 3).astype(np.float32) * 0.0001
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    return list_of_inputs
def uninitialized_parameter_inputs():
    list_of_inputs = []
    return list_of_inputs
def ldexp_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randint(0, 5, (3, 4)).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(2, 2, 2).numpy()
    other2 = torch.randint(-3, 3, (2, 2, 2)).numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(5).numpy()
    other3 = torch.randint(-10, 10, (5,)).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(1, 1, 1, 1).numpy()
    other4 = torch.randint(-5, 5, (1, 1, 1, 1)).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(2, 3).double().numpy()
    other5 = torch.randint(0, 8, (2, 3)).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = torch.randn(4).int().numpy()
    other6 = torch.randint(-2, 2, (4,)).numpy()
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = torch.randn(2, 2).float().numpy()
    other7 = torch.randint(0, 16, (2, 2)).numpy()
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
def bitwise_right_shift_inputs():
    list_of_inputs = []
    input1 = np.array([10, 20, 30], dtype=np.int32)
    other1 = np.array([2, 3, 1], dtype=np.int32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([[15, 25], [35, 45]], dtype=np.int64)
    other2 = np.array([[1, 2], [3, 0]], dtype=np.int64)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[-10, -20], [-30, -40]], dtype=np.int8)
    other3 = np.array([[2, 1], [3, 2]], dtype=np.int8)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    other4 = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]], dtype=np.int16)
    other5 = np.array([[[1, 2], [0, 1]], [[2, 1], [1, 0]]], dtype=np.int16)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    input6 = np.array([255], dtype=np.uint8)
    other6 = np.array([4], dtype=np.uint8)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    input7 = np.array([1024, 2048], dtype=np.int32)
    other7 = np.array([5, 10], dtype=np.int32)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    return list_of_inputs
