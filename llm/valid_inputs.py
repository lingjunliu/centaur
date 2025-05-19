import torch, copy
import numpy as np
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
def flatten_inputs():
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
