import torch, copy
import numpy as np
def full_inputs():
    list_of_inputs = []
    input_dict = {
        "size": (2, 3),
        "fill_value": 1.0,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "size": (5,),
        "fill_value": 2.5,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "size": (2, 2, 2),
        "fill_value": -1.0,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "size": (1, 4, 4),
        "fill_value": 0.0,
        "dtype": np.float16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "size": (3, 1),
        "fill_value": 100.0,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def empty_strided_inputs():
    list_of_inputs = []
    size = (2, 3)
    stride = (3, 1)
    input_dict = {
        "size": size,
        "stride": stride
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    size = (4, 5, 2)
    stride = (10, 2, 1)
    input_dict = {
        "size": size,
        "stride": stride
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    size = (1, 1, 1)
    stride = (1, 1, 1)
    input_dict = {
        "size": size,
        "stride": stride
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    size = (7,)
    stride = (1,)
    input_dict = {
        "size": size,
        "stride": stride
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    size = (2, 2, 2, 2)
    stride = (8, 4, 2, 1)
    input_dict = {
        "size": size,
        "stride": stride
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def broadcast_shapes_inputs():
    list_of_inputs = []
    shapes = [(2, 3), (1, 3)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    shapes = [(2, 1, 4), (1, 3, 4)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    shapes = [(5, 4), (1, 4)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    shapes = [(15, 3, 5), (1, 3, 5)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    shapes = [(8, 1, 6, 1), (8, 7, 6, 5)]
    input_dict = {"shapes": shapes}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def rand_inputs():
    list_of_inputs = []
    input1 = (2,)
    input_dict1 = {
        "size": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = (3, 4)
    input_dict2 = {
        "size": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = (2, 3, 5)
    input_dict3 = {
        "size": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = (4, 2, 3, 2)
    input_dict4 = {
        "size": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = (1, 2, 3, 4, 5)
    input_dict5 = {
        "size": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def zeros_inputs():
    list_of_inputs = []
    input_dict = {
        "size": (5,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "size": (2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "size": (2, 3, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "size": (2, 3, 4, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "size": (2, 3, 4, 5, 6)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def ones_inputs():
    list_of_inputs = []
    input_dict = {
        "size": (1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "size": (5,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "size": (10,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "size": (100,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "size": (1000,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def max_pool1d_inputs():
    list_of_inputs = []
    input = torch.randn(1, 3, 10).numpy()
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    ceil_mode = False
    return_indices = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 4, 15).numpy()
    kernel_size = 4
    stride = 3
    padding = 0
    dilation = 2
    ceil_mode = True
    return_indices = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 2, 20).numpy()
    kernel_size = 5
    stride = 1
    padding = 2
    dilation = 1
    ceil_mode = False
    return_indices = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(3, 1, 12).numpy()
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    ceil_mode = True
    return_indices = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 8, 25).numpy()
    kernel_size = 6
    stride = 4
    padding = 1
    dilation = 2
    ceil_mode = False
    return_indices = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def mean_inputs():
    list_of_inputs = []
    input1 = torch.randn(2, 3, 4).numpy()
    input_dict1 = {
        "input": input1,
        "dim": 1,
        "keepdim": False,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(5, 5).numpy()
    input_dict2 = {
        "input": input2,
        "dim": 0,
        "keepdim": True,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(10).numpy()
    input_dict3 = {
        "input": input3,
        "dim": 0,
        "keepdim": False,
        "dtype": np.float16
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(2, 2, 2, 2).numpy()
    input_dict4 = {
        "input": input4,
        "dim": 2,
        "keepdim": True,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(3, 5).numpy()
    input_dict5 = {
        "input": input5,
        "dim": 1,
        "keepdim": True,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def rot90_inputs():
    list_of_inputs = []
    input1 = torch.randn(2, 3, 4, 5).numpy()
    k1 = 1
    dims1 = (2, 3)
    input_dict1 = {
        "input": input1,
        "k": k1,
        "dims": dims1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(2, 3, 4).numpy()
    k2 = 2
    dims2 = (1, 2)
    input_dict2 = {
        "input": input2,
        "k": k2,
        "dims": dims2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(5, 5).numpy()
    k3 = 3
    dims3 = (0, 1)
    input_dict3 = {
        "input": input3,
        "k": k3,
        "dims": dims3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(1, 1, 10, 10).numpy()
    k4 = -1
    dims4 = (2, 3)
    input_dict4 = {
        "input": input4,
        "k": k4,
        "dims": dims4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(3, 5, 7, 9).numpy()
    k5 = 0
    dims5 = (1, 3)
    input_dict5 = {
        "input": input5,
        "k": k5,
        "dims": dims5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs
def max_pool3d_inputs():
    list_of_inputs = []
    input = torch.randn(2, 3, 10, 10, 10).numpy()
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    ceil_mode = False
    return_indices = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 4, 12, 12, 12).numpy()
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    ceil_mode = True
    return_indices = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 2, 8, 8, 8).numpy()
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 2
    ceil_mode = False
    return_indices = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4, 1, 15, 15, 15).numpy()
    kernel_size = 4
    stride = 3
    padding = 1
    dilation = 1
    ceil_mode = True
    return_indices = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 8, 20, 20, 20).numpy()
    kernel_size = 5
    stride = 2
    padding = 2
    dilation = 1
    ceil_mode = False
    return_indices = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def var_inputs():
    list_of_inputs = []
    input_tensor = torch.randn(2, 3, 4).numpy()
    dim = 1
    correction = 0
    keepdim = False
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(5, 5).numpy()
    dim = 0
    correction = 1
    keepdim = True
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(2, 2, 2).numpy()
    dim = 2
    correction = 0
    keepdim = True
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(10).numpy()
    dim = 0
    correction = 1
    keepdim = False
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(3, 4, 5, 6).numpy()
    dim = 2
    correction = 0
    keepdim = False
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def transpose_inputs():
    list_of_inputs = []
    input_tensor = torch.randn(2, 3).numpy()
    dim0 = 0
    dim1 = 1
    input_dict = {
        "input": input_tensor,
        "dim0": dim0,
        "dim1": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(4, 5, 6).numpy()
    dim0 = 1
    dim1 = 2
    input_dict = {
        "input": input_tensor,
        "dim0": dim0,
        "dim1": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(1, 7, 8, 9).numpy()
    dim0 = 2
    dim1 = 3
    input_dict = {
        "input": input_tensor,
        "dim0": dim0,
        "dim1": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(10,).numpy()
    dim0 = 0
    dim1 = 0
    input_dict = {
        "input": input_tensor,
        "dim0": dim0,
        "dim1": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 2, 2, 2, 2).numpy()
    dim0 = 1
    dim1 = 3
    input_dict = {
        "input": input_tensor,
        "dim0": dim0,
        "dim1": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def max_pool2d_inputs():
    list_of_inputs = []
    input = torch.randn(1, 3, 32, 32).numpy()
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    ceil_mode = False
    return_indices = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 1, 16, 16).numpy()
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    ceil_mode = False
    return_indices = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 3, 64, 64).numpy()
    kernel_size = 4
    stride = 4
    padding = 0
    dilation = 1
    ceil_mode = True
    return_indices = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 5, 28, 28).numpy()
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 2
    ceil_mode = False
    return_indices = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4, 1, 10, 10).numpy()
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 1
    ceil_mode = True
    return_indices = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def normalize_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 5).numpy()
    dim1 = 0
    eps1 = 1e-12
    p1 = 2.0
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "eps": eps1,
        "p": p1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(2, 4, 6).numpy()
    dim2 = 1
    eps2 = 1e-8
    p2 = 1.0
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "eps": eps2,
        "p": p2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(10).numpy()
    dim3 = 0
    eps3 = 1e-5
    p3 = float('inf')
    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "eps": eps3,
        "p": p3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(2, 2, 2).numpy()
    dim4 = 2
    eps4 = 1e-6
    p4 = -2.0
    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "eps": eps4,
        "p": p4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(1, 5, 7, 9).numpy()
    dim5 = 3
    eps5 = 1e-4
    p5 = 0.5
    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "eps": eps5,
        "p": p5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def split_inputs():
    list_of_inputs = []
    tensor = torch.randn(4, 4).numpy()
    split_size_or_sections = 2
    dim = 0
    input_dict = {
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    tensor = torch.randn(6, 6).numpy()
    split_size_or_sections = 3
    dim = 1
    input_dict = {
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    tensor = torch.randn(2, 8, 4).numpy()
    split_size_or_sections = 4
    dim = 1
    input_dict = {
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    tensor = torch.randn(5, 5, 5).numpy()
    split_size_or_sections = 1
    dim = 2
    input_dict = {
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    tensor = torch.randn(10).numpy()
    split_size_or_sections = 5
    dim = 0
    input_dict = {
        "tensor": tensor,
        "split_size_or_sections": split_size_or_sections,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def var_mean_inputs():
    list_of_inputs = []
    input = torch.randn(3, 4, 5).numpy()
    dim = 1
    correction = 1
    keepdim = False
    input_dict = {
        "input": input,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 2).numpy()
    dim = 0
    correction = 0
    keepdim = True
    input_dict = {
        "input": input,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(5).numpy()
    dim = 0
    correction = 1
    keepdim = False
    input_dict = {
        "input": input,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 3, 4).numpy()
    dim = 2
    correction = 2
    keepdim = True
    input_dict = {
        "input": input,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 5, 5).numpy()
    dim = 1
    correction = 1
    keepdim = False
    input_dict = {
        "input": input,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def std_inputs():
    list_of_inputs = []
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    dim = (0, 1)
    correction = 1
    keepdim = True
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(5, 5).numpy()
    dim = (0,)
    correction = 0
    keepdim = False
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(2, 2, 2).numpy()
    dim = (1, 2)
    correction = 1
    keepdim = True
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(10).numpy()
    dim = (0,)
    correction = 0
    keepdim = False
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(3, 4).numpy()
    dim = (0, 1)
    correction = 2
    keepdim = False
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def unsqueeze_inputs():
    list_of_inputs = []
    input_tensor = torch.randn(2, 3).numpy()
    dim = 0
    input_dict = {
        "tensor": input_tensor,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(2, 3, 4).numpy()
    dim = 1
    input_dict = {
        "tensor": input_tensor,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(5).numpy()
    dim = 0
    input_dict = {
        "tensor": input_tensor,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(2, 2, 2, 2).numpy()
    dim = 3
    input_dict = {
        "tensor": input_tensor,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(3, 4, 5).numpy()
    dim = -1
    input_dict = {
        "tensor": input_tensor,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def nansum_inputs():
    list_of_inputs = []
    # Input 1
    input = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 4.0]])
    dim = (0,)
    keepdim = False
    dtype = np.float32
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2
    input = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 4.0]])
    dim = (1,)
    keepdim = True
    dtype = np.float64
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3
    input = np.array([[[1.0, np.nan], [2.0, 3.0]], [[np.nan, 4.0], [5.0, 6.0]]])
    dim = (0, 1)
    keepdim = False
    dtype = np.float32
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4
    input = np.array([[[1.0, np.nan], [2.0, 3.0]], [[np.nan, 4.0], [5.0, 6.0]]])
    dim = (2,)
    keepdim = True
    dtype = np.float64
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5
    input = np.array([np.nan, np.nan, np.nan])
    dim = (0,)
    keepdim = False
    dtype = np.float32
    input_dict = {"input": input, "dim": dim, "keepdim": keepdim, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def pixel_shuffle_inputs():
    list_of_inputs = []
    
    input = torch.randn(1, 9, 4, 4).numpy()
    upscale_factor = 3
    input_dict = {
        "input": input,
        "upscale_factor": upscale_factor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(1, 4, 8, 8).numpy()
    upscale_factor = 2
    input_dict = {
        "input": input,
        "upscale_factor": upscale_factor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 16, 5, 5).numpy()
    upscale_factor = 4
    input_dict = {
        "input": input,
        "upscale_factor": upscale_factor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(1, 25, 2, 2).numpy()
    upscale_factor = 5
    input_dict = {
        "input": input,
        "upscale_factor": upscale_factor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 64, 3, 3).numpy()
    upscale_factor = 8
    input_dict = {
        "input": input,
        "upscale_factor": upscale_factor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def logcumsumexp_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 4, 5).numpy()
    dim1 = 1
    input_dict1 = {
        "input": input1,
        "dim": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(2, 2).numpy()
    dim2 = 0
    input_dict2 = {
        "input": input2,
        "dim": dim2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(1, 5, 7).numpy()
    dim3 = 2
    input_dict3 = {
        "input": input3,
        "dim": dim3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(4).numpy()
    dim4 = 0
    input_dict4 = {
        "input": input4,
        "dim": dim4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(2, 3, 4, 5).numpy()
    dim5 = 3
    input_dict5 = {
        "input": input5,
        "dim": dim5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def movedim_inputs():
    list_of_inputs = []
    input = torch.randn(2, 3, 4, 5).numpy()
    source = 1
    destination = 3
    input_dict = {
        "input": input,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 3, 4).numpy()
    source = 0
    destination = 2
    input_dict = {
        "input": input,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(5, 6).numpy()
    source = 0
    destination = 1
    input_dict = {
        "input": input,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 3, 4, 5, 6).numpy()
    source = 4
    destination = 0
    input_dict = {
        "input": input,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 3, 4).numpy()
    source = 2
    destination = 0
    input_dict = {
        "input": input,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def narrow_inputs():
    list_of_inputs = []
    input = torch.randn(5, 5).numpy()
    dim = 0
    start = 1
    length = 3
    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(3, 4, 5).numpy()
    dim = 1
    start = 0
    length = 2
    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 6).numpy()
    dim = 1
    start = 2
    length = 3
    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(10).numpy()
    dim = 0
    start = 5
    length = 4
    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 3, 4, 5).numpy()
    dim = 2
    start = 1
    length = 2
    input_dict = {
        "input": input,
        "dim": dim,
        "start": start,
        "length": length
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def pad_inputs():
    list_of_inputs = []
    input = torch.randn(2, 3, 4, 5).numpy()
    pad = (1, 1, 2, 2, 0, 0, 0, 0)
    mode = 'constant'
    value = 0.0
    input_dict = {
        "input": input,
        "pad": pad,
        "mode": mode,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 2, 3).numpy()
    pad = (2, 1, 0, 0, 0, 0)
    mode = 'constant'
    value = 0.0
    input_dict = {
        "input": input,
        "pad": pad,
        "mode": mode,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4, 5).numpy()
    pad = (0, 1, 2, 0)
    mode = 'constant'
    value = 0.0
    input_dict = {
        "input": input,
        "pad": pad,
        "mode": mode,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 3, 32, 32).numpy()
    pad = (3, 3, 2, 2, 0, 0, 0, 0)
    mode = 'constant'
    value = 0.0
    input_dict = {
        "input": input,
        "pad": pad,
        "mode": mode,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(1, 1, 5, 5).numpy()
    pad = (1, 2, 3, 4, 0, 0, 0, 0)
    mode = 'constant'
    value = 1.5
    input_dict = {
        "input": input,
        "pad": pad,
        "mode": mode,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def reshape_inputs():
    list_of_inputs = []
    input = torch.randn(2, 3, 4).numpy()
    shape = (24,)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(5, 2).numpy()
    shape = (10,)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(3, 4, 5).numpy()
    shape = (60,)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(10).numpy()
    shape = (10,)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 2, 2, 2).numpy()
    shape = (16,)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def std_mean_inputs():
    list_of_inputs = []
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    dim_val = 1
    correction_val = 0
    keepdim_val = False
    input_dict = {
        "input": input_tensor,
        "dim": dim_val,
        "correction": correction_val,
        "keepdim": keepdim_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(5, 5).numpy()
    dim_val = 0
    correction_val = 1
    keepdim_val = True
    input_dict = {
        "input": input_tensor,
        "dim": dim_val,
        "correction": correction_val,
        "keepdim": keepdim_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(2, 2, 2).numpy()
    dim_val = 2
    correction_val = 2
    keepdim_val = False
    input_dict = {
        "input": input_tensor,
        "dim": dim_val,
        "correction": correction_val,
        "keepdim": keepdim_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(10).numpy()
    dim_val = 0
    correction_val = 0
    keepdim_val = True
    input_dict = {
        "input": input_tensor,
        "dim": dim_val,
        "correction": correction_val,
        "keepdim": keepdim_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(1, 5, 1, 5).numpy()
    dim_val = 3
    correction_val = 1
    keepdim_val = False
    input_dict = {
        "input": input_tensor,
        "dim": dim_val,
        "correction": correction_val,
        "keepdim": keepdim_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def linspace_inputs():
    list_of_inputs = []
    start = torch.tensor(0.0).item()
    end = torch.tensor(1.0).item()
    steps = 5
    dtype = torch.float32
    requires_grad = False
    
    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    start = torch.tensor(-1.0).item()
    end = torch.tensor(1.0).item()
    steps = 10
    dtype = torch.float64
    requires_grad = True
    
    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    start = torch.tensor(2.0).item()
    end = torch.tensor(5.0).item()
    steps = 7
    dtype = torch.float32
    requires_grad = False
    
    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    start = torch.tensor(-5.0).item()
    end = torch.tensor(-2.0).item()
    steps = 4
    dtype = torch.float64
    requires_grad = True
    
    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    start = torch.tensor(10.0).item()
    end = torch.tensor(20.0).item()
    steps = 12
    dtype = torch.float32
    requires_grad = False
    
    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def logspace_inputs():
    list_of_inputs = []
    start = torch.tensor(0.0).numpy()
    end = torch.tensor(5.0).numpy()
    steps = 10
    base = 10.0
    dtype = torch.float32
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
    start = torch.tensor(-2.0).numpy()
    end = torch.tensor(2.0).numpy()
    steps = 5
    base = 2.0
    dtype = torch.float64
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
    start = torch.tensor(1.0).numpy()
    end = torch.tensor(10.0).numpy()
    steps = 20
    base = np.e  # Euler's number
    dtype = torch.float32
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
    start = torch.tensor(-1.0).numpy()
    end = torch.tensor(1.0).numpy()
    steps = 15
    base = 5.0
    dtype = torch.float64
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
    start = torch.tensor(2.0).numpy()
    end = torch.tensor(8.0).numpy()
    steps = 7
    base = 3.0
    dtype = torch.float32
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
    
    return list_of_inputs
def lstsq_inputs():
    list_of_inputs = []
    A = torch.randn(5, 3).numpy()
    B = torch.randn(5, 2).numpy()
    rcond = 1e-15
    input_dict = {
        "A": A,
        "B": B,
        "rcond": rcond
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(10, 5).numpy()
    B = torch.randn(10, 1).numpy()
    rcond = 1e-10
    input_dict = {
        "A": A,
        "B": B,
        "rcond": rcond
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(7, 4).numpy()
    B = torch.randn(7, 3).numpy()
    rcond = 1e-5
    input_dict = {
        "A": A,
        "B": B,
        "rcond": rcond
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(3, 2).numpy()
    B = torch.randn(3, 1).numpy()
    rcond = 1e-2
    input_dict = {
        "A": A,
        "B": B,
        "rcond": rcond
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(6, 4).numpy()
    B = torch.randn(6, 2).numpy()
    rcond = 0.1
    input_dict = {
        "A": A,
        "B": B,
        "rcond": rcond
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def trapz_inputs():
    list_of_inputs = []
    y = torch.randn(5).numpy()
    x = torch.arange(0, 5).numpy()
    dx = 1.0
    dim = 0
    input_dict = {
        "y": y,
        "x": x,
        "dx": dx,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    y = torch.randn(3, 5).numpy()
    x = torch.arange(0, 5).numpy()
    dx = 1.0
    dim = 1
    input_dict = {
        "y": y,
        "x": x,
        "dx": dx,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    y = torch.randn(5, 3).numpy()
    x = torch.arange(0, 5).numpy()
    dx = 2.0
    dim = 0
    input_dict = {
        "y": y,
        "x": x,
        "dx": dx,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    y = torch.randn(5).numpy()
    x = torch.arange(0, len(y)).numpy()
    dx = 2.0
    dim = 0
    input_dict = {
        "y": y,
        "x": x,
        "dx": dx,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    y = torch.randn(3, 5, 2).numpy()
    x = torch.arange(0, 5).numpy()
    dx = 0.5
    dim = 1
    input_dict = {
        "y": y,
        "x": x,
        "dx": dx,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def triangular_solve_inputs():
    list_of_inputs = []
    A = torch.randn(3, 3).numpy()
    b = torch.randn(3, 1).numpy()
    input_dict = {
        "b": b,
        "A": A,
        "upper": True,
        "transpose": False,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(4, 4).numpy()
    b = torch.randn(4, 2).numpy()
    input_dict = {
        "b": b,
        "A": A,
        "upper": False,
        "transpose": True,
        "unitriangular": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(5, 5).numpy()
    b = torch.randn(5, 1).numpy()
    input_dict = {
        "b": b,
        "A": A,
        "upper": True,
        "transpose": True,
        "unitriangular": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(2, 2).numpy()
    b = torch.randn(2, 3).numpy()
    input_dict = {
        "b": b,
        "A": A,
        "upper": False,
        "transpose": False,
        "unitriangular": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(6, 6).numpy()
    b = torch.randn(6, 4).numpy()
    input_dict = {
        "b": b,
        "A": A,
        "upper": True,
        "transpose": False,
        "unitriangular": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def tensordot_inputs():
    list_of_inputs = []
    a = torch.randn(3, 4, 5).numpy()
    b = torch.randn(4, 3, 2).numpy()
    dims = ([1, 0], [0, 1])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    a = torch.randn(2, 3).numpy()
    b = torch.randn(3, 4).numpy()
    dims = (([1], [0]))
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    a = torch.randn(3, 4, 5).numpy()
    b = torch.randn(3, 5, 2).numpy()
    dims = ([0, 2], [0, 1])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    a = torch.randn(2, 3, 4).numpy()
    b = torch.randn(4, 5).numpy()
    dims = ([2], [0])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    a = torch.randn(2, 2).numpy()
    b = torch.randn(2).numpy()
    dims = ([1], [0])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def multilabel_soft_margin_loss_inputs():
    list_of_inputs = []
    input_tensor = torch.randn(3, 5)
    target_tensor = torch.randint(0, 2, (3, 5)).float()
    weight_tensor = torch.randn(5)
    input_dict = {
        "input": input_tensor.numpy(),
        "target": target_tensor.numpy(),
        "weight": weight_tensor.numpy(),
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(2, 4)
    target_tensor = torch.randint(0, 2, (2, 4)).float()
    weight_tensor = None
    input_dict = {
        "input": input_tensor.numpy(),
        "target": target_tensor.numpy(),
        "weight": weight_tensor,
        "size_average": False,
        "reduce": True,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(5, 3)
    target_tensor = torch.randint(0, 2, (5, 3)).float()
    weight_tensor = torch.ones(3)
    input_dict = {
        "input": input_tensor.numpy(),
        "target": target_tensor.numpy(),
        "weight": weight_tensor.numpy(),
        "size_average": None,
        "reduce": False,
        "reduction": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(4, 2)
    target_tensor = torch.randint(0, 2, (4, 2)).float()
    weight_tensor = torch.tensor([0.5, 1.0])
    input_dict = {
        "input": input_tensor.numpy(),
        "target": target_tensor.numpy(),
        "weight": weight_tensor.numpy(),
        "size_average": True,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 6)
    target_tensor = torch.randint(0, 2, (1, 6)).float()
    weight_tensor = None
    input_dict = {
        "input": input_tensor.numpy(),
        "target": target_tensor.numpy(),
        "weight": weight_tensor,
        "size_average": False,
        "reduce": False,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def linear_inputs():
    list_of_inputs = []
    input = torch.randn(3, 5).numpy()
    weight = torch.randn(2, 5).numpy()
    bias = torch.randn(2).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 10).numpy()
    weight = torch.randn(5, 10).numpy()
    bias = torch.randn(5).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 4).numpy()
    weight = torch.randn(3, 4).numpy()
    bias = torch.randn(3).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(5, 7).numpy()
    weight = torch.randn(1, 7).numpy()
    bias = torch.randn(1).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4, 6).numpy()
    weight = torch.randn(8, 6).numpy()
    bias = torch.randn(8).numpy()
    input_dict = {
        "input": input,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def lu_solve_inputs():
    list_of_inputs = []
    # Input 1
    A = torch.randn(3, 3)
    b = torch.randn(3, 1)
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b_np = b.numpy()
    LU_data_np = LU_data.numpy()
    LU_pivots_np = LU_pivots.numpy()
    input_dict = {
        "b": b_np,
        "LU_data": LU_data_np,
        "LU_pivots": LU_pivots_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2
    A = torch.randn(4, 4)
    b = torch.randn(4, 2)
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b_np = b.numpy()
    LU_data_np = LU_data.numpy()
    LU_pivots_np = LU_pivots.numpy()
    input_dict = {
        "b": b_np,
        "LU_data": LU_data_np,
        "LU_pivots": LU_pivots_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3
    A = torch.randn(5, 5)
    b = torch.randn(5, 1)
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b_np = b.numpy()
    LU_data_np = LU_data.numpy()
    LU_pivots_np = LU_pivots.numpy()
    input_dict = {
        "b": b_np,
        "LU_data": LU_data_np,
        "LU_pivots": LU_pivots_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4
    A = torch.randn(2, 2)
    b = torch.randn(2, 3)
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b_np = b.numpy()
    LU_data_np = LU_data.numpy()
    LU_pivots_np = LU_pivots.numpy()
    input_dict = {
        "b": b_np,
        "LU_data": LU_data_np,
        "LU_pivots": LU_pivots_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5
    A = torch.randn(6, 6)
    b = torch.randn(6, 2)
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b_np = b.numpy()
    LU_data_np = LU_data.numpy()
    LU_pivots_np = LU_pivots.numpy()
    input_dict = {
        "b": b_np,
        "LU_data": LU_data_np,
        "LU_pivots": LU_pivots_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def margin_ranking_loss_inputs():
    list_of_inputs = []
    input1 = torch.randn(5).numpy()
    input2 = torch.randn(5).numpy()
    target = torch.randint(0, 2, (5,)).float().numpy() * 2 - 1
    margin = 0.0
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
    input1 = torch.randn(3, 4).numpy()
    input2 = torch.randn(3, 4).numpy()
    target = torch.randint(0, 2, (3, 4)).float().numpy() * 2 - 1
    margin = 0.5
    size_average = False
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
    input1 = torch.randn(10).numpy()
    input2 = torch.randn(10).numpy()
    target = torch.randint(0, 2, (10,)).float().numpy() * 2 - 1
    margin = 1.0
    size_average = False
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
    
    input1 = torch.randn(2, 2, 2).numpy()
    input2 = torch.randn(2, 2, 2).numpy()
    target = torch.randint(0, 2, (2, 2, 2)).float().numpy() * 2 - 1
    margin = 0.2
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
    input1 = torch.randn(1).numpy()
    input2 = torch.randn(1).numpy()
    target = torch.randint(0, 2, (1,)).float().numpy() * 2 - 1
    margin = 0.8
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
    return list_of_inputs
def where_inputs():
    list_of_inputs = []
    condition = (torch.randn(3, 2) > 0).numpy()
    input_tensor = torch.randn(3, 2).numpy()
    other_tensor = torch.randn(3, 2).numpy()
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    condition = (torch.randn(2, 2, 2) > 0).numpy()
    input_tensor = torch.randn(2, 2, 2).numpy()
    other_tensor = torch.randn(2, 2, 2).numpy()
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    condition = (torch.randn(5) > 0).numpy()
    input_tensor = torch.randn(5).numpy()
    other_tensor = torch.randn(5).numpy()
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    condition = (torch.randn(1, 4, 4) > 0).numpy()
    input_tensor = torch.randn(1, 4, 4).numpy()
    other_tensor = torch.randn(1, 4, 4).numpy()
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    condition = (torch.randn(2, 3, 4, 5) > 0).numpy()
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    other_tensor = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def pow_inputs():
    list_of_inputs = []
    input_tensor = torch.randn(2, 3).numpy()
    exponent_tensor = torch.tensor(2.0)
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(4, 4).numpy()
    exponent_tensor = torch.tensor(0.5)
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(1, 5, 5).numpy()
    exponent_tensor = torch.randn(1, 5, 5)
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(3, 2, 1).numpy()
    exponent_tensor = torch.tensor(-1.0)
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 2).numpy()
    exponent_tensor = torch.tensor(2.0)
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def mm_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 4).numpy()
    input2 = torch.randn(4, 5).numpy()
    input_dict = {
        "input": input1,
        "mat2": input2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randn(2, 3).numpy()
    input2 = torch.randn(3, 1).numpy()
    input_dict = {
        "input": input1,
        "mat2": input2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randn(1, 5).numpy()
    input2 = torch.randn(5, 7).numpy()
    input_dict = {
        "input": input1,
        "mat2": input2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randn(6, 2).numpy()
    input2 = torch.randn(2, 8).numpy()
    input_dict = {
        "input": input1,
        "mat2": input2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randn(4, 4).numpy()
    input2 = torch.randn(4, 4).numpy()
    input_dict = {
        "input": input1,
        "mat2": input2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def mv_inputs():
    list_of_inputs = []
    input_matrix = torch.randn(3, 3).numpy()
    vec = torch.randn(3).numpy()
    input_dict = {"input": input_matrix, "vec": vec}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_matrix = torch.randn(5, 5).numpy()
    vec = torch.randn(5).numpy()
    input_dict = {"input": input_matrix, "vec": vec}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_matrix = torch.randn(2, 4).numpy()
    vec = torch.randn(4).numpy()
    input_dict = {"input": input_matrix, "vec": vec}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_matrix = torch.randn(4, 2).numpy()
    vec = torch.randn(2).numpy()
    input_dict = {"input": input_matrix, "vec": vec}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_matrix = torch.randn(1, 1).numpy()
    vec = torch.randn(1).numpy()
    input_dict = {"input": input_matrix, "vec": vec}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def polar_inputs():
    list_of_inputs = []
    abs_val = torch.randn(3, 4).numpy()
    angle_val = torch.randn(3, 4).numpy()
    input_dict = {
        "abs": abs_val,
        "angle": angle_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    abs_val = torch.randn(2, 2, 2).numpy()
    angle_val = torch.randn(2, 2, 2).numpy()
    input_dict = {
        "abs": abs_val,
        "angle": angle_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    abs_val = torch.randn(5).numpy()
    angle_val = torch.randn(5).numpy()
    input_dict = {
        "abs": abs_val,
        "angle": angle_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    abs_val = torch.randn(1, 1).numpy()
    angle_val = torch.randn(1, 1).numpy()
    input_dict = {
        "abs": abs_val,
        "angle": angle_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    abs_val = torch.randn(2, 3, 4, 5).numpy()
    angle_val = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {
        "abs": abs_val,
        "angle": angle_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def prelu_inputs():
    list_of_inputs = []
    input = torch.randn(3, 4, 5).numpy()
    weight = torch.randn(1).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 3, 4, 5).numpy()
    weight = torch.randn(1).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 5).numpy()
    weight = torch.randn(1).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 2).numpy()
    weight = torch.randn(1).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4).numpy()
    weight = torch.randn(1).numpy()
    input_dict = {
        "input": input,
        "weight": weight
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def solve_inputs():
    list_of_inputs = []
    A = torch.randn(3, 3).numpy()
    B = torch.randn(3, 3).numpy()
    left = True
    input_dict = {
        "A": A,
        "B": B,
        "left": left
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(5, 5).numpy()
    B = torch.randn(5, 5).numpy()
    left = False
    input_dict = {
        "A": A,
        "B": B,
        "left": left
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(2, 2).numpy()
    B = torch.randn(2, 2).numpy()
    left = True
    input_dict = {
        "A": A,
        "B": B,
        "left": left
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(4, 4).numpy()
    B = torch.randn(4, 4).numpy()
    left = False
    input_dict = {
        "A": A,
        "B": B,
        "left": left
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    A = torch.randn(6, 6).numpy()
    B = torch.randn(6, 6).numpy()
    left = True
    input_dict = {
        "A": A,
        "B": B,
        "left": left
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def sub_inputs():
    list_of_inputs = []
    
    input = torch.randn(2, 3, 4, 5).numpy()
    other = torch.randn(2, 3, 4, 5).numpy()
    alpha = 1.0
    
    input_dict = {
        "input": input,
        "other": other,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 5, 5).numpy()
    other = torch.randn(1, 5, 5).numpy()
    alpha = 0.5
    
    input_dict = {
        "input": input,
        "other": other,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(3, 2).numpy()
    other = torch.randn(3, 2).numpy()
    alpha = 2.0
    
    input_dict = {
        "input": input,
        "other": other,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4).numpy()
    other = torch.randn(4).numpy()
    alpha = 0.0
    
    input_dict = {
        "input": input,
        "other": other,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 1, 1).numpy()
    other = torch.randn(1, 1, 1).numpy()
    alpha = -1.0
    
    input_dict = {
        "input": input,
        "other": other,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def nll_loss_inputs():
    list_of_inputs = []
    # Example 1
    input = torch.randn(3, 5).log_softmax(dim=1).numpy()
    target = torch.randint(0, 5, (3,)).numpy()
    weight = torch.randn(5).numpy()
    ignore_index = -100
    reduction = 'mean'
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": ignore_index,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 2
    input = torch.randn(2, 10).log_softmax(dim=1).numpy()
    target = torch.randint(0, 10, (2,)).numpy()
    weight = None
    ignore_index = -1
    reduction = 'sum'
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": ignore_index,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 3
    input = torch.randn(5, 3).log_softmax(dim=1).numpy()
    target = torch.randint(0, 3, (5,)).numpy()
    weight = torch.ones(3).numpy()
    ignore_index = 1
    reduction = 'mean'
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": ignore_index,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 4
    input = torch.randn(1, 7).log_softmax(dim=1).numpy()
    target = torch.randint(0, 7, (1,)).numpy()
    weight = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    ignore_index = 6
    reduction = 'mean'
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": ignore_index,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 5
    input = torch.randn(4, 4).log_softmax(dim=1).numpy()
    target = torch.randint(0, 4, (4,)).numpy()
    weight = None
    ignore_index = -100
    reduction = 'mean'
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": ignore_index,
        "reduction": reduction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def flatten_inputs():
    list_of_inputs = []
    input1 = torch.randn(2, 3, 4, 5).numpy()
    start_dim1 = 1
    end_dim1 = 2
    input_dict1 = {
        "input": input1,
        "start_dim": start_dim1,
        "end_dim": end_dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(1, 5, 7).numpy()
    start_dim2 = 0
    end_dim2 = 1
    input_dict2 = {
        "input": input2,
        "start_dim": start_dim2,
        "end_dim": end_dim2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(3, 2, 6).numpy()
    start_dim3 = 0
    end_dim3 = -1
    input_dict3 = {
        "input": input3,
        "start_dim": start_dim3,
        "end_dim": end_dim3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(4, 4, 4).numpy()
    start_dim4 = 0
    end_dim4 = 2
    input_dict4 = {
        "input": input4,
        "start_dim": start_dim4,
        "end_dim": end_dim4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(2, 2, 2, 2).numpy()
    start_dim5 = 1
    end_dim5 = 3
    input_dict5 = {
        "input": input5,
        "start_dim": start_dim5,
        "end_dim": end_dim5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def interpolate_inputs():
    list_of_inputs = []
    input = torch.randn(1, 3, 10, 10).numpy()
    input_dict = {
        "input": input,
        "size": (12, 12)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 4, 5, 5).numpy()
    input_dict = {
        "input": input,
        "scale_factor": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 1, 12, 12).numpy()
    input_dict = {
        "input": input,
        "size": (6, 6)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4, 2, 8, 8).numpy()
    input_dict = {
        "input": input,
        "scale_factor": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(3, 5, 15, 15).numpy()
    input_dict = {
        "input": input,
        "size": (20, 20)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def is_nonzero_inputs():
    list_of_inputs = []
    input1 = torch.randn(1).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.zeros(1).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([0.0]).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([1.0]).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randint(-5,5,(1,)).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def layer_norm_inputs():
    list_of_inputs = []
    input = torch.randn(2, 3, 4).numpy()
    normalized_shape = [4]
    eps = 1e-5
    elementwise_affine = True
    input_dict = {
        "input": input,
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 5, 5, 2).numpy()
    normalized_shape = [2]
    eps = 1e-8
    elementwise_affine = False
    input_dict = {
        "input": input,
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(3, 10).numpy()
    normalized_shape = [10]
    eps = 1e-12
    elementwise_affine = True
    input_dict = {
        "input": input,
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4, 2, 3, 5).numpy()
    normalized_shape = [3, 5]
    eps = 1e-6
    elementwise_affine = False
    input_dict = {
        "input": input,
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 8, 8).numpy()
    normalized_shape = [8, 8]
    eps = 1e-3
    elementwise_affine = True
    input_dict = {
        "input": input,
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def scatter_add_inputs():
    list_of_inputs = []
    input = torch.zeros(5, 3).numpy()
    dim = 0
    index = torch.tensor([[0, 1, 2], [0, 2, 4], [1, 4, 3], [1, 2, 3], [2, 3, 4]]).numpy()
    src = torch.randn(5, 3).numpy()
    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.zeros(3, 5).numpy()
    dim = 1
    index = torch.tensor([[0, 1, 2, 0, 0], [0, 2, 4, 1, 1], [1, 4, 3, 2, 2]]).numpy()
    src = torch.randn(3, 5).numpy()
    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.zeros(2, 3, 4).numpy()
    dim = 0
    index = torch.tensor([[[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]], [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]]).numpy()
    src = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.zeros(2, 3, 4).numpy()
    dim = 1
    index = torch.tensor([[[0, 1, 0, 1], [1, 0, 1, 0]], [[0, 0, 1, 1], [1, 1, 0, 0]]]).numpy()
    src = torch.randn(2, 2, 4).numpy()
    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.zeros(2, 3, 4).numpy()
    dim = 2
    index = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]]).numpy()
    src = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": input,
        "dim": dim,
        "index": index,
        "src": src
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def chunk_inputs():
    list_of_inputs = []
    input = torch.randn(4, 4).numpy()
    chunks = 2
    dim = 0
    input_dict = {
        "input": input,
        "chunks": chunks,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4, 4).numpy()
    chunks = 2
    dim = 1
    input_dict = {
        "input": input,
        "chunks": chunks,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(3, 5, 7).numpy()
    chunks = 3
    dim = 0
    input_dict = {
        "input": input,
        "chunks": chunks,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(3, 5, 7).numpy()
    chunks = 5
    dim = 1
    input_dict = {
        "input": input,
        "chunks": chunks,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 6, 4, 8).numpy()
    chunks = 2
    dim = 3
    input_dict = {
        "input": input,
        "chunks": chunks,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def isclose_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randn(3, 4).numpy()
    rtol1 = 1e-05
    atol1 = 1e-08
    equal_nan1 = False
    input_dict1 = {
        "input": input1,
        "other": other1,
        "rtol": rtol1,
        "atol": atol1,
        "equal_nan": equal_nan1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.tensor([float('nan'), 1.0, 2.0]).numpy()
    other2 = torch.tensor([float('nan'), 1.0, 2.0]).numpy()
    rtol2 = 1e-05
    atol2 = 1e-08
    equal_nan2 = True
    input_dict2 = {
        "input": input2,
        "other": other2,
        "rtol": rtol2,
        "atol": atol2,
        "equal_nan": equal_nan2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other3 = torch.tensor([1.0 + 1e-7, 2.0, 3.0 - 1e-7]).numpy()
    rtol3 = 1e-06
    atol3 = 0.0
    equal_nan3 = False
    input_dict3 = {
        "input": input3,
        "other": other3,
        "rtol": rtol3,
        "atol": atol3,
        "equal_nan": equal_nan3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other4 = torch.tensor([1.0, 2.0 + 1e-9, 3.0]).numpy()
    rtol4 = 0.0
    atol4 = 1e-08
    equal_nan4 = False
    input_dict4 = {
        "input": input4,
        "other": other4,
        "rtol": rtol4,
        "atol": atol4,
        "equal_nan": equal_nan4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(2, 2, dtype=torch.float64).numpy()
    other5 = torch.randn(2, 2, dtype=torch.float64).numpy()
    rtol5 = 1e-08
    atol5 = 1e-05
    equal_nan5 = False
    input_dict5 = {
        "input": input5,
        "other": other5,
        "rtol": rtol5,
        "atol": atol5,
        "equal_nan": equal_nan5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def heaviside_inputs():
    list_of_inputs = []
    input_tensor = torch.randn(3, 4).numpy()
    values_tensor = torch.tensor(0.5).numpy()
    input_dict = {"input": input_tensor, "values": values_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(2, 2, 2).numpy()
    values_tensor = torch.tensor(1.0).numpy()
    input_dict = {"input": input_tensor, "values": values_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(5).numpy()
    values_tensor = torch.tensor(0.0).numpy()
    input_dict = {"input": input_tensor, "values": values_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randint(-5, 5, (2, 3)).float().numpy()
    values_tensor = torch.tensor(2.0).numpy()
    input_dict = {"input": input_tensor, "values": values_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 5, 5).numpy()
    values_tensor = torch.tensor(-1.0).numpy()
    input_dict = {"input": input_tensor, "values": values_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def gt_inputs():
    list_of_inputs = []
    input1 = np.array([[1, 2], [3, 4]])
    other1 = np.array([[0, 3], [3, 1]])
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([5, 6, 7])
    other2 = 6
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[0.1, 0.2], [0.3, 0.4]])
    other3 = np.array([[0.2, 0.1], [0.4, 0.3]])
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([-1, 0, 1])
    other4 = 0
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    other5 = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def hypot_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 4).numpy()
    input2 = torch.randn(3, 4).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randn(5).numpy()
    input2 = torch.randn(5).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randn(2, 2, 2).numpy()
    input2 = torch.randn(2, 2, 2).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = torch.randn(1).numpy()
    input2 = torch.randn(1).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randn(4, 1).numpy()
    input2 = torch.randn(4, 1).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def igamma_inputs():
    list_of_inputs = []
    input1 = torch.randn(2, 3).abs().numpy()
    other1 = torch.randn(2, 3).abs().numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(5).abs().numpy()
    other2 = torch.randn(5).abs().numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(1, 4, 4).abs().numpy()
    other3 = torch.randn(1, 4, 4).abs().numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = (torch.rand(2, 2) * 10).numpy()
    other4 = (torch.rand(2, 2) * 10).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = (torch.rand(3, 1, 5) * 5).numpy()
    other5 = (torch.rand(3, 1, 5) * 5).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def inner_inputs():
    list_of_inputs = []
    input1 = np.random.randn(3).astype(np.float32)
    input2 = np.random.randn(3).astype(np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.random.randn(2, 3).astype(np.float32)
    input2 = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.random.randn(2, 3, 4).astype(np.float32)
    input2 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.random.randn(1, 5).astype(np.float32)
    input2 = np.random.randn(5).astype(np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.random.randn(5).astype(np.float32)
    input2 = np.random.randn(1, 5).astype(np.float32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def lcm_inputs():
    list_of_inputs = []
    input1 = torch.randint(1, 10, (3, 4)).numpy()
    input2 = torch.randint(1, 10, (3, 4)).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randint(1, 10, (2, 2, 2)).numpy()
    input2 = torch.randint(1, 10, (2, 2, 2)).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randint(1, 10, (5,)).numpy()
    input2 = torch.randint(1, 10, (5,)).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = torch.randint(1, 10, (1, 5, 5)).numpy()
    input2 = torch.randint(1, 10, (1, 5, 5)).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randint(1, 10, (2, 3, 4, 5)).numpy()
    input2 = torch.randint(1, 10, (2, 3, 4, 5)).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def le_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randn(3, 4).numpy()
    input_dict1 = {
        "input": input1,
        "other": other1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randint(0, 10, (2, 2)).numpy()
    other2 = 5
    input_dict2 = {
        "input": input2,
        "other": other2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([1, 2, 3, 4, 5])
    other3 = np.array([5, 4, 3, 2, 1])
    input_dict3 = {
        "input": input3,
        "other": other3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(1, 5, 5).numpy()
    other4 = torch.randn(1, 5, 5).numpy()
    input_dict4 = {
        "input": input4,
        "other": other4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other5 = 2.5
    input_dict5 = {
        "input": input5,
        "other": other5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def einsum_inputs():
    list_of_inputs = []
    # Case 1: Matrix multiplication
    a = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(3, 4).astype(np.float32)
    input_dict = {
        "equation": "ij,jk->ik",
        "operands": [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 2: Batch matrix multiplication
    a = np.random.rand(5, 2, 3).astype(np.float32)
    b = np.random.rand(5, 3, 4).astype(np.float32)
    input_dict = {
        "equation": "bij,bjk->bik",
        "operands": [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 3: Trace
    a = np.random.rand(4, 4).astype(np.float32)
    input_dict = {
        "equation": "ii->",
        "operands": [a]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 4: Sum along an axis
    a = np.random.rand(3, 5).astype(np.float32)
    input_dict = {
        "equation": "ij->i",
        "operands": [a]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Case 5: Dot product
    a = np.random.rand(5).astype(np.float32)
    b = np.random.rand(5).astype(np.float32)
    input_dict = {
        "equation": "i,i->",
        "operands": [a, b]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def index_select_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 4, 5).numpy()
    dim1 = 0
    index1 = torch.tensor([0, 2]).long().numpy()
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "index": index1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(2, 2).numpy()
    dim2 = 1
    index2 = torch.tensor([0]).long().numpy()
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "index": index2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(5, 3).numpy()
    dim3 = 0
    index3 = torch.tensor([1, 4, 2]).long().numpy()
    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "index": index3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(2, 3, 4).numpy()
    dim4 = 1
    index4 = torch.tensor([0, 2]).long().numpy()
    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "index": index4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(4, 4).numpy()
    dim5 = 0
    index5 = torch.tensor([3, 1, 0]).long().numpy()
    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "index": index5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def as_strided_inputs():
    list_of_inputs = []
    # Example 1
    input_tensor = torch.arange(1, 7, dtype=torch.float32).reshape(1, 6).numpy()
    size = (1, 3)
    stride = (0, 2)
    storage_offset = 0
    input_dict = {
        "input": input_tensor,
        "size": size,
        "stride": stride,
        "storage_offset": storage_offset
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 2
    input_tensor = torch.arange(1, 17, dtype=torch.float32).reshape(1, 16).numpy()
    size = (2, 3)
    stride = (6, 2)
    storage_offset = 0
    input_dict = {
        "input": input_tensor,
        "size": size,
        "stride": stride,
        "storage_offset": storage_offset
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 3
    input_tensor = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3).numpy()
    size = (2, 2)
    stride = (1, 1)
    storage_offset = 0
    input_dict = {
        "input": input_tensor,
        "size": size,
        "stride": stride,
        "storage_offset": storage_offset
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 4
    input_tensor = torch.arange(24).reshape(2, 3, 4).numpy()
    size = (2, 2, 2)
    stride = (12, 4, 1)
    storage_offset = 0
    input_dict = {
        "input": input_tensor,
        "size": size,
        "stride": stride,
        "storage_offset": storage_offset
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 5
    input_tensor = torch.arange(10).numpy()
    size = (5,)
    stride = (2,)
    storage_offset = 0
    input_dict = {
        "input": input_tensor,
        "size": size,
        "stride": stride,
        "storage_offset": storage_offset
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def broadcast_to_inputs():
    list_of_inputs = []
    # Example 1
    input = np.array([1, 2, 3])
    shape = [3, 3]
    input_dict = {"input": input, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 2
    input = np.array([[1], [2]])
    shape = [2, 3]
    input_dict = {"input": input, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 3
    input = np.array([1])
    shape = [2, 3, 4]
    input_dict = {"input": input, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 4
    input = np.array([[1, 2], [3, 4]])
    shape = [2, 2, 2]
    input_dict = {"input": input, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 5
    input = np.array([[[1], [2]], [[3], [4]]])
    shape = [2, 2, 3]
    input_dict = {"input": input, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def cat_inputs():
    list_of_inputs = []
    # Example 1
    tensors = [torch.randn(2, 3).numpy(), torch.randn(2, 3).numpy()]
    dim = 0
    input_dict = {"tensors": tensors, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 2
    tensors = [torch.randn(2, 3, 4).numpy(), torch.randn(2, 3, 4).numpy(), torch.randn(2, 3, 4).numpy()]
    dim = 1
    input_dict = {"tensors": tensors, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 3
    tensors = [torch.randn(1, 5).numpy(), torch.randn(1, 5).numpy(), torch.randn(1, 5).numpy(), torch.randn(1, 5).numpy()]
    dim = 1
    input_dict = {"tensors": tensors, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 4
    tensors = [torch.randn(3, 1, 2).numpy(), torch.randn(3, 1, 2).numpy()]
    dim = 2
    input_dict = {"tensors": tensors, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Example 5
    tensors = [torch.randn(4, 2, 3).numpy(), torch.randn(4, 2, 3).numpy()]
    dim = 0
    input_dict = {"tensors": tensors, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def flip_inputs():
    list_of_inputs = []
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    dims = (2, 3)
    input_dict = {"input": input_tensor, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(1, 5, 5).numpy()
    dims = (1,)
    input_dict = {"input": input_tensor, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.arange(24).reshape(2, 3, 4).numpy()
    dims = (0, 2)
    input_dict = {"input": input_tensor, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = np.array([[1, 2], [3, 4]]).astype(np.float32)
    dims = (0, 1)
    input_dict = {"input": input_tensor, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(2, 2).numpy()
    dims = (0,)
    input_dict = {"input": input_tensor, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def flatten_inputs_2():
    list_of_inputs = []
    input1 = torch.randn(1, 3, 5, 5).numpy()
    start_dim1 = 1
    end_dim1 = 3
    input_dict1 = {
        "input": input1,
        "start_dim": start_dim1,
        "end_dim": end_dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(2, 4, 6).numpy()
    start_dim2 = 0
    end_dim2 = 1
    input_dict2 = {
        "input": input2,
        "start_dim": start_dim2,
        "end_dim": end_dim2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(10).numpy()
    start_dim3 = 0
    end_dim3 = 0
    input_dict3 = {
        "input": input3,
        "start_dim": start_dim3,
        "end_dim": end_dim3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(2, 3, 4, 5).numpy()
    start_dim4 = 2
    end_dim4 = 2
    input_dict4 = {
        "input": input4,
        "start_dim": start_dim4,
        "end_dim": end_dim4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(1, 2, 3).numpy()
    start_dim5 = 0
    end_dim5 = 2
    input_dict5 = {
        "input": input5,
        "start_dim": start_dim5,
        "end_dim": end_dim5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs
def fractional_max_pool2d_inputs():
    list_of_inputs = []
    input = torch.randn(1, 3, 32, 32).numpy()
    kernel_size = (3, 3)
    output_size = (16, 16)
    output_ratio = None
    return_indices = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 1, 64, 64).numpy()
    kernel_size = (2, 2)
    output_size = None
    output_ratio = (0.5, 0.5)
    return_indices = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 4, 128, 128).numpy()
    kernel_size = (4, 4)
    output_size = (32, 32)
    output_ratio = None
    return_indices = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 3, 256, 256).numpy()
    kernel_size = (5, 5)
    output_size = None
    output_ratio = (0.25, 0.25)
    return_indices = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4, 2, 10, 10).numpy()
    kernel_size = (2, 2)
    output_size = (3, 3)
    output_ratio = None
    return_indices = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def adaptive_avg_pool2d_inputs():
    list_of_inputs = []
    input = torch.randn(1, 3, 32, 32).numpy()
    output_size = 7
    input_dict = {
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 4, 64, 64).numpy()
    output_size = 14
    input_dict = {
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 1, 128, 128).numpy()
    output_size = 28
    input_dict = {
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4, 2, 256, 256).numpy()
    output_size = 56
    input_dict = {
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 8, 512, 512).numpy()
    output_size = 112
    input_dict = {
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def amax_inputs():
    list_of_inputs = []
    input_tensor = torch.randn(2, 3, 4).numpy()
    dim = (0,)
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(2, 3, 4).numpy()
    dim = (1, 2)
    keepdim = True
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(1, 5, 7, 2).numpy()
    dim = (2,)
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(3, 2).numpy()
    dim = (0, 1)
    keepdim = True
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(4, 4, 4, 4).numpy()
    dim = (1, 3)
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def atan2_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 4).numpy()
    input2 = torch.randn(3, 4).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randn(2, 2, 2).numpy()
    input2 = torch.randn(2, 2, 2).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randn(5).numpy()
    input2 = torch.randn(5).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.array(3.14).astype(np.float32)
    input2 = np.array(1.57).astype(np.float32)
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randn(1, 10).numpy()
    input2 = torch.randn(1, 10).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def cosine_similarity_inputs():
    list_of_inputs = []
    x1 = torch.randn(3, 5).numpy()
    x2 = torch.randn(3, 5).numpy()
    dim = 1
    eps = 1e-8
    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = torch.randn(2, 4, 6).numpy()
    x2 = torch.randn(2, 4, 6).numpy()
    dim = 2
    eps = 1e-6
    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = torch.randn(10).numpy()
    x2 = torch.randn(10).numpy()
    dim = 0
    eps = 1e-4
    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = torch.randn(5, 5, 5).numpy()
    x2 = torch.randn(5, 5, 5).numpy()
    dim = 0
    eps = 1e-5
    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = torch.randn(2, 3).numpy()
    x2 = torch.randn(2, 3).numpy()
    dim = 1
    eps = 1e-12
    input_dict = {
        "x1": x1,
        "x2": x2,
        "dim": dim,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def cross_inputs():
    list_of_inputs = []
    input1 = np.array([1, 0, 0])
    other1 = np.array([0, 1, 0])
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = np.array([1, 2, 3])
    other2 = np.array([4, 5, 6])
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = np.array([[1, 2, 3], [4, 5, 6]])
    other3 = np.array([[7, 8, 9], [10, 11, 12]])
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = np.array([1.5, 2.5, 3.5])
    other4 = np.array([4.5, 5.5, 6.5])
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = np.array([0, -1, 1])
    other5 = np.array([1, 1, 0])
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def cross_entropy_inputs():
    list_of_inputs = []
    input_tensor = torch.randn(3, 5).numpy()
    target_tensor = torch.randint(0, 5, (3,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(2, 10).numpy()
    target_tensor = torch.randint(0, 10, (2,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(5, 3).numpy()
    target_tensor = torch.randint(0, 3, (5,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(1, 7).numpy()
    target_tensor = torch.randint(0, 7, (1,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(4, 2).numpy()
    target_tensor = torch.randint(0, 2, (4,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def dist_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 4).numpy()
    input2 = torch.randn(3, 4).numpy()
    p1 = 2.0
    input_dict1 = {
        "input": input1,
        "other": input2,
        "p": p1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input3 = torch.randn(5, 5).numpy()
    input4 = torch.randn(5, 5).numpy()
    p2 = 1.0
    input_dict2 = {
        "input": input3,
        "other": input4,
        "p": p2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input5 = torch.randn(2, 2, 2).numpy()
    input6 = torch.randn(2, 2, 2).numpy()
    p3 = 0.5
    input_dict3 = {
        "input": input5,
        "other": input6,
        "p": p3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input7 = torch.randn(10).numpy()
    input8 = torch.randn(10).numpy()
    p4 = float('inf')
    input_dict4 = {
        "input": input7,
        "other": input8,
        "p": p4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input9 = torch.randn(1, 5).numpy()
    input10 = torch.randn(1, 5).numpy()
    p5 = 3.0
    input_dict5 = {
        "input": input9,
        "other": input10,
        "p": p5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def floor_divide_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randint(10, (2, 2)).numpy()
    other2 = np.array([2]).astype(input2.dtype)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(5).numpy()
    other3 = torch.randn(5).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randint(1, 10, (2,3,4)).numpy()
    other4 = np.array([3]).astype(input4.dtype)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.tensor([5.0, 2.0, 3.0]).numpy()
    other5 = torch.tensor([2.0, 2.0, 2.0]).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs
def fmin_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(5, 2).numpy()
    other2 = torch.randn(5, 2).numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(2, 2, 2).numpy()
    other3 = torch.randn(2, 2, 2).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(4).numpy()
    other4 = torch.randn(4).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(1, 5, 5).numpy()
    other5 = torch.randn(1, 5, 5).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def ger_inputs():
    list_of_inputs = []
    vec1 = torch.randn(5).numpy()
    vec2 = torch.randn(3).numpy()
    input_dict = {
        "vec1": vec1,
        "vec2": vec2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    vec1 = torch.randn(10).numpy()
    vec2 = torch.randn(1).numpy()
    input_dict = {
        "vec1": vec1,
        "vec2": vec2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    vec1 = torch.randn(1).numpy()
    vec2 = torch.randn(7).numpy()
    input_dict = {
        "vec1": vec1,
        "vec2": vec2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    vec1 = torch.randn(4).numpy()
    vec2 = torch.randn(5).numpy()
    input_dict = {
        "vec1": vec1,
        "vec2": vec2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    vec1 = torch.randn(4).numpy()
    vec2 = torch.randn(4).numpy()
    input_dict = {
        "vec1": vec1,
        "vec2": vec2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def groupnorm_inputs():
    list_of_inputs = []
    input = torch.randn(2, 6, 5, 5).numpy()
    num_groups = 3
    num_channels = 6
    eps = 1e-5
    affine = True
    input_dict = {
        "input": input,
        "num_groups": num_groups,
        "num_channels": num_channels,
        "eps": eps,
        "affine": affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 12, 10, 10).numpy()
    num_groups = 4
    num_channels = 12
    eps = 1e-8
    affine = False
    input_dict = {
        "input": input,
        "num_groups": num_groups,
        "num_channels": num_channels,
        "eps": eps,
        "affine": affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4, 8, 8, 8).numpy()
    num_groups = 2
    num_channels = 8
    eps = 1e-3
    affine = True
    input_dict = {
        "input": input,
        "num_groups": num_groups,
        "num_channels": num_channels,
        "eps": eps,
        "affine": affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 16, 12, 12).numpy()
    num_groups = 8
    num_channels = 16
    eps = 1e-6
    affine = False
    input_dict = {
        "input": input,
        "num_groups": num_groups,
        "num_channels": num_channels,
        "eps": eps,
        "affine": affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(3, 4, 7, 7).numpy()
    num_groups = 1
    num_channels = 4
    eps = 1e-7
    affine = True
    input_dict = {
        "input": input,
        "num_groups": num_groups,
        "num_channels": num_channels,
        "eps": eps,
        "affine": affine
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def maxpool3d_inputs():
    list_of_inputs = []
    input = torch.randn(1, 3, 10, 10, 10).numpy()
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    ceil_mode = False
    return_indices = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 4, 5, 5, 5).numpy()
    kernel_size = 2
    stride = 1
    padding = 0
    dilation = 1
    ceil_mode = True
    return_indices = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 2, 8, 8, 8).numpy()
    kernel_size = 4
    stride = 3
    padding = 2
    dilation = 2
    ceil_mode = False
    return_indices = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(1, 1, 7, 7, 7).numpy()
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    ceil_mode = True
    return_indices = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 8, 12, 12, 12).numpy()
    kernel_size = 5
    stride = 2
    padding = 1
    dilation = 1
    ceil_mode = False
    return_indices = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def pixel_shuffle_inputs():
    list_of_inputs = []
    input = torch.randn(1, 9, 4, 4).numpy()
    upscale_factor = 3
    input_dict = {
        "input": input,
        "upscale_factor": upscale_factor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 4, 8, 8).numpy()
    upscale_factor = 2
    input_dict = {
        "input": input,
        "upscale_factor": upscale_factor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 16, 2, 2).numpy()
    upscale_factor = 4
    input_dict = {
        "input": input,
        "upscale_factor": upscale_factor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4, 25, 5, 5).numpy()
    upscale_factor = 5
    input_dict = {
        "input": input,
        "upscale_factor": upscale_factor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 49, 7, 7).numpy()
    upscale_factor = 7
    input_dict = {
        "input": input,
        "upscale_factor": upscale_factor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def adaptive_max_pool2d_inputs():
    list_of_inputs = []
    input = torch.randn(1, 3, 32, 32).numpy()
    output_size = 7
    input_dict = {
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 4, 64, 64).numpy()
    output_size = 14
    input_dict = {
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 1, 128, 128).numpy()
    output_size = 28
    input_dict = {
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4, 2, 256, 256).numpy()
    output_size = 56
    input_dict = {
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 8, 512, 512).numpy()
    output_size = 112
    input_dict = {
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def alpha_dropout_inputs():
    list_of_inputs = []
    input = torch.randn(2, 3, 4, 5).numpy()
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
    input = torch.randn(1, 5, 5).numpy()
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
    input = torch.randn(3, 2).numpy()
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
    input = torch.randn(10).numpy()
    p = 0.1
    training = False
    inplace = True
    input_dict = {
        "input": input,
        "p": p,
        "training": training,
        "inplace": inplace
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 2, 2, 2, 2).numpy()
    p = 0.3
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
def bitwise_and_inputs():
    list_of_inputs = []
    input1 = torch.tensor([1, 2, 3, 4], dtype=torch.int32).numpy()
    input2 = torch.tensor([0, 2, 5, 8], dtype=torch.int32).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8).numpy()
    input2 = torch.tensor([[0, 2], [5, 0]], dtype=torch.uint8).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.tensor([True, False, True], dtype=torch.bool).numpy()
    input2 = torch.tensor([False, True, True], dtype=torch.bool).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randint(0, 10, (2, 2), dtype=torch.int64).numpy()
    input2 = torch.randint(0, 10, (2, 2), dtype=torch.int64).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.array([1,2,3], dtype=np.int16)
    input2 = np.array([3,2,1], dtype=np.int16)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def layernorm_inputs():
    list_of_inputs = []
    input = torch.randn(2, 3, 4).numpy()
    normalized_shape = [3, 4]
    eps = 1e-5
    elementwise_affine = True
    input_dict = {
        "input": input,
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine,
        "bias": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 5).numpy()
    normalized_shape = [5]
    eps = 1e-8
    elementwise_affine = False
    input_dict = {
        "input": input,
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine,
        "bias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 2, 3, 4).numpy()
    normalized_shape = [4]
    eps = 1e-6
    elementwise_affine = True
    input_dict = {
        "input": input,
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine,
        "bias": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 4, 5, 6).numpy()
    normalized_shape = [5, 6]
    eps = 1e-5
    elementwise_affine = False
    input_dict = {
        "input": input,
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine,
        "bias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(3, 2, 2).numpy()
    normalized_shape = [2,2]
    eps = 1e-4
    elementwise_affine = True
    input_dict = {
        "input": input,
        "normalized_shape": normalized_shape,
        "eps": eps,
        "elementwise_affine": elementwise_affine,
        "bias": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def linear_inputs():
    list_of_inputs = []
    input = torch.randn(1, 5).numpy()
    in_features = 5
    out_features = 3
    bias = True
    
    input_dict = {
        "input": input,
        "in_features": in_features,
        "out_features": out_features,
        "bias": bias
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 10).numpy()
    in_features = 10
    out_features = 7
    bias = False
    
    input_dict = {
        "input": input,
        "in_features": in_features,
        "out_features": out_features,
        "bias": bias
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 1).numpy()
    in_features = 1
    out_features = 1
    bias = True
    
    input_dict = {
        "input": input,
        "in_features": in_features,
        "out_features": out_features,
        "bias": bias
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4, 20).numpy()
    in_features = 20
    out_features = 15
    bias = False
    
    input_dict = {
        "input": input,
        "in_features": in_features,
        "out_features": out_features,
        "bias": bias
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(3, 8).numpy()
    in_features = 8
    out_features = 12
    bias = True
    
    input_dict = {
        "input": input,
        "in_features": in_features,
        "out_features": out_features,
        "bias": bias
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def maxpool2d_inputs():
    list_of_inputs = []
    input = torch.randn(1, 3, 32, 32).numpy()
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    ceil_mode = False
    return_indices = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 1, 16, 16).numpy()
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    ceil_mode = False
    return_indices = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 3, 64, 64).numpy()
    kernel_size = 4
    stride = 2
    padding = 1
    dilation = 1
    ceil_mode = True
    return_indices = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 1, 28, 28).numpy()
    kernel_size = 2
    stride = None
    padding = 0
    dilation = 1
    ceil_mode = False
    return_indices = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": kernel_size,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(1, 3, 12, 12).numpy()
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 2
    ceil_mode = False
    return_indices = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def prelu_inputs():
    list_of_inputs = []
    input = torch.randn(1, 3, 5, 5).numpy()
    num_parameters = 1
    init = 0.25
    input_dict = {
        "input": input,
        "num_parameters": num_parameters,
        "init": init
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 4, 4, 4).numpy()
    num_parameters = 4
    init = 0.0
    input_dict = {
        "input": input,
        "num_parameters": num_parameters,
        "init": init
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 2, 6, 6).numpy()
    num_parameters = 2
    init = 0.5
    input_dict = {
        "input": input,
        "num_parameters": num_parameters,
        "init": init
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 1, 5, 5).numpy()
    num_parameters = 1
    init = -0.25
    input_dict = {
        "input": input,
        "num_parameters": num_parameters,
        "init": init
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 8, 10, 10).numpy()
    num_parameters = 8
    init = 1.0
    input_dict = {
        "input": input,
        "num_parameters": num_parameters,
        "init": init
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def softmax_inputs():
    list_of_inputs = []
    input1 = torch.randn(2, 3).numpy()
    dim1 = 1
    input_dict1 = {
        "input": input1,
        "dim": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(5).numpy()
    dim2 = 0
    input_dict2 = {
        "input": input2,
        "dim": dim2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(2, 2, 2).numpy()
    dim3 = 2
    input_dict3 = {
        "input": input3,
        "dim": dim3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(1, 5).numpy()
    dim4 = 1
    input_dict4 = {
        "input": input4,
        "dim": dim4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(3, 4, 5).numpy()
    dim5 = 0
    input_dict5 = {
        "input": input5,
        "dim": dim5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def softmin_inputs():
    list_of_inputs = []
    input1 = torch.randn(2, 3).numpy()
    dim1 = 1
    input_dict1 = {
        "input": input1,
        "dim": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(5,).numpy()
    dim2 = 0
    input_dict2 = {
        "input": input2,
        "dim": dim2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(2, 2, 2).numpy()
    dim3 = 2
    input_dict3 = {
        "input": input3,
        "dim": dim3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(3, 4, 5).numpy()
    dim4 = 0
    input_dict4 = {
        "input": input4,
        "dim": dim4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(1, 10).numpy()
    dim5 = 1
    input_dict5 = {
        "input": input5,
        "dim": dim5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs
def bincount_inputs():
    list_of_inputs = []
    input = np.array([1, 2, 3, 4, 4, 1]).astype(np.int64)
    weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).astype(np.float32)
    minlength = 7
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = np.array([0, 1, 1, 3, 2, 1, 7]).astype(np.int64)
    weights = np.array([0.5, 0.2, 0.1, 0.8, 0.3, 0.9, 0.4]).astype(np.float32)
    minlength = 10
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = np.array([0, 0, 1, 2, 3, 3, 3]).astype(np.int64)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).astype(np.float32)
    minlength = 5
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = np.array([2, 2, 1, 1, 0, 1, 2]).astype(np.int64)
    weights = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]).astype(np.float32)
    minlength = 4
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = np.array([5, 4, 3, 2, 1, 0, 0]).astype(np.int64)
    weights = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]).astype(np.float32)
    minlength = 8
    input_dict = {
        "input": input,
        "weights": weights,
        "minlength": minlength
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def bitwise_or_inputs():
    list_of_inputs = []
    input1 = torch.tensor([1, 2, 3, 4], dtype=torch.int32).numpy()
    other1 = torch.tensor([4, 3, 2, 1], dtype=torch.int32).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.tensor([[1, 0], [0, 1]], dtype=torch.int8).numpy()
    other2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.int8).numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.tensor([True, False, True, False]).numpy()
    other3 = torch.tensor([False, True, False, True]).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randint(0, 10, (2, 2), dtype=torch.int64).numpy()
    other4 = torch.randint(0, 10, (2, 2), dtype=torch.int64).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.tensor([1, 2, 3, 4], dtype=torch.uint8).numpy()
    other5 = torch.tensor([4, 3, 2, 1], dtype=torch.uint8).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def bitwise_xor_inputs():
    list_of_inputs = []
    input1 = torch.randint(0, 2, (2, 3), dtype=torch.int8).numpy()
    input2 = torch.randint(0, 2, (2, 3), dtype=torch.int8).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randint(0, 256, (4, 4), dtype=torch.int16).numpy()
    input2 = torch.randint(0, 256, (4, 4), dtype=torch.int16).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randint(0, 1024, (1, 5, 5), dtype=torch.int32).numpy()
    input2 = torch.randint(0, 1024, (1, 5, 5), dtype=torch.int32).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randint(0, 65536, (2, 2, 2, 2), dtype=torch.int64).numpy()
    input2 = torch.randint(0, 65536, (2, 2, 2, 2), dtype=torch.int64).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randint(0, 2, (3, 3), dtype=torch.uint8).numpy()
    input2 = torch.randint(0, 2, (3, 3), dtype=torch.uint8).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def bmm_inputs():
    list_of_inputs = []
    input1 = torch.randn(10, 3, 4).numpy()
    input2 = torch.randn(10, 4, 5).numpy()
    input_dict = {
        "input": input1,
        "mat2": input2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randn(5, 2, 2).numpy()
    input2 = torch.randn(5, 2, 3).numpy()
    input_dict = {
        "input": input1,
        "mat2": input2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randn(2, 1, 7).numpy()
    input2 = torch.randn(2, 7, 1).numpy()
    input_dict = {
        "input": input1,
        "mat2": input2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randn(8, 5, 10).numpy()
    input2 = torch.randn(8, 10, 5).numpy()
    input_dict = {
        "input": input1,
        "mat2": input2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = torch.randn(3, 6, 6).numpy()
    input2 = torch.randn(3, 6, 6).numpy()
    input_dict = {
        "input": input1,
        "mat2": input2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def cdist_inputs():
    list_of_inputs = []
    x1 = torch.randn(5, 3).numpy()
    x2 = torch.randn(7, 3).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = torch.randn(10, 2).numpy()
    x2 = torch.randn(5, 2).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = torch.randn(2, 4, 3).numpy()
    x2 = torch.randn(2, 5, 3).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = torch.randn(1, 8).numpy()
    x2 = torch.randn(1, 8).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = torch.randn(3, 1, 5).numpy()
    x2 = torch.randn(3, 1, 5).numpy()
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def complex_inputs():
    list_of_inputs = []
    real = torch.randn(3, 4).numpy()
    imag = torch.randn(3, 4).numpy()
    input_dict = {"real": real, "imag": imag}
    list_of_inputs.append(copy.deepcopy(input_dict))
    real = torch.randn(2, 2, 2).numpy()
    imag = torch.randn(2, 2, 2).numpy()
    input_dict = {"real": real, "imag": imag}
    list_of_inputs.append(copy.deepcopy(input_dict))
    real = np.array([1.0, 2.0, 3.0])
    imag = np.array([4.0, 5.0, 6.0])
    input_dict = {"real": real, "imag": imag}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    real = np.array([[1.0, 2.0], [3.0, 4.0]])
    imag = np.array([[5.0, 6.0], [7.0, 8.0]])
    input_dict = {"real": real, "imag": imag}
    list_of_inputs.append(copy.deepcopy(input_dict))
    real = torch.zeros(5).numpy()
    imag = torch.ones(5).numpy()
    input_dict = {"real": real, "imag": imag}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def copysign_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(2, 2, 2).numpy()
    other2 = torch.randn(2, 2, 2).numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(5).numpy()
    other3 = torch.randn(5).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.randn(1, 5, 5).numpy()
    other4 = torch.randn(1, 5, 5).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(2, 3, 4, 5).numpy()
    other5 = torch.randn(2, 3, 4, 5).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def dot_inputs():
    list_of_inputs = []
    input1 = np.array([1, 2, 3])
    input2 = np.array([4, 5, 6])
    input_dict = {"input": input1, "tensor": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.array([1.5, 2.5, 3.5])
    input2 = np.array([4.5, 5.5, 6.5])
    input_dict = {"input": input1, "tensor": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.array([-1, -2, -3])
    input2 = np.array([4, 5, 6])
    input_dict = {"input": input1, "tensor": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.array([0, 0, 0])
    input2 = np.array([1, 2, 3])
    input_dict = {"input": input1, "tensor": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.array([1, 1, 1])
    input2 = np.array([1, 1, 1])
    input_dict = {"input": input1, "tensor": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def eq_inputs():
    list_of_inputs = []
    input1 = torch.randn(2, 3).numpy()
    input2 = torch.randn(2, 3).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.tensor([1, 2, 3]).numpy()
    input2 = torch.tensor([1, 2, 4]).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = np.array([1.0, 2.0, 3.0])
    input2 = np.array([1.0, 2.0, 3.0])
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randint(0, 10, (5, 5)).numpy()
    input2 = torch.randint(0, 10, (5, 5)).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randn(1, 10, 10).numpy()
    input2 = torch.randn(1, 10, 10).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def float_power_inputs():
    list_of_inputs = []
    input1 = torch.randn(2, 3).numpy()
    exponent1 = torch.tensor(2.0).numpy()
    input_dict1 = {"input": input1, "exponent": exponent1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randn(4, 4).numpy()
    exponent2 = torch.randn(4, 4).numpy()
    input_dict2 = {"input": input2, "exponent": exponent2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(1, 5, 5).numpy()
    exponent3 = torch.tensor(0.5).numpy()
    input_dict3 = {"input": input3, "exponent": exponent3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([1.0, 2.0, 3.0])
    exponent4 = np.array([2.0, 3.0, 4.0])
    input_dict4 = {"input": input4, "exponent": exponent4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.ones(2, 2).numpy()
    exponent5 = torch.full((2, 2), 3.5).numpy()
    input_dict5 = {"input": input5, "exponent": exponent5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def ge_inputs():
    list_of_inputs = []
    input1 = torch.randn(2, 3).numpy()
    other1 = torch.randn(2, 3).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    input2 = torch.randint(0, 10, (5,)).numpy()
    other2 = np.array(5)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    input3 = torch.randn(1, 4, 4).numpy()
    other3 = torch.ones(1, 4, 4).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    input4 = torch.arange(5).numpy()
    other4 = torch.tensor([2, 3, 4, 1, 0]).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    input5 = torch.randn(2, 1, 3).numpy()
    other5 = torch.randn(1, 3).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def allclose_inputs():
    list_of_inputs = []
    input1 = torch.randn(3, 4).double().numpy()
    input2 = input1 + np.random.normal(0, 1e-7, input1.shape)
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 1e-08,
        "rtol": 1e-05,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randn(2, 2).double().numpy()
    input2 = input1 + np.random.normal(0, 1e-6, input1.shape)
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 1e-07,
        "rtol": 1e-05,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.tensor([float('nan'), 1.0, 2.0], dtype=torch.float64).numpy()
    input2 = torch.tensor([float('nan'), 1.0, 2.0], dtype=torch.float64).numpy()
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 1e-08,
        "rtol": 1e-05,
        "equal_nan": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input1 = torch.randn(5, 5).double().numpy()
    input2 = input1 + 1
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 1.1,
        "rtol": 1e-05,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = torch.randn(2, 3, 4).double().numpy()
    input2 = input1 * (1 + 1e-4)
    input_dict = {
        "input": input1,
        "other": input2,
        "atol": 1e-08,
        "rtol": 1e-03,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def l1loss_inputs():
    list_of_inputs = []
    input_tensor = torch.randn(3, 5).numpy()
    target_tensor = torch.randn(3, 5).numpy()
    reduction_mode = 'mean'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(2, 4, 6).numpy()
    target_tensor = torch.randn(2, 4, 6).numpy()
    reduction_mode = 'sum'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(1, 10, 10).numpy()
    target_tensor = torch.randn(1, 10, 10).numpy()
    reduction_mode = 'mean'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(5,).numpy()
    target_tensor = torch.randn(5,).numpy()
    reduction_mode = 'mean'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 2, 2, 2).numpy()
    target_tensor = torch.randn(2, 2, 2, 2).numpy()
    reduction_mode = 'mean'
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "reduction": reduction_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def MSELoss_inputs():
    list_of_inputs = []
    input_tensor = torch.randn(3, 5).numpy()
    target_tensor = torch.randn(3, 5).numpy()
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(2, 4, 3).numpy()
    target_tensor = torch.randn(2, 4, 3).numpy()
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": False,
        "reduce": True,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(1, 10).numpy()
    target_tensor = torch.randn(1, 10).numpy()
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(4, 4).numpy()
    target_tensor = torch.randn(4, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 3, 2, 2).numpy()
    target_tensor = torch.randn(2, 3, 2, 2).numpy()
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def multimarginloss_inputs():
    list_of_inputs = []
    input = torch.randn(3, 5).numpy()
    target = torch.randint(1, 5, (3,)).numpy()
    input_dict = {
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(5, 10).numpy()
    target = torch.randint(1, 10, (5,)).numpy()
    input_dict = {
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(10, 3).numpy()
    target = torch.randint(1, 3, (10,)).numpy()
    input_dict = {
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 4).numpy()
    target = torch.randint(1, 4, (2,)).numpy()
    input_dict = {
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4, 2).numpy()
    target = torch.randint(1, 2, (4,)).numpy()
    input_dict = {
        "input": input,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def pairwise_distance_inputs():
    list_of_inputs = []
    x1 = torch.randn(10, 128).numpy()
    x2 = torch.randn(10, 128).numpy()
    p = 2.0
    eps = 1e-6
    keepdim = False
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": p,
        "eps": eps,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = torch.randn(5, 64).numpy()
    x2 = torch.randn(5, 64).numpy()
    p = 1.5
    eps = 1e-8
    keepdim = True
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": p,
        "eps": eps,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = torch.randn(20, 256).numpy()
    x2 = torch.randn(20, 256).numpy()
    p = 3.0
    eps = 1e-4
    keepdim = False
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": p,
        "eps": eps,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = torch.randn(2, 32).numpy()
    x2 = torch.randn(2, 32).numpy()
    p = 2.5
    eps = 1e-5
    keepdim = True
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": p,
        "eps": eps,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    x1 = torch.randn(8, 512).numpy()
    x2 = torch.randn(8, 512).numpy()
    p = 1.0
    eps = 1e-7
    keepdim = False
    input_dict = {
        "x1": x1,
        "x2": x2,
        "p": p,
        "eps": eps,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def poisson_nll_loss_inputs():
    list_of_inputs = []
    input_dict = {
        "input": torch.randn(3, 5).numpy(),
        "target": torch.randint(0, 10, (3, 5)).float().numpy(),
        "log_input": False,
        "full": False,
        "eps": 1e-8,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "input": torch.randn(2, 4, 3).numpy(),
        "target": torch.randint(0, 5, (2, 4, 3)).float().numpy(),
        "log_input": True,
        "full": True,
        "eps": 1e-6,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "input": torch.randn(1, 7).numpy(),
        "target": torch.randint(0, 15, (1, 7)).float().numpy(),
        "log_input": False,
        "full": True,
        "eps": 1e-10,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "input": torch.randn(4, 2, 2).numpy(),
        "target": torch.randint(0, 8, (4, 2, 2)).float().numpy(),
        "log_input": True,
        "full": False,
        "eps": 1e-4,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_dict = {
        "input": torch.randn(2, 3, 4, 5).numpy(),
        "target": torch.randint(0, 3, (2, 3, 4, 5)).float().numpy(),
        "log_input": False,
        "full": False,
        "eps": 1e-5,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def addmm_inputs():
    list_of_inputs = []
    input = torch.randn(3, 5).numpy()
    mat1 = torch.randn(3, 4).numpy()
    mat2 = torch.randn(4, 5).numpy()
    beta = 1.0
    alpha = 1.0
    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 3).numpy()
    mat1 = torch.randn(2, 4).numpy()
    mat2 = torch.randn(4, 3).numpy()
    beta = 0.5
    alpha = 2.0
    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 1).numpy()
    mat1 = torch.randn(1, 5).numpy()
    mat2 = torch.randn(5, 1).numpy()
    beta = 0.0
    alpha = 1.0
    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4, 4).numpy()
    mat1 = torch.randn(4, 2).numpy()
    mat2 = torch.randn(2, 4).numpy()
    beta = -1.0
    alpha = 0.5
    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(5, 2).numpy()
    mat1 = torch.randn(5, 3).numpy()
    mat2 = torch.randn(3, 2).numpy()
    beta = 2.0
    alpha = -1.0
    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def embedding_bag_inputs():
    list_of_inputs = []
    # Example 1
    input1 = torch.tensor([1, 2, 4, 5, 4, 3, 0]).numpy()
    weight1 = torch.randn(10, 3).numpy()
    offsets1 = torch.tensor([0, 1, 2, 5]).numpy()
    input_dict1 = {
        "input": input1,
        "weight": weight1,
        "offsets": offsets1,
        "max_norm": None,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
        "padding_idx": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    # Example 2
    input2 = torch.tensor([0, 1, 2, 3]).numpy()
    weight2 = torch.randn(5, 4).numpy()
    offsets2 = torch.tensor([0, 2]).numpy()
    input_dict2 = {
        "input": input2,
        "weight": weight2,
        "offsets": offsets2,
        "max_norm": 1.0,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "mean",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
        "padding_idx": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    # Example 3
    input3 = torch.tensor([1, 2, 3, 4, 0, 1]).numpy()
    weight3 = torch.randn(6, 2).numpy()
    offsets3 = torch.tensor([0, 3]).numpy()
    per_sample_weights3 = torch.randn(6).numpy()
    input_dict3 = {
        "input": input3,
        "weight": weight3,
        "offsets": offsets3,
        "max_norm": None,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": True,
        "per_sample_weights": per_sample_weights3,
        "include_last_offset": False,
        "padding_idx": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    # Example 4
    input4 = torch.tensor([0, 1, 2, 0]).numpy()
    weight4 = torch.randn(4, 5).numpy()
    offsets4 = torch.tensor([0, 2, 4]).numpy()
    input_dict4 = {
        "input": input4,
        "weight": weight4,
        "offsets": offsets4,
        "max_norm": 2.0,
        "norm_type": 1,
        "scale_grad_by_freq": False,
        "mode": "max",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
        "padding_idx": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Example 5
    input5 = torch.tensor([0, 1, 2, 3, 0, 1, 2]).numpy()
    weight5 = torch.randn(7, 3).numpy()
    offsets5 = torch.tensor([0, 3, 5]).numpy()
    input_dict5 = {
        "input": input5,
        "weight": weight5,
        "offsets": offsets5,
        "max_norm": None,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": True,
        "padding_idx": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs
def binary_cross_entropy_with_logits_inputs():
    list_of_inputs = []
    input_tensor = torch.randn(3, requires_grad=False).numpy()
    target_tensor = torch.randint(0, 2, (3,), dtype=torch.float32).numpy()
    weight_tensor = torch.randn(3, requires_grad=False).numpy()
    pos_weight_tensor = torch.randn(1, requires_grad=False).numpy()
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "reduction": 'mean',
        "pos_weight": pos_weight_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(2, 5, requires_grad=False).numpy()
    target_tensor = torch.randint(0, 2, (2, 5), dtype=torch.float32).numpy()
    weight_tensor = torch.randn(2, 5, requires_grad=False).numpy()
    pos_weight_tensor = torch.randn(1, requires_grad=False).numpy()
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "reduction": 'sum',
        "pos_weight": pos_weight_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(1, 4, 4, requires_grad=False).numpy()
    target_tensor = torch.randint(0, 2, (1, 4, 4), dtype=torch.float32).numpy()
    weight_tensor = torch.randn(1, 4, 4, requires_grad=False).numpy()
    pos_weight_tensor = torch.randn(1, requires_grad=False).numpy()
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "reduction": 'mean',
        "pos_weight": pos_weight_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(5, requires_grad=False).numpy()
    target_tensor = torch.randint(0, 2, (5,), dtype=torch.float32).numpy()
    weight_tensor = None
    pos_weight_tensor = None
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "reduction": 'mean',
        "pos_weight": pos_weight_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(2, 3, 4, 5, requires_grad=False).numpy()
    target_tensor = torch.randint(0, 2, (2, 3, 4, 5), dtype=torch.float32).numpy()
    weight_tensor = torch.randn(2, 3, 4, 5, requires_grad=False).numpy()
    pos_weight_tensor = torch.tensor([2.0], requires_grad=False).numpy()
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "weight": weight_tensor,
        "reduction": 'mean',
        "pos_weight": pos_weight_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def addcdiv_inputs():
    list_of_inputs = []
    input = torch.randn(3, 4).numpy()
    tensor1 = torch.randn(3, 4).numpy()
    tensor2 = torch.randn(3, 4).numpy()
    value = 2.0
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 2, 2).numpy()
    tensor1 = torch.randn(2, 2, 2).numpy()
    tensor2 = torch.randn(2, 2, 2).numpy()
    value = 0.5
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(5).numpy()
    tensor1 = torch.randn(5).numpy()
    tensor2 = torch.randn(5).numpy()
    value = -1.0
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 5, 5).numpy()
    tensor1 = torch.randn(1, 5, 5).numpy()
    tensor2 = torch.randn(1, 5, 5).numpy()
    value = 1.5
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 3, 4, 5).numpy()
    tensor1 = torch.randn(2, 3, 4, 5).numpy()
    tensor2 = torch.randn(2, 3, 4, 5).numpy()
    value = 0.0
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def addmv_inputs():
    list_of_inputs = []
    input = torch.randn(5).numpy()
    mat = torch.randn(5, 3).numpy()
    vec = torch.randn(3).numpy()
    beta = 1.0
    alpha = 1.0
    input_dict = {
        "input": input,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4).numpy()
    mat = torch.randn(4, 2).numpy()
    vec = torch.randn(2).numpy()
    beta = 0.5
    alpha = 2.0
    input_dict = {
        "input": input,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(6).numpy()
    mat = torch.randn(6, 4).numpy()
    vec = torch.randn(4).numpy()
    beta = 0.0
    alpha = 1.0
    input_dict = {
        "input": input,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(3).numpy()
    mat = torch.randn(3, 5).numpy()
    vec = torch.randn(5).numpy()
    beta = 1.5
    alpha = 0.5
    input_dict = {
        "input": input,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(7).numpy()
    mat = torch.randn(7, 3).numpy()
    vec = torch.randn(3).numpy()
    beta = 0.8
    alpha = 1.2
    input_dict = {
        "input": input,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def addr_inputs():
    list_of_inputs = []
    input_tensor = torch.randn(3, 3).numpy()
    vec1 = torch.randn(3).numpy()
    vec2 = torch.randn(3).numpy()
    beta = 1.0
    alpha = 1.0
    
    input_dict = {
        "input": input_tensor,
        "vec1": vec1,
        "vec2": vec2,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(2, 4).numpy()
    vec1 = torch.randn(2).numpy()
    vec2 = torch.randn(4).numpy()
    beta = 0.5
    alpha = 2.0
    
    input_dict = {
        "input": input_tensor,
        "vec1": vec1,
        "vec2": vec2,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(5, 5).numpy()
    vec1 = torch.randn(5).numpy()
    vec2 = torch.randn(5).numpy()
    beta = 0.0
    alpha = -1.0
    
    input_dict = {
        "input": input_tensor,
        "vec1": vec1,
        "vec2": vec2,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input_tensor = torch.randn(4, 2).numpy()
    vec1 = torch.randn(4).numpy()
    vec2 = torch.randn(2).numpy()
    beta = -0.5
    alpha = 0.5
    
    input_dict = {
        "input": input_tensor,
        "vec1": vec1,
        "vec2": vec2,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 1).numpy()
    vec1 = torch.randn(1).numpy()
    vec2 = torch.randn(1).numpy()
    beta = 2.0
    alpha = 3.0
    
    input_dict = {
        "input": input_tensor,
        "vec1": vec1,
        "vec2": vec2,
        "beta": beta,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def multilabel_soft_margin_loss_inputs():
    list_of_inputs = []
    input = torch.randn(3, 5).numpy()
    target = torch.empty(3, 5).random_(0, 2).numpy()
    weight = torch.randn(5).numpy()
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
    input = torch.randn(2, 4).numpy()
    target = torch.empty(2, 4).random_(0, 2).numpy()
    weight = None
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
    input = torch.randn(1, 3).numpy()
    target = torch.empty(1, 3).random_(0, 2).numpy()
    weight = torch.randn(3).numpy()
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
    input = torch.randn(4, 2).numpy()
    target = torch.empty(4, 2).random_(0, 2).numpy()
    weight = None
    size_average = False
    reduce = False
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
    input = torch.randn(5, 6).numpy()
    target = torch.empty(5, 6).random_(0, 2).numpy()
    weight = torch.randn(6).numpy()
    size_average = True
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
    return list_of_inputs
def nllloss_inputs():
    list_of_inputs = []
    input = torch.randn(3, 5).log_softmax(dim=1).numpy()
    target = torch.tensor([1, 0, 4]).numpy()
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).float().numpy()
    ignore_index = -100
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": ignore_index,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(2, 3).log_softmax(dim=1).numpy()
    target = torch.tensor([0, 2]).numpy()
    weight = None
    ignore_index = -1
    reduction = 'sum'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": ignore_index,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(4, 4).log_softmax(dim=1).numpy()
    target = torch.tensor([3, 1, 2, 0]).numpy()
    weight = torch.tensor([0.25, 0.25, 0.25, 0.25]).float().numpy()
    ignore_index = 5
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": ignore_index,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    input = torch.randn(1, 10).log_softmax(dim=1).numpy()
    target = torch.tensor([7]).numpy()
    weight = None
    ignore_index = 7
    reduction = 'mean'
    
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": ignore_index,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(2, 2).log_softmax(dim=1).numpy()
    target = torch.tensor([0, 1]).numpy()
    weight = torch.tensor([0.7, 0.3]).float().numpy()
    ignore_index = -100
    reduction = 'mean'
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": ignore_index,
        "reduction": reduction
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def batch_norm_inputs():
    list_of_inputs = []
    # Input 1
    input = torch.randn(2, 3, 4, 5).numpy()
    num_features = input.shape[1]
    running_mean = torch.randn(num_features).numpy()
    running_var = torch.rand(num_features).numpy()
    weight = torch.randn(num_features).numpy()
    bias = torch.randn(num_features).numpy()
    training = True
    momentum = 0.1
    eps = 1e-5
    input_dict = {
        "input": input,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": training,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2
    input = torch.randn(1, 5, 6, 7).numpy()
    num_features = input.shape[1]
    running_mean = torch.randn(num_features).numpy()
    running_var = torch.rand(num_features).numpy()
    weight = torch.randn(num_features).numpy()
    bias = torch.randn(num_features).numpy()
    training = False
    momentum = 0.2
    eps = 1e-4
    input_dict = {
        "input": input,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": training,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3
    input = torch.randn(3, 2, 8, 9).numpy()
    num_features = input.shape[1]
    running_mean = torch.randn(num_features).numpy()
    running_var = torch.rand(num_features).numpy()
    weight = torch.randn(num_features).numpy()
    bias = torch.randn(num_features).numpy()
    training = True
    momentum = 0.05
    eps = 1e-6
    input_dict = {
        "input": input,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": training,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4
    input = torch.randn(4, 4, 10, 11).numpy()
    num_features = input.shape[1]
    running_mean = torch.randn(num_features).numpy()
    running_var = torch.rand(num_features).numpy()
    weight = torch.randn(num_features).numpy()
    bias = torch.randn(num_features).numpy()
    training = False
    momentum = 0.15
    eps = 1e-3
    input_dict = {
        "input": input,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": training,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5
    input = torch.randn(1, 1, 12, 13).numpy()
    num_features = input.shape[1]
    running_mean = torch.randn(num_features).numpy()
    running_var = torch.rand(num_features).numpy()
    weight = torch.randn(num_features).numpy()
    bias = torch.randn(num_features).numpy()
    training = True
    momentum = 0.3
    eps = 1e-7
    input_dict = {
        "input": input,
        "running_mean": running_mean,
        "running_var": running_var,
        "weight": weight,
        "bias": bias,
        "training": training,
        "momentum": momentum,
        "eps": eps
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs
def lstm_cell_inputs():
    list_of_inputs = []
    # Input 1
    input_size = 10
    hidden_size = 20
    input = torch.randn(5, input_size).numpy()
    hx = torch.randn(5, hidden_size).numpy()
    cx = torch.randn(5, hidden_size).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size).numpy()
    bias_ih = torch.randn(4 * hidden_size).numpy()
    bias_hh = torch.randn(4 * hidden_size).numpy()
    bias = True
    input_dict = {
        "input": input,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 2
    input_size = 5
    hidden_size = 10
    input = torch.randn(2, input_size).numpy()
    hx = torch.randn(2, hidden_size).numpy()
    cx = torch.randn(2, hidden_size).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size).numpy()
    bias_ih = torch.randn(4 * hidden_size).numpy()
    bias_hh = torch.randn(4 * hidden_size).numpy()
    bias = True
    input_dict = {
        "input": input,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 3
    input_size = 7
    hidden_size = 12
    input = torch.randn(3, input_size).numpy()
    hx = torch.randn(3, hidden_size).numpy()
    cx = torch.randn(3, hidden_size).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size).numpy()
    bias_ih = torch.randn(4 * hidden_size).numpy()
    bias_hh = torch.randn(4 * hidden_size).numpy()
    bias = True
    input_dict = {
        "input": input,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 4
    input_size = 4
    hidden_size = 8
    input = torch.randn(1, input_size).numpy()
    hx = torch.randn(1, hidden_size).numpy()
    cx = torch.randn(1, hidden_size).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size).numpy()
    bias_ih = torch.randn(4 * hidden_size).numpy()
    bias_hh = torch.randn(4 * hidden_size).numpy()
    bias = True
    input_dict = {
        "input": input,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    # Input 5
    input_size = 6
    hidden_size = 9
    input = torch.randn(4, input_size).numpy()
    hx = torch.randn(4, hidden_size).numpy()
    cx = torch.randn(4, hidden_size).numpy()
    weight_ih = torch.randn(4 * hidden_size, input_size).numpy()
    weight_hh = torch.randn(4 * hidden_size, hidden_size).numpy()
    bias_ih = torch.randn(4 * hidden_size).numpy()
    bias_hh = torch.randn(4 * hidden_size).numpy()
    bias = True
    input_dict = {
        "input": input,
        "hx": hx,
        "cx": cx,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": bias,
        "weight_ih": weight_ih,
        "weight_hh": weight_hh,
        "bias_ih": bias_ih,
        "bias_hh": bias_hh
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs
def margin_ranking_loss_inputs():
    list_of_inputs = []
    input1 = torch.randn(5).numpy()
    input2 = torch.randn(5).numpy()
    target = torch.randint(0, 2, (5,)).float().numpy() * 2 - 1
    margin = 0.0
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
    input1 = torch.randn(10).numpy()
    input2 = torch.randn(10).numpy()
    target = torch.randint(0, 2, (10,)).float().numpy() * 2 - 1
    margin = 1.0
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
    input1 = torch.randn(3, 4).numpy()
    input2 = torch.randn(3, 4).numpy()
    target = torch.randint(0, 2, (3, 4)).float().numpy() * 2 - 1
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
    input1 = torch.randn(2, 2, 2).numpy()
    input2 = torch.randn(2, 2, 2).numpy()
    target = torch.randint(0, 2, (2, 2, 2)).float().numpy() * 2 - 1
    margin = 0.2
    size_average = False
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
    input1 = torch.randn(7).numpy()
    input2 = torch.randn(7).numpy()
    target = torch.randint(0, 2, (7,)).float().numpy() * 2 - 1
    margin = 1.5
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
    return list_of_inputs
