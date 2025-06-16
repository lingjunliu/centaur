import torch, copy
import numpy as np

generated_inputs = {}

def flatten_inputs():
    list_of_inputs = []

    input1 = torch.randn(32, 1, 5, 5).numpy()
    input_dict1 = {
        "input": input1,
        "start_dim": 1,
        "end_dim": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(32, 1, 5, 5).numpy()
    input_dict2 = {
        "input": input2,
        "start_dim": 0,
        "end_dim": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(2, 3, 4, 5, 6).numpy()
    input_dict3 = {
        "input": input3,
        "start_dim": 1,
        "end_dim": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(2, 3, 4).numpy()
    input_dict4 = {
        "input": input4,
        "start_dim": 0,
        "end_dim": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 2, 3, 4).numpy()
    input_dict5 = {
        "input": input5,
        "start_dim": 2,
        "end_dim": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randint(0, 10, (2, 3, 4, 5)).numpy()
    input_dict6 = {
        "input": input6,
        "start_dim": 1,
        "end_dim": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(2, 3, 4, 5).double().numpy()
    input_dict7 = {
        "input": input7,
        "start_dim": 1,
        "end_dim": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs


generated_inputs["torch.nn.Flatten"] = flatten_inputs()

import torch, copy
import numpy as np

def FractionalMaxPool2d_inputs():
    list_of_inputs = []

    # Example 1: Using output_size
    input = torch.randn(20, 16, 50, 32).numpy()
    kernel_size = 3
    output_size = (13, 12)
    output_ratio = None
    return_indices = False
    input_dict = {
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Using output_ratio
    input = torch.randn(20, 16, 50, 32).numpy()
    kernel_size = 3
    output_size = None
    output_ratio = (0.5, 0.5)
    return_indices = False
    input_dict = {
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Square kernel and output size
    input = torch.randn(1, 1, 20, 20).numpy()
    kernel_size = 2
    output_size = 10
    output_ratio = None
    return_indices = True
    input_dict = {
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Different input size and kernel size
    input = torch.randn(4, 3, 28, 28).numpy()
    kernel_size = (2, 2)
    output_size = (14, 14)
    output_ratio = None
    return_indices = False
    input_dict = {
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: Single channel input
    input = torch.randn(1, 1, 32, 32).numpy()
    kernel_size = 4
    output_size = (16, 16)
    output_ratio = None
    return_indices = False
    input_dict = {
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 6: different batch size
    input = torch.randn(8, 3, 64, 64).numpy()
    kernel_size = (3,3)
    output_size = None
    output_ratio = (0.25, 0.25)
    return_indices = True

    input_dict = {
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs


generated_inputs["torch.nn.FractionalMaxPool2d_1"] = FractionalMaxPool2d_inputs()

import torch, copy
import numpy as np

def FractionalMaxPool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(20, 16, 50, 32).numpy()
    kernel_size1 = (3, 2)
    output_size1 = (13, 12)
    output_ratio1 = None
    return_indices1 = False
    input_dict1 = {
        "kernel_size": kernel_size1,
        "output_size": output_size1,
        "output_ratio": output_ratio1,
        "return_indices": return_indices1,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(20, 16, 50, 32).numpy()
    kernel_size2 = 3
    output_size2 = None
    output_ratio2 = (0.5, 0.5)
    return_indices2 = True
    input_dict2 = {
        "kernel_size": kernel_size2,
        "output_size": output_size2,
        "output_ratio": output_ratio2,
        "return_indices": return_indices2,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 1, 20, 20).numpy()
    kernel_size3 = 5
    output_size3 = (10, 10)
    output_ratio3 = None
    return_indices3 = False
    input_dict3 = {
        "kernel_size": kernel_size3,
        "output_size": output_size3,
        "output_ratio": output_ratio3,
        "return_indices": return_indices3,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1, 20, 20).numpy()
    kernel_size4 = (5, 5)
    output_size4 = None
    output_ratio4 = (0.25, 0.25)
    return_indices4 = True
    input_dict4 = {
        "kernel_size": kernel_size4,
        "output_size": output_size4,
        "output_ratio": output_ratio4,
        "return_indices": return_indices4,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(5, 3, 40, 40).numpy()
    kernel_size5 = 2
    output_size5 = (30, 20)
    output_ratio5 = None
    return_indices5 = False
    input_dict5 = {
        "kernel_size": kernel_size5,
        "output_size": output_size5,
        "output_ratio": output_ratio5,
        "return_indices": return_indices5,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(5, 3, 40, 40).numpy()
    kernel_size6 = (4, 3)
    output_size6 = None
    output_ratio6 = (0.7, 0.6)
    return_indices6 = True
    input_dict6 = {
        "kernel_size": kernel_size6,
        "output_size": output_size6,
        "output_ratio": output_ratio6,
        "return_indices": return_indices6,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs


generated_inputs["torch.nn.FractionalMaxPool2d_2"] = FractionalMaxPool2d_inputs()

import torch, copy
import numpy as np

def hardshrink_inputs():
    list_of_inputs = []

    input_1 = torch.randn(2).numpy()
    lambd_1 = 0.5
    input_dict_1 = {"lambd": lambd_1, "input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = torch.randn(2, 3).numpy()
    lambd_2 = 0.7
    input_dict_2 = {"lambd": lambd_2, "input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = torch.randn(2, 3, 4).numpy()
    lambd_3 = 0.3
    input_dict_3 = {"lambd": lambd_3, "input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_4 = torch.randn(1, 5, 5, 5).numpy()
    lambd_4 = 0.9
    input_dict_4 = {"lambd": lambd_4, "input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_5 = torch.randn(size=(10,)).numpy()
    lambd_5 = 0.1
    input_dict_5 = {"lambd": lambd_5, "input": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    input_6 = torch.randn(3, 3).float().numpy()
    lambd_6 = 1.2
    input_dict_6 = {"lambd": lambd_6, "input": input_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    return list_of_inputs


generated_inputs["torch.nn.Hardshrink"] = hardshrink_inputs()

import torch
import copy
import numpy as np

def Hardswish_inputs():
    list_of_inputs = []

    input1 = np.random.randn(2).astype(np.float32)
    input_dict1 = {
        "inplace": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randn(2, 3).astype(np.float64)
    input_dict2 = {
        "inplace": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.randn(2, 3, 4).astype(np.float16)
    input_dict3 = {
        "inplace": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.randn(1, 1, 1, 1).astype(np.float32)
    input_dict4 = {
        "inplace": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([-4, -2, 0, 2, 4]).astype(np.float32)
    input_dict5 = {
        "inplace": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.randn(5).astype(np.float32)
    input_dict6 = {
        "inplace": True,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.randn(1, 5, 5, 5).astype(np.float32)
    input_dict7 = {
        "inplace": False,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs


generated_inputs["torch.nn.Hardswish"] = Hardswish_inputs()

import torch, copy
import numpy as np

def Hardtanh_inputs():
    list_of_inputs = []

    input1 = torch.randn(2).numpy()
    input_dict1 = {
        "min_val": -2.0,
        "max_val": 2.0,
        "inplace": False,
        "min_value": None,
        "max_value": None,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3).numpy()
    input_dict2 = {
        "min_val": -0.5,
        "max_val": 1.5,
        "inplace": True,
        "min_value": None,
        "max_value": None,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 2, 3, 4).numpy()
    input_dict3 = {
        "min_val": -1.0,
        "max_val": 1.0,
        "inplace": False,
        "min_value": None,
        "max_value": None,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(5).numpy()
    input_dict4 = {
        "min_val": -5.0,
        "max_val": 0.0,
        "inplace": True,
        "min_value": None,
        "max_value": None,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 2, 2).numpy()
    input_dict5 = {
        "min_val": 0.0,
        "max_val": 5.0,
        "inplace": False,
        "min_value": None,
        "max_value": None,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randint(-10, 10, (3,4)).float().numpy()
    input_dict6 = {
        "min_val": -3.0,
        "max_val": 7.0,
        "inplace": False,
        "min_value": None,
        "max_value": None,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs


generated_inputs["torch.nn.Hardtanh"] = Hardtanh_inputs()

import torch
import torch.nn as nn
import numpy as np
import copy

def GroupNorm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32 and affine=True
    input1 = torch.randn(20, 6, 10, 10).numpy()
    num_groups1 = 3
    num_channels1 = 6
    input_dict1 = {
        "num_groups": num_groups1,
        "num_channels": num_channels1,
        "eps": 1e-05,
        "affine": True,
        "input": input1,
        "dtype": torch.float32  # Use torch.dtype here
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different number of groups, equivalent to InstanceNorm
    input2 = torch.randn(5, 8, 7, 7).numpy()
    num_groups2 = 8
    num_channels2 = 8
    input_dict2 = {
        "num_groups": num_groups2,
        "num_channels": num_channels2,
        "eps": 1e-08,
        "affine": False,
        "input": input2,
        "dtype": torch.float32 # Use torch.dtype here
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Equivalent to LayerNorm
    input3 = torch.randn(10, 4, 5, 5).numpy()
    num_groups3 = 1
    num_channels3 = 4
    input_dict3 = {
        "num_groups": num_groups3,
        "num_channels": num_channels3,
        "eps": 1e-04,
        "affine": True,
        "input": input3,
        "dtype": torch.float32 # Use torch.dtype here
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 7: Different epsilon
    input7 = torch.randn(1, 32, 8, 8).numpy()
    num_groups7 = 4
    num_channels7 = 32
    input_dict7 = {
        "num_groups": num_groups7,
        "num_channels": num_channels7,
        "eps": 1e-02,
        "affine": True,
        "input": input7,
        "dtype": torch.float32 # Use torch.dtype here
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs


generated_inputs["torch.nn.GroupNorm"] = GroupNorm_inputs()

import torch, copy
import numpy as np

def InstanceNorm1d_inputs():
    list_of_inputs = []

    num_features = 100
    input1 = torch.randn(20, num_features, 40).numpy()
    input_dict1 = {
        "num_features": num_features,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": False,
        "dtype": None,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    num_features = 50
    input2 = torch.randn(5, num_features, 20).numpy()
    input_dict2 = {
        "num_features": num_features,
        "eps": 1e-04,
        "momentum": 0.2,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    num_features = 25
    input3 = torch.randn(10, num_features, 10).numpy()
    input_dict3 = {
        "num_features": num_features,
        "eps": 1e-06,
        "momentum": 0.05,
        "affine": False,
        "track_running_stats": True,
        "dtype": None,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    num_features = 64
    input4 = torch.randn(1, num_features, 32).numpy()
    input_dict4 = {
        "num_features": num_features,
        "eps": 1e-07,
        "momentum": 0.9,
        "affine": True,
        "track_running_stats": False,
        "dtype": None,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    num_features = 128
    input5 = torch.randn(32, num_features, 64).numpy()
    input_dict5 = {
        "num_features": num_features,
        "eps": 1e-03,
        "momentum": 0.3,
        "affine": False,
        "track_running_stats": True,
        "dtype": None,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs


generated_inputs["torch.nn.InstanceNorm1d"] = InstanceNorm1d_inputs()

import torch
import numpy as np
import copy

def InstanceNorm2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with N, C, H, W
    input1 = np.random.randn(2, 3, 32, 32).astype(np.float32)
    input_dict1 = {
        "num_features": 3,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: With affine=True
    input2 = np.random.randn(1, 5, 16, 16).astype(np.float32)
    input_dict2 = {
        "num_features": 5,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": False,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: With track_running_stats=True
    input3 = np.random.randn(4, 7, 64, 64).astype(np.float32)
    input_dict3 = {
        "num_features": 7,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": True,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different momentum and eps
    input4 = np.random.randn(1, 10, 8, 8).astype(np.float32)
    input_dict4 = {
        "num_features": 10,
        "eps": 1e-3,
        "momentum": 0.5,
        "affine": False,
        "track_running_stats": False,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Input with C, H, W
    input5 = np.random.randn(12, 24, 24).astype(np.float32)
    input_dict5 = {
        "num_features": 12,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs


generated_inputs["torch.nn.InstanceNorm2d"] = InstanceNorm2d_inputs()

import torch
import copy
import numpy as np

def InstanceNorm3d_inputs():
    list_of_inputs = []

    input1 = torch.randn(20, 100, 35, 45, 10).numpy()
    input_dict1 = {
        "num_features": 100,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": False,
        "dtype": torch.float32,
        "input": input1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3, 4, 5, 6).numpy()
    input_dict2 = {
        "num_features": 3,
        "eps": 1e-04,
        "momentum": 0.2,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": input2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 64, 16, 16, 16).numpy()
    input_dict3 = {
        "num_features": 64,
        "eps": 1e-08,
        "momentum": 0.05,
        "affine": False,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": input3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(5, 128, 8, 8, 8).numpy()
    input_dict4 = {
        "num_features": 128,
        "eps": 1e-03,
        "momentum": 0.3,
        "affine": True,
        "track_running_stats": False,
        "dtype": torch.float32,
        "input": input4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(10, 32, 24, 24, 24).numpy()
    input_dict5 = {
        "num_features": 32,
        "eps": 1e-06,
        "momentum": 0.15,
        "affine": False,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": input5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs


generated_inputs["torch.nn.InstanceNorm3d"] = InstanceNorm3d_inputs()

import torch
import numpy as np
import copy

def L1Loss_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 5).numpy()
    target1 = torch.randn(3, 5).numpy()
    input_dict1 = {
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 4, 6).numpy()
    target2 = torch.randn(2, 4, 6).numpy()
    input_dict2 = {
        "size_average": False,
        "reduce": True,
        "reduction": 'sum',
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 7).numpy()
    target3 = torch.randn(1, 7).numpy()
    input_dict3 = {
        "size_average": None,
        "reduce": False,
        "reduction": 'none',
        "input": input3,
        "target": target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(4, 4).numpy()
    target4 = torch.randn(4, 4).numpy()
    input_dict4 = {
        "size_average": True,
        "reduce": None,
        "reduction": 'mean',
        "input": input4,
        "target": target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 3, 4, 5).numpy()
    target5 = torch.randn(2, 3, 4, 5).numpy()
    input_dict5 = {
        "size_average": False,
        "reduce": False,
        "reduction": 'none',
        "input": input5,
        "target": target5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

if __name__ == '__main__':
    
    generated_inputs["torch.nn.L1Loss"] = L1Loss_inputs()
    #print(generated_inputs)

import torch
import copy
import numpy as np

def LPPool1d_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    input_dict = {
        "norm_type": 2,
        "kernel_size": 3,
        "stride": 2,
        "ceil_mode": False,
        "input": torch.randn(20, 16, 50).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different norm_type, no stride
    input_dict = {
        "norm_type": 1,
        "kernel_size": 5,
        "stride": None,
        "ceil_mode": True,
        "input": torch.randn(5, 8, 100).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different input shape (C, Lin)
    input_dict = {
        "norm_type": 3,
        "kernel_size": 4,
        "stride": 3,
        "ceil_mode": False,
        "input": torch.randn(16, 50).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different input with negative values
    input_dict = {
        "norm_type": 2,
        "kernel_size": 3,
        "stride": 2,
        "ceil_mode": True,
        "input": torch.randn(2, 4, 20).numpy() * -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: kernel_size = 1
    input_dict = {
        "norm_type": 2,
        "kernel_size": 1,
        "stride": 1,
        "ceil_mode": False,
        "input": torch.randn(10, 5, 30).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: stride greater than kernel size
    input_dict = {
        "norm_type": 2,
        "kernel_size": 2,
        "stride": 3,
        "ceil_mode": False,
        "input": torch.randn(5, 3, 15).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs


generated_inputs["torch.nn.LPPool1d"] = LPPool1d_inputs()

import torch, copy
import numpy as np

def LPPool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3, 32, 32).numpy()
    norm_type1 = 2.0
    kernel_size1 = 3
    stride1 = 2
    ceil_mode1 = False
    input_dict1 = {
        "norm_type": norm_type1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "ceil_mode": ceil_mode1,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 16, 64, 64).numpy()
    norm_type2 = 1.5
    kernel_size2 = (5, 5)
    stride2 = (3, 3)
    ceil_mode2 = True
    input_dict2 = {
        "norm_type": norm_type2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "ceil_mode": ceil_mode2,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(4, 8, 16, 16).numpy()
    norm_type3 = 3.0
    kernel_size3 = 2
    stride3 = None
    ceil_mode3 = False
    input_dict3 = {
        "norm_type": norm_type3,
        "kernel_size": kernel_size3,
        "stride": stride3,
        "ceil_mode": ceil_mode3,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(3, 32, 32).numpy()
    norm_type4 = 1.0
    kernel_size4 = (3, 2)
    stride4 = (2, 1)
    ceil_mode4 = True
    input_dict4 = {
        "norm_type": norm_type4,
        "kernel_size": kernel_size4,
        "stride": stride4,
        "ceil_mode": ceil_mode4,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 4, 28, 28).numpy()
    norm_type5 = float('inf')
    kernel_size5 = 7
    stride5 = 1
    ceil_mode5 = False
    input_dict5 = {
        "norm_type": norm_type5,
        "kernel_size": kernel_size5,
        "stride": stride5,
        "ceil_mode": ceil_mode5,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(1, 1, 10, 10).numpy()
    norm_type6 = 0.5
    kernel_size6 = 3
    stride6 = 1
    ceil_mode6 = True
    input_dict6 = {
        "norm_type": norm_type6,
        "kernel_size": kernel_size6,
        "stride": stride6,
        "ceil_mode": ceil_mode6,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs


generated_inputs["torch.nn.LPPool2d_1"] = LPPool2d_inputs()

import torch, copy
import numpy as np

def LPPool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(20, 16, 50, 32).numpy()
    input_dict1 = {
        "norm_type": 2.0,
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "ceil_mode": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(20, 16, 50, 32).numpy()
    input_dict2 = {
        "norm_type": 1.2,
        "kernel_size": (3, 2),
        "stride": (2, 1),
        "ceil_mode": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(5, 3, 25, 25).numpy()
    input_dict3 = {
        "norm_type": 1.0,
        "kernel_size": (5, 5),
        "stride": (5, 5),
        "ceil_mode": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1, 100, 100).numpy()
    input_dict4 = {
        "norm_type": 2.0,
        "kernel_size": (10, 10),
        "stride": (10, 10),
        "ceil_mode": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 1, 50, 50).numpy()
    input_dict5 = {
        "norm_type": 0.5,
        "kernel_size": (7, 7),
        "stride": (1, 1),
        "ceil_mode": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs


generated_inputs["torch.nn.LPPool2d_2"] = LPPool2d_inputs()

import torch, copy
import numpy as np

def L1Loss_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 5).numpy()
    target1 = torch.randn(3, 5).numpy()
    input_dict1 = {
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 4, 6).numpy()
    target2 = torch.randn(2, 4, 6).numpy()
    input_dict2 = {
        "size_average": False,
        "reduce": True,
        "reduction": 'sum',
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 3, 5, 7).numpy()
    target3 = torch.randn(1, 3, 5, 7).numpy()
    input_dict3 = {
        "size_average": None,
        "reduce": None,
        "reduction": 'none',
        "input": input3,
        "target": target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randint(-10, 10, (4, 4), dtype=torch.int32).float().numpy()
    target4 = torch.randint(-10, 10, (4, 4), dtype=torch.int32).float().numpy()
    input_dict4 = {
        "size_average": True,
        "reduce": False,
        "reduction": 'mean',
        "input": input4,
        "target": target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(2, 3, dtype=torch.complex64).numpy()
    target5 = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict5 = {
        "size_average": None,
        "reduce": False,
        "reduction": 'sum',
        "input": input5,
        "target": target5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

if __name__ == '__main__':
    
    generated_inputs["torch.nn.L1Loss"] = L1Loss_inputs()
    #print(generated_inputs)

import torch, copy
import numpy as np

def layernorm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with integer normalized_shape
    input_1 = torch.randn(20, 5, 10).numpy()
    input_dict_1 = {
        "normalized_shape": 10,
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: normalized_shape as a list
    input_2 = torch.randn(20, 5, 10, 10).numpy()
    input_dict_2 = {
        "normalized_shape": [10, 10],
        "eps": 1e-05,
        "elementwise_affine": False,
        "bias": False,
        "dtype": None,
        "input": input_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Different eps value
    input_3 = torch.randn(20, 5, 10).numpy()
    input_dict_3 = {
        "normalized_shape": 10,
        "eps": 1e-08,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Float32 dtype
    input_4 = torch.randn(20, 5, 10).numpy()
    input_dict_4 = {
        "normalized_shape": 10,
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": torch.float32,
        "input": input_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Different input shape and normalized shape
    input_5 = torch.randn(20, 5, 7, 13).numpy()
    input_dict_5 = {
        "normalized_shape": [7, 13],
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs


generated_inputs["torch.nn.LayerNorm_1"] = layernorm_inputs()

import torch
import numpy as np
import copy

def torch_nn_LayerNorm_inputs():
    list_of_inputs = []

    # Case 1: Basic case with int normalized_shape
    input_1 = np.random.randn(20, 5, 10).astype(np.float32)
    input_dict_1 = {
        "normalized_shape": [10],
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: normalized_shape as a list
    input_2 = np.random.randn(20, 5, 10, 10).astype(np.float32)
    input_dict_2 = {
        "normalized_shape": [10, 10],
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Float input with elementwise_affine=False, bias=False
    input_3 = np.random.randn(20, 5, 10).astype(np.float64)
    input_dict_3 = {
        "normalized_shape": [10],
        "eps": 1e-05,
        "elementwise_affine": False,
        "bias": False,
        "dtype": torch.float64,
        "input": input_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Different eps value
    input_4 = np.random.randn(20, 5, 10).astype(np.float32)
    input_dict_4 = {
        "normalized_shape": [10],
        "eps": 1e-03,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Case 5: Input with different dimensions and normalized_shape
    input_5 = np.random.randn(2, 3, 4, 5, 6).astype(np.float32)
    input_dict_5 = {
        "normalized_shape": [6],
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Small input values and different dtype
    input_6 = (np.random.rand(2, 3, 4) * 0.01).astype(np.float16)
    input_dict_6 = {
        "normalized_shape": [4],
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": torch.float16,
        "input": input_6
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Case 7: Single dimension input, adjusted normalized_shape
    input_7 = np.random.randn(10).astype(np.float32)
    input_dict_7 = {
        "normalized_shape": [10],
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input_7
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs


generated_inputs["torch.nn.LayerNorm_2"] = torch_nn_LayerNorm_inputs()

import torch
import numpy as np
import copy

def LayerNorm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with a single integer normalized_shape
    input1 = torch.randn(20, 10).numpy()
    input_dict1 = {
        "normalized_shape": (10,),
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2:  normalized_shape as a tuple
    input2 = torch.randn(2, 3, 5, 5).numpy()
    input_dict2 = {
        "normalized_shape": (5, 5),
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3:  elementwise_affine=False and bias=False
    input3 = torch.randn(5, 4, 6).numpy()
    input_dict3 = {
        "normalized_shape": (6,),
        "eps": 1e-05,
        "elementwise_affine": False,
        "bias": False,
        "dtype": None,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4:  different eps value
    input4 = torch.randn(10, 20, 30).numpy()
    input_dict4 = {
        "normalized_shape": (30,),
        "eps": 1e-03,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5:  normalized_shape as a list
    input5 = torch.randn(4, 5, 7, 7).numpy()
    input_dict5 = {
        "normalized_shape": (7, 7),
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs


generated_inputs["torch.nn.LayerNorm_3"] = LayerNorm_inputs()

import torch, copy
import numpy as np

def L1Loss_inputs():
    list_of_inputs = []

    input_1 = torch.randn(3, 5, requires_grad=True).detach().numpy()
    target_1 = torch.randn(3, 5).numpy()
    input_dict_1 = {
        "size_average": True,
        "reduce": True,
        "reduction": 'mean',
        "input": input_1,
        "target": target_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = torch.randint(-10, 10, (2, 4), dtype=torch.int32).numpy()
    target_2 = torch.randint(-5, 5, (2, 4), dtype=torch.int32).numpy()
    input_dict_2 = {
        "size_average": False,
        "reduce": False,
        "reduction": 'none',
        "input": input_2,
        "target": target_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = torch.randn(1, 2, 3, 4).numpy()
    target_3 = torch.randn(1, 2, 3, 4).numpy()
    input_dict_3 = {
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "input": input_3,
        "target": target_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_4 = torch.randn(size=(4,)).numpy()
    target_4 = torch.randn(size=(4,)).numpy()
    input_dict_4 = {
        "size_average": True,
        "reduce": False,
        "reduction": 'none',
        "input": input_4,
        "target": target_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    input_5 = (torch.randn(2, 3) + 1j * torch.randn(2, 3)).numpy()
    target_5 = (torch.randn(2, 3) + 1j * torch.randn(2, 3)).numpy()

    input_dict_5 = {
        "size_average": False,
        "reduce": True,
        "reduction": 'mean',
        "input": input_5,
        "target": target_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs


generated_inputs["torch.nn.L1Loss"] = L1Loss_inputs()

import torch, copy
import numpy as np

def LeakyReLU_inputs():
    list_of_inputs = []

    input1 = torch.randn(2).numpy()
    input_dict1 = {
        "negative_slope": 0.1,
        "inplace": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3).numpy()
    input_dict2 = {
        "negative_slope": 0.01,
        "inplace": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randint(-5, 5, (2, 3, 4)).float().numpy()
    input_dict3 = {
        "negative_slope": 0.2,
        "inplace": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1, 5, 5).numpy()
    input_dict4 = {
        "negative_slope": 0.0,
        "inplace": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 8, 10, 10).numpy()
    input_dict5 = {
        "negative_slope": 0.5,
        "inplace": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(size=(3,4,5), dtype=torch.float64).numpy()
    input_dict6 = {
        "negative_slope": 0.3,
        "inplace": True,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randint(-10, 10, (4,4), dtype=torch.int32).float().numpy()
    input_dict7 = {
        "negative_slope": 0.05,
        "inplace": False,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.nn.LeakyReLU"] = LeakyReLU_inputs()

import torch
import copy
import numpy as np

def Linear_inputs():
    list_of_inputs = []

    # Test case 1: Basic case with bias=True and float32
    input = torch.randn(128, 20).numpy()
    input_dict = {
        "in_features": 20,
        "out_features": 30,
        "bias": True,
        "dtype": torch.float32,
        "input": input,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: No bias and float32
    input = torch.randn(64, 15).numpy()
    input_dict = {
        "in_features": 15,
        "out_features": 25,
        "bias": False,
        "dtype": torch.float32,
        "input": input,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: int64 input, but cast to float32
    input = torch.randint(0, 10, (32, 10)).float().numpy()
    input_dict = {
        "in_features": 10,
        "out_features": 5,
        "bias": True,
        "dtype": torch.float32,
        "input": input,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Different input dimensions (3D)
    input = torch.randn(10, 5, 8).numpy()
    input_dict = {
        "in_features": 8,
        "out_features": 12,
        "bias": True,
        "dtype": torch.float32,
        "input": input,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Another basic case
    input = torch.randn(256, 40).numpy()
    input_dict = {
        "in_features": 40,
        "out_features": 60,
        "bias": False,
        "dtype": torch.float32,
        "input": input,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.Linear"] = Linear_inputs()

import torch, copy
import numpy as np

def LogSigmoid_inputs():
    list_of_inputs = []

    input1 = np.random.randn(2).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randn(2, 3).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.randn(2, 3, 4).astype(np.float16)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.randn(1, 1, 1, 1).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([-1, 0, 1]).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.randn(5).astype(np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs["torch.nn.LogSigmoid"] = LogSigmoid_inputs()

import torch, copy
import numpy as np

def MSELoss_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(3, 5, requires_grad=True).detach().numpy()
    target_tensor = torch.randn(3, 5).numpy()

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 4, requires_grad=True).detach().numpy()
    target_tensor = torch.randn(2, 4).numpy()

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": False,
        "reduce": True,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 1, 10, 10, requires_grad=True).detach().numpy()
    target_tensor = torch.randn(1, 1, 10, 10).numpy()

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": None,
        "reduce": None,
        "reduction": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(4, 2, requires_grad=True).detach().numpy()
    target_tensor = torch.randn(4, 2).numpy()
    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": True,
        "reduce": False,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 3, 4, 5, requires_grad=True).detach().numpy()
    target_tensor = torch.randn(2, 3, 4, 5).numpy()

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": False,
        "reduce": False,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(size=(10,), requires_grad=True).detach().numpy()
    target_tensor = torch.randn(size=(10,)).numpy()

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(size=(2,2,2), requires_grad=True).detach().numpy()
    target_tensor = torch.randn(size=(2,2,2)).numpy()

    input_dict = {
        "input": input_tensor,
        "target": target_tensor,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MSELoss"] = MSELoss_inputs()

import torch
import numpy as np
import copy

def MarginRankingLoss_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0])
    input2 = np.array([4.0, 5.0, 6.0])
    target = np.array([1, -1, 1])
    input_dict = {
        "margin": 0.0,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([-1.0, -2.0, -3.0])
    input2 = np.array([-4.0, -5.0, -6.0])
    target = np.array([-1, 1, -1])
    input_dict = {
        "margin": 0.5,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([0.5, 1.5])
    input2 = np.array([1.0, 2.0])
    target = np.array([1, -1])
    input_dict = {
        "margin": 1.0,
        "size_average": None,
        "reduce": None,
        "reduction": 'none',
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([2.0])
    input2 = np.array([1.0])
    target = np.array([1])
    input_dict = {
        "margin": 0.2,
        "size_average": True,
        "reduce": False,
        "reduction": 'mean',
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([1.5, 2.5, 3.5, 4.5])
    input2 = np.array([2.0, 3.0, 4.0, 5.0])
    target = np.array([1, 1, -1, -1])
    input_dict = {
        "margin": 1.5,
        "size_average": False,
        "reduce": True,
        "reduction": 'mean',
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([0.0])
    input2 = np.array([0.0])
    target = np.array([1])
    input_dict = {
        "margin": 2.0,
        "size_average": False,
        "reduce": False,
        "reduction": 'mean',
        "input1": input1,
        "input2": input2,
        "target": target
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MarginRankingLoss"] = MarginRankingLoss_inputs()

import torch, copy
import numpy as np

def MaxPool2d_inputs():
    list_of_inputs = []

    input = torch.randn(20, 16, 50, 32).numpy()
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 1
    return_indices = False
    ceil_mode = False

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(20, 16, 50, 32).numpy()
    kernel_size = (3, 2)
    stride = (2, 1)
    padding = (1, 0)
    dilation = (1, 2)
    return_indices = True
    ceil_mode = True

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(5, 3, 256, 256).numpy()
    kernel_size = 5
    stride = None
    padding = 2
    dilation = 1
    return_indices = False
    ceil_mode = False

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": kernel_size if stride is None else stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 128, 128).numpy()
    kernel_size = (7, 7)
    stride = (4, 4)
    padding = (3, 3)
    dilation = (1, 1)
    return_indices = False
    ceil_mode = False

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 3, 64, 64).numpy()
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    return_indices = False
    ceil_mode = False

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }

    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs


generated_inputs["torch.nn.MaxPool2d_1"] = MaxPool2d_inputs()

import torch
import numpy as np
import copy

def torch_nn_MaxPool2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        'kernel_size': (3, 2),
        'stride': (2, 1),
        'padding': (0, 0),
        'dilation': (1, 1),
        'return_indices': False,
        'ceil_mode': False,
        'input': input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(1, 3, 256, 256).numpy()
    input_dict = {
        'kernel_size': (2, 2),
        'stride': (2, 2),
        'padding': (0, 0),
        'dilation': (1, 1),
        'return_indices': True,
        'ceil_mode': True,
        'input': input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(5, 1, 64, 64).numpy()
    input_dict = {
        'kernel_size': (3, 3),
        'stride': (1, 1),
        'padding': (1, 1),
        'dilation': (1, 1),
        'return_indices': False,
        'ceil_mode': False,
        'input': input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_tensor = torch.randn(1, 1, 128, 128).numpy()
    input_dict = {
        'kernel_size': (4, 4),
        'stride': (4, 4),
        'padding': (0, 0),
        'dilation': (1, 1),
        'return_indices': True,
        'ceil_mode': False,
        'input': input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(3, 3, 32, 32).numpy()
    input_dict = {
        'kernel_size': (2, 2),
        'stride': (1, 1),
        'padding': (1, 1),
        'dilation': (1, 1),
        'return_indices': False,
        'ceil_mode': True,
        'input': input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["torch.nn.MaxPool2d_2"] = torch_nn_MaxPool2d_inputs()

import torch, copy
import numpy as np

def maxpool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(20, 16, 50, 32).numpy()
    input_dict1 = {
        "kernel_size": 3,
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3, 20, 20).numpy()
    input_dict2 = {
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 1, 10, 10).numpy()
    input_dict3 = {
        "kernel_size": 3,
        "stride": None,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1, 10, 10).numpy()
    input_dict4 = {
        "kernel_size": (3, 2),
        "stride": (2, 1),
        "padding": (1, 0),
        "dilation": (2, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(4, 5, 28, 28).numpy()
    input_dict5 = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(1, 3, 32, 32).numpy()
    input_dict6 = {
        "kernel_size": (4, 4),
        "stride": (2, 2),
        "padding": (1, 1),
        "dilation": (1, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_3"] = maxpool2d_inputs()

import torch, copy
import numpy as np

def MaxPool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(20, 16, 50, 32).numpy()
    kernel_size1 = 3
    stride1 = 2
    padding1 = (0, 0)
    dilation1 = (1, 1)
    return_indices1 = False
    ceil_mode1 = False
    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "dilation": dilation1,
        "return_indices": return_indices1,
        "ceil_mode": ceil_mode1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(5, 3, 25, 25).numpy()
    kernel_size2 = (5, 5)
    stride2 = (2, 2)
    padding2 = (1, 1)
    dilation2 = (1, 1)
    return_indices2 = True
    ceil_mode2 = True
    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "dilation": dilation2,
        "return_indices": return_indices2,
        "ceil_mode": ceil_mode2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 1, 100, 100).numpy()
    kernel_size3 = 7
    stride3 = 7
    padding3 = (3, 3)
    dilation3 = (1, 1)
    return_indices3 = False
    ceil_mode3 = True
    input_dict3 = {
        "input": input3,
        "kernel_size": kernel_size3,
        "stride": stride3,
        "padding": padding3,
        "dilation": dilation3,
        "return_indices": return_indices3,
        "ceil_mode": ceil_mode3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(4, 3, 32, 32).numpy()
    kernel_size4 = (2, 4)
    stride4 = (1, 2)
    padding4 = (0, 1)
    dilation4 = (1, 1)
    return_indices4 = True
    ceil_mode4 = False
    input_dict4 = {
        "input": input4,
        "kernel_size": kernel_size4,
        "stride": stride4,
        "padding": padding4,
        "dilation": dilation4,
        "return_indices": return_indices4,
        "ceil_mode": ceil_mode4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 64, 64, 64).numpy()
    kernel_size5 = 2
    stride5 = 2
    padding5 = (0, 0)
    dilation5 = (1, 1)
    return_indices5 = False
    ceil_mode5 = False
    input_dict5 = {
        "input": input5,
        "kernel_size": kernel_size5,
        "stride": stride5,
        "padding": padding5,
        "dilation": dilation5,
        "return_indices": return_indices5,
        "ceil_mode": ceil_mode5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs


generated_inputs["torch.nn.MaxPool2d_4"] = MaxPool2d_inputs()

import torch, copy
import numpy as np

def torch_nn_MaxPool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(20, 16, 50, 32).numpy()
    input_dict1 = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(20, 16, 50, 32).numpy()
    input_dict2 = {
        "kernel_size": (3, 2),
        "stride": (2, 1),
        "padding": 0,
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 3, 256, 256).numpy()
    input_dict3 = {
        "kernel_size": 5,
        "stride": 3,
        "padding": 2,
        "dilation": (2, 2),
        "return_indices": True,
        "ceil_mode": True,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 3, 64, 64).numpy()
    input_dict4 = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (1, 1),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(4, 1, 128, 128).numpy()
    input_dict5 = {
        "kernel_size": 7,
        "stride": 4,
        "padding": 3,
        "dilation": (1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(1, 1, 20, 20).numpy()
    input_dict6 = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_5"] = torch_nn_MaxPool2d_inputs()

import torch, copy
import numpy as np

def maxpool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(20, 16, 50, 32).numpy()
    input_dict1 = {
        "kernel_size": (3, 2),
        "stride": (2, 1),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 3, 256, 256).numpy()
    input_dict2 = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 1, 64, 64).numpy()
    input_dict3 = {
        "kernel_size": 4,
        "stride": 4,
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(4, 3, 32, 32).numpy()
    input_dict4 = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (1, 1),
        "dilation": (2, 2),
        "return_indices": False,
        "ceil_mode": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 3, 128, 128).numpy()
    input_dict5 = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs


generated_inputs["torch.nn.MaxPool2d_6"] = maxpool2d_inputs()

import torch
import numpy as np
import copy

def MaxPool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(20, 16, 50, 32).numpy()
    kernel_size1 = (3, 2)
    stride1 = (2, 1)
    padding1 = 0
    dilation1 = 1
    return_indices1 = False
    ceil_mode1 = False

    input_dict1 = {
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "dilation": dilation1,
        "return_indices": return_indices1,
        "ceil_mode": ceil_mode1,
        "input": input1
    }

    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 3, 256, 256).numpy()
    kernel_size2 = 2
    stride2 = None
    padding2 = 1
    dilation2 = 1
    return_indices2 = False
    ceil_mode2 = False

    input_dict2 = {
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "dilation": dilation2,
        "return_indices": return_indices2,
        "ceil_mode": ceil_mode2,
        "input": input2
    }

    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(4, 1, 64, 64).numpy()
    kernel_size3 = 3
    stride3 = 2
    padding3 = 0
    dilation3 = 2
    return_indices3 = True
    ceil_mode3 = True

    input_dict3 = {
        "kernel_size": kernel_size3,
        "stride": stride3,
        "padding": padding3,
        "dilation": dilation3,
        "return_indices": return_indices3,
        "ceil_mode": ceil_mode3,
        "input": input3
    }

    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1, 10, 10).numpy()
    kernel_size4 = (5, 5)
    stride4 = (3, 3)
    padding4 = (2, 2)
    dilation4 = (1, 1)
    return_indices4 = False
    ceil_mode4 = False

    input_dict4 = {
        "kernel_size": kernel_size4,
        "stride": stride4,
        "padding": padding4,
        "dilation": dilation4,
        "return_indices": return_indices4,
        "ceil_mode": ceil_mode4,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(3, 3, 32, 32).numpy()
    kernel_size5 = (2, 2)
    stride5 = (1, 1)
    padding5 = (1, 1)
    dilation5 = (2, 2)
    return_indices5 = True
    ceil_mode5 = True

    input_dict5 = {
        "kernel_size": kernel_size5,
        "stride": stride5,
        "padding": padding5,
        "dilation": dilation5,
        "return_indices": return_indices5,
        "ceil_mode": ceil_mode5,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_7"] = MaxPool2d_inputs()

import torch, copy
import numpy as np

def torch_nn_MaxPool2d_inputs():
    list_of_inputs = []

    input = torch.randn(20, 16, 50, 32).numpy()
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 1
    return_indices = False
    ceil_mode = False

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 256, 256).numpy()
    kernel_size = (3, 2)
    stride = (2, 1)
    padding = (1, 1)
    dilation = (2, 2)
    return_indices = True
    ceil_mode = True

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 64, 64).numpy()
    kernel_size = 5
    stride = None
    padding = 2
    dilation = 1
    return_indices = False
    ceil_mode = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 5, 32, 32).numpy()
    kernel_size = (2, 2)
    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)
    return_indices = True
    ceil_mode = False

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 3, 128, 128).numpy()
    kernel_size = 7
    stride = 4
    padding = 3
    dilation = 2
    return_indices = False
    ceil_mode = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }

    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs


generated_inputs["torch.nn.MaxPool2d_8"] = torch_nn_MaxPool2d_inputs()

import torch
import copy
import numpy as np

def torch_nn_MaxPool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3, 32, 32).numpy()
    input_dict1 = {
        "kernel_size": (3, 3),
        "stride": 2,
        "padding": 1,
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 1, 16, 16).numpy()
    input_dict2 = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": input2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(4, 5, 64, 64).numpy()
    input_dict3 = {
        "kernel_size": (5, 5),
        "stride": 3,
        "padding": 2,
        "dilation": (2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 3, 28, 28).numpy()
    input_dict4 = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": input4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(3, 1, 48, 48).numpy()
    input_dict5 = {
        "kernel_size": (4, 4),
        "stride": 2,
        "padding": (1, 1),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_9"] = torch_nn_MaxPool2d_inputs()

import torch
import copy
import numpy as np

def torch_nn_MaxPool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(1, 3, 32, 32).numpy()
    input_dict1 = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": (1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 1, 28, 28).numpy()
    input_dict2 = {
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 3, 64, 64).numpy()
    input_dict3 = {
        "kernel_size": (4, 4),
        "stride": (4, 4),
        "padding": (0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1, 16, 16).numpy()
    input_dict4 = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": (1, 1),
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(4, 3, 128, 128).numpy()
    input_dict5 = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs


generated_inputs["torch.nn.MaxPool2d_10"] = torch_nn_MaxPool2d_inputs()

import torch, copy
import numpy as np

def MaxPool2d_inputs():
    list_of_inputs = []

    input = torch.randn(20, 16, 50, 32).numpy()
    kernel_size = (3, 2)
    stride = (2, 1)
    padding = (0, 0)
    dilation = (1, 1)
    return_indices = False
    ceil_mode = False
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(10, 8, 25, 16).numpy()
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = True
    ceil_mode = True
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(5, 4, 12, 8).numpy()
    kernel_size = (2, 2)
    stride = (1, 1)
    padding = (0, 0)
    dilation = (2, 2)
    return_indices = False
    ceil_mode = False
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 2, 60, 40).numpy()
    kernel_size = 5
    stride = None
    padding = 2
    dilation = 1
    return_indices = False
    ceil_mode = False
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(1, 1, 100, 100).numpy()
    kernel_size = (7, 5)
    stride = (3, 2)
    padding = (1, 1)
    dilation = (3, 2)
    return_indices = True
    ceil_mode = True
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs


generated_inputs["torch.nn.MaxPool2d_12"] = MaxPool2d_inputs()

import torch, copy
import numpy as np

def MaxPool2d_inputs():
    list_of_inputs = []

    input = torch.randn(20, 16, 50, 32).numpy()
    kernel_size = (3, 2)
    stride = (2, 1)
    padding = (0, 1)
    dilation = (1, 2)
    return_indices = True
    ceil_mode = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 3, 25, 25).numpy()
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = False
    ceil_mode = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(5, 1, 100, 100).numpy()
    kernel_size = (7, 7)
    stride = (4, 4)
    padding = (2, 2)
    dilation = (3, 3)
    return_indices = True
    ceil_mode = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 10, 10).numpy()
    kernel_size = 2
    stride = 1
    padding = 0
    dilation = 1
    return_indices = False
    ceil_mode = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 4, 20, 20).numpy()
    kernel_size = (4, 4)
    stride = (2, 2)
    padding = (1, 1)
    dilation = (1, 1)
    return_indices = True
    ceil_mode = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs


generated_inputs["torch.nn.MaxPool2d_13"] = MaxPool2d_inputs()

import torch, copy
import numpy as np

def MaxPool3d_inputs():
    list_of_inputs = []

    input = torch.randn(2, 3, 20, 20, 20).numpy()
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 1
    return_indices = False
    ceil_mode = False

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 10, 10, 10).numpy()
    kernel_size = (2, 2, 2)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    dilation = (2, 2, 2)
    return_indices = True
    ceil_mode = True

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 2, 15, 15, 15).numpy()
    kernel_size = (3, 2, 3)
    stride = None
    padding = (0, 1, 0)
    dilation = 1
    return_indices = False
    ceil_mode = False

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }

    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(1, 4, 8, 8, 8).numpy()
    kernel_size = 2
    stride = 1
    padding = 0
    dilation = 1
    return_indices = False
    ceil_mode = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 5, 5, 5).numpy()
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)
    return_indices = False
    ceil_mode = False

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["torch.nn.MaxPool3d_1"] = MaxPool3d_inputs()

import torch, copy
import numpy as np

def torch_nn_MaxPool3d_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    input1 = torch.randn(2, 3, 10, 10, 10).numpy()
    input_dict1 = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different kernel size, stride, and padding
    input2 = torch.randn(1, 1, 20, 20, 20).numpy()
    input_dict2 = {
        "kernel_size": (5, 5, 5),
        "stride": (3, 3, 3),
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Single int for kernel_size, stride, padding, dilation
    input3 = torch.randn(4, 2, 15, 15, 15).numpy()
    input_dict3 = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Asymmetric kernel_size, stride and padding
    input4 = torch.randn(1, 1, 25, 30, 35).numpy()
    input_dict4 = {
        "kernel_size": (3, 4, 5),
        "stride": (1, 2, 3),
        "padding": (1, 1, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Input with small dimensions to test ceil_mode
    input5 = torch.randn(1, 1, 5, 5, 5).numpy()
    input_dict5 = {
        "kernel_size": (2, 2, 2),
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_2"] = torch_nn_MaxPool3d_inputs()

import torch
import copy
import numpy as np

def MaxUnpool2d_inputs():
    list_of_inputs = []

    # Case 1: Basic case with specified kernel_size and stride
    input = torch.randn(1, 1, 2, 2)
    indices = torch.randint(0, 4, (1, 1, 2, 2)).long()
    kernel_size = 2
    stride = 2
    padding = 0
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different kernel_size and stride values
    input = torch.randn(1, 3, 3, 3)
    indices = torch.randint(0, 9, (1, 3, 3, 3)).long()
    kernel_size = 3
    stride = 1
    padding = 0
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: With padding
    input = torch.randn(1, 1, 2, 2)
    indices = torch.randint(0, 4, (1, 1, 2, 2)).long()
    kernel_size = 2
    stride = 2
    padding = 1
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Using output_size
    input = torch.randn(1, 1, 2, 2)
    indices = torch.randint(0, 4, (1, 1, 2, 2)).long()
    kernel_size = 2
    stride = 2
    padding = 0
    output_size = (1, 1, 5, 5)
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input.numpy(),
        "indices": indices.numpy(),
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Multiple channels
    input = torch.randn(1, 4, 2, 2)
    indices = torch.randint(0, 4, (1, 4, 2, 2)).long()
    kernel_size = 2
    stride = 2
    padding = 0
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": input.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs


generated_inputs["torch.nn.MaxUnpool2d_1"] = MaxUnpool2d_inputs()

import torch, copy
import numpy as np

def MaxUnpool2d_inputs():
    list_of_inputs = []

    # Case 1: Basic case with N, C, H_in, W_in
    input_tensor = torch.randn(1, 1, 2, 2)
    indices_tensor = torch.tensor([[[[0, 1], [2, 3]]]])
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "input": input_tensor.numpy(),
        "indices": indices_tensor.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: C, H_in, W_in
    input_tensor = torch.randn(1, 2, 2)
    indices_tensor = torch.tensor([[[0, 1], [2, 3]]])
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "input": input_tensor.numpy(),
        "indices": indices_tensor.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different kernel_size, stride, and padding
    input_tensor = torch.randn(1, 1, 3, 3)
    indices_tensor = torch.tensor([[[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]])
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "input": input_tensor.numpy(),
        "indices": indices_tensor.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: With output_size specified
    input_tensor = torch.randn(1, 1, 2, 2)
    indices_tensor = torch.tensor([[[[0, 1], [2, 3]]]])
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "input": input_tensor.numpy(),
        "indices": indices_tensor.numpy(),
        "output_size": (1, 1, 5, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Multiple channels
    input_tensor = torch.randn(1, 3, 2, 2)
    indices_tensor = torch.tensor([[[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]]] )
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "input": input_tensor.numpy(),
        "indices": indices_tensor.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["torch.nn.MaxUnpool2d_2"] = MaxUnpool2d_inputs()

import torch, copy
import numpy as np

def MultiLabelSoftMarginLoss_inputs():
    list_of_inputs = []

    # Input 1: Basic case with default parameters
    input1 = torch.randn(2, 3).numpy()
    target1 = torch.randint(0, 2, (2, 3)).numpy()
    input_dict1 = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: With weight
    input2 = torch.randn(2, 3).numpy()
    target2 = torch.randint(0, 2, (2, 3)).numpy()
    weight2 = torch.randn(3).numpy()
    input_dict2 = {
        "weight": weight2,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: With different reduction
    input3 = torch.randn(2, 3).numpy()
    target3 = torch.randint(0, 2, (2, 3)).numpy()
    input_dict3 = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "input": input3,
        "target": target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: With 'none' reduction
    input4 = torch.randn(2, 3).numpy()
    target4 = torch.randint(0, 2, (2, 3)).numpy()
    input_dict4 = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'none',
        "input": input4,
        "target": target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger batch size
    input5 = torch.randn(5, 4).numpy()
    target5 = torch.randint(0, 2, (5, 4)).numpy()
    input_dict5 = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input": input5,
        "target": target5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Different input range
    input6 = torch.randn(2, 3) * 100.0
    target6 = torch.randint(0, 2, (2, 3)).numpy()
    input_dict6 = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input": input6.numpy(),
        "target": target6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs


generated_inputs["torch.nn.MultiLabelSoftMarginLoss"] = MultiLabelSoftMarginLoss_inputs()

import torch
import copy
import numpy as np

def MultiMarginLoss_inputs():
    list_of_inputs = []

    input1 = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).numpy()
    target1 = torch.tensor([3]).numpy()
    input_dict1 = {
        "p": 1,
        "margin": 1.0,
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 5).numpy()
    target2 = torch.randint(0, 5, (2,)).numpy()
    weight2 = torch.randn(5).numpy()
    input_dict2 = {
        "p": 2,
        "margin": 0.5,
        "weight": weight2,
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(10, 3).numpy()
    target3 = torch.randint(0, 3, (10,)).numpy()
    input_dict3 = {
        "p": 1,
        "margin": 2.0,
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'none',
        "input": input3,
        "target": target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 4).numpy()
    target4 = torch.tensor([0]).numpy()
    weight4 = torch.tensor([0.2, 0.3, 0.1, 0.4]).numpy()
    input_dict4 = {
        "p": 1,
        "margin": 0.75,
        "weight": weight4,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input": input4,
        "target": target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(5, 2).numpy()
    target5 = torch.tensor([0, 1, 0, 1, 0]).numpy()
    input_dict5 = {
        "p": 2,
        "margin": 1.5,
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input": input5,
        "target": target5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs


generated_inputs["torch.nn.MultiMarginLoss"] = MultiMarginLoss_inputs()

import torch, copy
import numpy as np

def NLLLoss_inputs():
    list_of_inputs = []

    # Case 1: Basic case with 'mean' reduction
    input_dict = {
        "weight": None,
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'mean',
        "input": torch.randn(3, 5).log_softmax(dim=1).numpy(),
        "target": torch.tensor([1, 0, 4]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 'sum' reduction
    input_dict = {
        "weight": None,
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'sum',
        "input": torch.randn(3, 5).log_softmax(dim=1).numpy(),
        "target": torch.tensor([1, 0, 4]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 'none' reduction
    input_dict = {
        "weight": None,
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'none',
        "input": torch.randn(3, 5).log_softmax(dim=1).numpy(),
        "target": torch.tensor([1, 0, 4]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: with weight
    input_dict = {
        "weight": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).numpy(),
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": 'mean',
        "input": torch.randn(3, 5).log_softmax(dim=1).numpy(),
        "target": torch.tensor([1, 0, 4]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: with ignore_index
    input_dict = {
        "weight": None,
        "size_average": None,
        "ignore_index": 0,
        "reduce": None,
        "reduction": 'mean',
        "input": torch.randn(3, 5).log_softmax(dim=1).numpy(),
        "target": torch.tensor([1, 0, 4]).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["torch.nn.NLLLoss"] = NLLLoss_inputs()

import torch, copy
import numpy as np

def PReLU_inputs():
    list_of_inputs = []

    # Input 1: Basic case with default parameters and 1D input
    input1 = torch.randn(5).numpy()
    input_dict1 = {
        "num_parameters": 1,
        "init": 0.25,
        "dtype": None,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D input with specific init value
    input2 = torch.randn(2, 3).numpy()
    input_dict2 = {
        "num_parameters": 1,
        "init": -0.5,
        "dtype": None,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D input with num_parameters equal to the number of channels
    input3 = torch.randn(2, 4, 5).numpy()
    input_dict3 = {
        "num_parameters": 4,
        "init": 0.1,
        "dtype": None,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D input with a different dtype
    input4 = torch.randn(1, 3, 4, 4, dtype=torch.float64).numpy()
    input_dict4 = {
        "num_parameters": 3,
        "init": 0.75,
        "dtype": torch.float64,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D input with negative values
    input5 = torch.randn(5).numpy() * -1
    input_dict5 = {
        "num_parameters": 1,
        "init": 0.25,
        "dtype": None,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Empty tensor
    input6 = torch.randn(0).numpy()
    input_dict6 = {
        "num_parameters": 1,
        "init": 0.25,
        "dtype": None,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Tensor with only zeros
    input7 = torch.zeros(2, 3).numpy()
    input_dict7 = {
        "num_parameters": 1,
        "init": 0.25,
        "dtype": None,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs


generated_inputs["torch.nn.PReLU"] = PReLU_inputs()

import torch
import numpy as np
import copy

def pairwise_distance_inputs():
    list_of_inputs = []

    # Case 1: Basic case with N, D dimensions, p=2, default eps, keepdim=False
    input1 = np.random.randn(10, 5).astype(np.float32)
    input2 = np.random.randn(10, 5).astype(np.float32)
    input_dict = {
        "p": 2.0,
        "eps": 1e-06,
        "keepdim": False,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Single dimension D, p=1, keepdim=True
    input1 = np.random.randn(5).astype(np.float32)
    input2 = np.random.randn(5).astype(np.float32)
    input_dict = {
        "p": 1.0,
        "eps": 1e-06,
        "keepdim": True,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: N, D dimensions with negative p, custom eps
    input1 = np.random.randn(5, 3).astype(np.float32)
    input2 = np.random.randn(5, 3).astype(np.float32)
    input_dict = {
        "p": -1.5,
        "eps": 1e-04,
        "keepdim": False,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: N, D dimensions with p=0, keepdim=True
    input1 = np.random.randn(3, 4).astype(np.float32)
    input2 = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "p": 0.0,
        "eps": 1e-06,
        "keepdim": True,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Different shapes for inputs, but compatible
    input1 = np.random.randn(8, 7).astype(np.float32)
    input2 = np.random.randn(8, 7).astype(np.float32)
    input_dict = {
        "p": 3.0,
        "eps": 1e-08,
        "keepdim": False,
        "input1": input1,
        "input2": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["torch.nn.PairwiseDistance"] = pairwise_distance_inputs()

import torch
import numpy as np
import copy

def pixel_shuffle_inputs():
    list_of_inputs = []

    # Example 1
    upscale_factor = 2
    input_tensor = torch.randn(1, 4, 4, 4).numpy()
    input_dict = {"upscale_factor": upscale_factor, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2
    upscale_factor = 3
    input_tensor = torch.randn(2, 9, 2, 2).numpy()
    input_dict = {"upscale_factor": upscale_factor, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3
    upscale_factor = 4
    input_tensor = torch.randn(1, 16, 3, 3).numpy()
    input_dict = {"upscale_factor": upscale_factor, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4
    upscale_factor = 2
    input_tensor = torch.randn(3, 4, 5, 5).numpy()
    input_dict = {"upscale_factor": upscale_factor, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5
    upscale_factor = 3
    input_tensor = torch.randn(2, 9, 1, 1).numpy()
    input_dict = {"upscale_factor": upscale_factor, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 6: Different sized input
    upscale_factor = 2
    input_tensor = torch.randn(1, 4, 6, 8).numpy()
    input_dict = {"upscale_factor": upscale_factor, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 7: Different batch size
    upscale_factor = 3
    input_tensor = torch.randn(4, 9, 2, 2).numpy()
    input_dict = {"upscale_factor": upscale_factor, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["torch.nn.PixelShuffle"] = pixel_shuffle_inputs()

import torch, copy
import numpy as np

def PoissonNLLLoss_inputs():
    list_of_inputs = []

    # Case 1: Basic case with log_input=True
    input_tensor = torch.randn(5, 2).numpy()
    target_tensor = torch.randint(0, 5, (5, 2)).float().numpy()
    input_dict = {
        "log_input": True,
        "full": False,
        "size_average": None,
        "eps": 1e-08,
        "reduce": None,
        "reduction": 'mean',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: log_input=False, different reduction
    input_tensor = torch.rand(3, 4).numpy()
    target_tensor = torch.randint(0, 3, (3, 4)).float().numpy()
    input_dict = {
        "log_input": False,
        "full": True,
        "size_average": None,
        "eps": 1e-06,
        "reduce": None,
        "reduction": 'sum',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: reduction='none', different shape
    input_tensor = torch.randn(2, 3, 4).numpy()
    target_tensor = torch.randint(0, 4, (2, 3, 4)).float().numpy()
    input_dict = {
        "log_input": True,
        "full": False,
        "size_average": None,
        "eps": 1e-08,
        "reduce": None,
        "reduction": 'none',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 1D input
    input_tensor = torch.randn(10).numpy()
    target_tensor = torch.randint(0, 5, (10,)).float().numpy()
    input_dict = {
        "log_input": False,
        "full": True,
        "size_average": None,
        "eps": 1e-07,
        "reduce": None,
        "reduction": 'mean',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Different eps value
    input_tensor = torch.randn(4, 2).numpy()
    target_tensor = torch.randint(0, 6, (4, 2)).float().numpy()
    input_dict = {
        "log_input": True,
        "full": False,
        "size_average": None,
        "eps": 1e-04,
        "reduce": None,
        "reduction": 'mean',
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs


generated_inputs["torch.nn.PoissonNLLLoss"] = PoissonNLLLoss_inputs()

import torch, copy
import numpy as np

def ReLU6_inputs():
    list_of_inputs = []

    input1 = torch.randn(2).numpy()
    input_dict1 = {
        "inplace": False,
        "input": input1
    }
    list_of_inputs.append(input_dict1)

    input2 = torch.randn(2, 3).numpy()
    input_dict2 = {
        "inplace": True,
        "input": input2
    }
    list_of_inputs.append(input_dict2)

    input3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {
        "inplace": False,
        "input": input3
    }
    list_of_inputs.append(input_dict3)

    input4 = torch.randint(-5, 10, (2, 3, 4, 5)).numpy()
    input_dict4 = {
        "inplace": True,
        "input": input4
    }
    list_of_inputs.append(input_dict4)

    input5 = torch.randn(1, 5, 5).numpy()
    input_dict5 = {
        "inplace": False,
        "input": input5
    }
    list_of_inputs.append(input_dict5)
    
    return list_of_inputs


generated_inputs["torch.nn.ReLU6"] = ReLU6_inputs()

