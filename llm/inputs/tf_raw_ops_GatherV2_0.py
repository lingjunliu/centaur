
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_gather_v2_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D gather
    params = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    indices = np.array([0, 2, 4], dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    batch_dims = 0
    name = "gather_1d"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D gather along axis 0
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    batch_dims = 0
    name = "gather_2d_axis0"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D gather along axis 1
    params = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int32)
    axis = np.array(1, dtype=np.int32)
    batch_dims = 0
    name = "gather_2d_axis1"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D gather
    params = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    indices = np.array([0, 1], dtype=np.int32)
    axis = np.array(1, dtype=np.int32)
    batch_dims = 0
    name = "gather_3d"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative axis
    params = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int32)
    axis = np.array(-1, dtype=np.int32)
    batch_dims = 0
    name = "gather_negative_axis"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: Scalar index
    params = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    indices = np.array(2, dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    batch_dims = 0
    name = "gather_scalar_index"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional indices
    params = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    indices = np.array([[0, 1], [1, 2]], dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    batch_dims = 0
    name = "gather_multi_dim_indices"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: batch_dims = 1
    params = np.arange(12, dtype=np.int32).reshape((2, 2, 3))
    indices = np.array([[0, 1], [1, 0]], dtype=np.int32)
    axis = np.array(1, dtype=np.int32)
    batch_dims = 1
    name = "gather_batch_dims_1"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: different dtype
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int64)
    axis = np.array(0, dtype=np.int32)
    batch_dims = 0
    name = "gather_2d_axis0_int64_indices"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: axis is int64
    params = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    indices = np.array([0, 2], dtype=np.int32)
    axis = np.array(0, dtype=np.int64)
    batch_dims = 0
    name = "gather_2d_axis0_int64_axis"
    input_dict = {"params": params, "indices": indices, "axis": axis, "batch_dims": batch_dims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_gather_v2_inputs()
generated_inputs["tf.raw_ops.GatherV2"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.GatherV2"].append({
        "args": [],
        "kwargs": {
            "params": input_dict["params"],
            "indices": input_dict["indices"],
            "axis": input_dict["axis"],
            "batch_dims": input_dict["batch_dims"],
            "name": input_dict["name"]
        }
    })

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.GatherV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GatherV2'.")

check_valid('tf.raw_ops.GatherV2', generated_inputs['tf.raw_ops.GatherV2'], lib="tf", suffix=0)
