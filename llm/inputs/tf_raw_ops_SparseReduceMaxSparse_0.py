
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseReduceMaxSparse_inputs():
    list_of_inputs = []

    # Input 1
    input_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_shape = np.array([2, 2], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = False

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_shape = np.array([2, 2], dtype=np.int64)
    reduction_axes = np.array([1], dtype=np.int32)
    keep_dims = True

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_indices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.int64)
    input_values = np.array([1, 2, 3], dtype=np.int32)
    input_shape = np.array([1, 2, 2], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = False

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    input_values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_shape = np.array([3, 3], dtype=np.int64)
    reduction_axes = np.array([0, 1], dtype=np.int32)
    keep_dims = False

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_values = np.array([1, 2, 3], dtype=np.int64)
    input_shape = np.array([2, 2], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = True

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    input_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_values = np.array([1, 2, 3], dtype=np.int8)
    input_shape = np.array([2, 2], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = False

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_values = np.array([1, 2, 3], dtype=np.uint8)
    input_shape = np.array([2, 2], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = False

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_values = np.array([1, 2, 3], dtype=np.int16)
    input_shape = np.array([2, 2], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = False

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_values = np.array([1, 2, 3], dtype=np.uint16)
    input_shape = np.array([2, 2], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = False

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_values = np.array([1, 2, 3], dtype=np.uint32)
    input_shape = np.array([2, 2], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = False

    input_dict = {
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes,
        "keep_dims": keep_dims,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseReduceMaxSparse"] = tf_raw_ops_SparseReduceMaxSparse_inputs()

import tensorflow as tf
import numpy as np

def check_valid(api, input_list, lib="tf", suffix=0):
    def run_api(api, input_dict, cpu=True, lib="tf"):
        if lib == "torch":
            import torch
            device = torch.device('cpu')
            inp = {"args": [], "kwargs": {}}
            for k, v in input_dict.items():
                if isinstance(v, np.ndarray):
                    inp["kwargs"][k] = torch.from_numpy(v).to(device)
                else:
                    inp["kwargs"][k] = v
        elif lib == "tf":
            inp = {"args": [], "kwargs": {}}
            for k, v in input_dict.items():
                inp["kwargs"][k] = tf.convert_to_tensor(v) if isinstance(v, np.ndarray) else v
        try:
            if lib == "tf":
                func = eval(api)
                result = func(**inp["kwargs"])
            elif lib == "torch":
                func = eval(api)
                result = func(**inp["kwargs"])
            if isinstance(result, (tf.Tensor, tf.Variable)):
                result = result.numpy()
            elif isinstance(result, (list, tuple)):
                result = [r.numpy() if isinstance(r, (tf.Tensor, tf.Variable)) else r for r in result]
        except Exception as e:
            print(f"API {api} failed with error: {e}")
            return None

        return result
    for i, input_dict in enumerate(input_list):
        output = run_api(api, input_dict, cpu=True, lib=lib)

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseReduceMaxSparse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseReduceMaxSparse'.")

check_valid('tf.raw_ops.SparseReduceMaxSparse', generated_inputs['tf.raw_ops.SparseReduceMaxSparse'], lib="tf", suffix=0)
