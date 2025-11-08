
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def generate_sparse_tensor_inputs():
    list_of_inputs = []
    
    # Input 1: Basic 2D sparse tensor with integer values
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    dense_shape = np.array([3, 4], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 2: 3D sparse tensor with float values
    indices = np.array([[0, 1, 2], [1, 0, 1]], dtype=np.int64)
    values = np.array([3.14, 2.71], dtype=np.float32)
    dense_shape = np.array([2, 3, 4], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 3: 1D sparse tensor with negative values
    indices = np.array([[0], [2], [4]], dtype=np.int64)
    values = np.array([-1, -2, -3], dtype=np.int32)
    dense_shape = np.array([5], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 4: 2D sparse tensor with mixed types (int and float)
    indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    values = np.array([1, 2.5], dtype=np.float32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 5: 3D sparse tensor with zero values
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    values = np.array([0, 0], dtype=np.int32)
    dense_shape = np.array([2, 2, 2], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 6: 1D sparse tensor with string values
    indices = np.array([[0], [2]], dtype=np.int64)
    values = np.array(['hello', 'world'], dtype=np.object_)
    dense_shape = np.array([3], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 7: Large sparse tensor (10x10)
    indices = np.array([[0, 0], [9, 9]], dtype=np.int64)
    values = np.array([100, 200], dtype=np.int32)
    dense_shape = np.array([10, 10], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 8: 2D sparse tensor with duplicate indices (but valid for this API)
    indices = np.array([[0, 0], [0, 0]], dtype=np.int64)
    values = np.array([5, 5], dtype=np.int32)
    dense_shape = np.array([3, 3], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 9: 4D sparse tensor (high dimensional)
    indices = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    values = np.array([1000, 2000], dtype=np.int32)
    dense_shape = np.array([2, 3, 4, 5], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: 3D sparse tensor with negative index values (valid for this API)
    indices = np.array([[0, 1, 2], [1, 0, 1]], dtype=np.int64)
    values = np.array([-10, -20], dtype=np.int32)
    dense_shape = np.array([2, 3, 4], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.sparse.SparseTensor"] = generate_sparse_tensor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.SparseTensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.SparseTensor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.SparseTensor', generated_inputs['tf.sparse.SparseTensor'], lib="tf", suffix=0)
