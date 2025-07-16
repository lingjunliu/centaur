
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_euclidean_norm_inputs():
    list_of_inputs = []

    # Input 1: float32, axis=0, keep_dims=False
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis_tensor = np.array([0], dtype=np.int32)
    keep_dims = False
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, axis=1, keep_dims=True
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    axis_tensor = np.array([1], dtype=np.int32)
    keep_dims = True
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32, axis=[0, 1], keep_dims=False
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int32)
    axis_tensor = np.array([0, 1], dtype=np.int32)
    keep_dims = False
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64, axis=0, keep_dims=True
    input_tensor = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex64)
    axis_tensor = np.array([0], dtype=np.int32)
    keep_dims = True
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, axis=0, keep_dims=False, 3D tensor
    input_tensor = np.random.rand(2, 3, 4).astype(np.float32)
    axis_tensor = np.array([0], dtype=np.int32)
    keep_dims = False
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int64, axis=1, keep_dims=True, negative axis
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int64)
    axis_tensor = np.array([-1], dtype=np.int32)
    keep_dims = True
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, axis=[0, 2], keep_dims=False, 3D Tensor
    input_tensor = np.random.rand(2, 3, 4).astype(np.float32)
    axis_tensor = np.array([0, 2], dtype=np.int32)
    keep_dims = False
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, axis=[1, 2], keep_dims=True, 3D Tensor
    input_tensor = np.random.rand(2, 3, 4).astype(np.float64)
    axis_tensor = np.array([1, 2], dtype=np.int32)
    keep_dims = True
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: int32, axis=2, keep_dims=False, 4D Tensor
    input_tensor = np.random.randint(0, 10, size=(2, 3, 4, 5)).astype(np.int32)
    axis_tensor = np.array([2], dtype=np.int32)
    keep_dims = False
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: complex128, axis=[0, 1, 3], keep_dims=True, 4D Tensor
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.complex128) + 1j*np.random.rand(2, 3, 4, 5).astype(np.complex128)
    axis_tensor = np.array([0, 1, 3], dtype=np.int32)
    keep_dims = True
    input_dict = {"input": input_tensor, "axis": axis_tensor, "keep_dims": keep_dims, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.EuclideanNorm"] = tf_raw_ops_euclidean_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.EuclideanNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EuclideanNorm'.")

check_valid('tf.raw_ops.EuclideanNorm', generated_inputs['tf.raw_ops.EuclideanNorm'], lib="tf", suffix=0)
