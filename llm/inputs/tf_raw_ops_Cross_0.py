
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_cross_inputs():
    list_of_inputs = []

    # Input 1: Basic 3-element vectors
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {"a": a, "b": b, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Another set of 3-element vectors
    a = np.array([-1, 2, -3], dtype=np.int32)
    b = np.array([4, -5, 6], dtype=np.int32)
    input_dict = {"a": a, "b": b, "name": "cross_product_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch of 3-element vectors (2D tensor)
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    b = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float64)
    input_dict = {"a": a, "b": b, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch of 3-element vectors with negative values
    a = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int64)
    b = np.array([[7, 8, 9], [-10, 11, -12]], dtype=np.int64)
    input_dict = {"a": a, "b": b, "name": "cross_product_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor with innermost dimension of 3
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    b = np.array([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]], dtype=np.float32)
    input_dict = {"a": a, "b": b, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensor with different values
    a = np.array([[[1, -2, 3], [-4, 5, -6]], [[7, -8, 9], [-10, 11, -12]]], dtype=np.int32)
    b = np.array([[[13, 14, -15], [-16, 17, 18]], [[-19, 20, 21], [22, -23, 24]]], dtype=np.int32)
    input_dict = {"a": a, "b": b, "name": "cross_product_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D tensor
    a = np.random.rand(2, 2, 2, 3).astype(np.float64)
    b = np.random.rand(2, 2, 2, 3).astype(np.float64)
    input_dict = {"a": a, "b": b, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D tensor int type
    a = np.random.randint(-5, 5, size=(2, 2, 2, 3), dtype=np.int32)
    b = np.random.randint(-5, 5, size=(2, 2, 2, 3), dtype=np.int32)
    input_dict = {"a": a, "b": b, "name": "cross_product_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float16 type tensors
    a = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float16)
    input_dict = {"a": a, "b": b, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint8 type tensors
    a = np.array([1, 2, 3], dtype=np.uint8)
    b = np.array([4, 5, 6], dtype=np.uint8)
    input_dict = {"a": a, "b": b, "name": "cross_product_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Cross"] = tf_raw_ops_cross_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Cross'.")

check_valid('tf.raw_ops.Cross', generated_inputs['tf.raw_ops.Cross'], lib="tf", suffix=0)
