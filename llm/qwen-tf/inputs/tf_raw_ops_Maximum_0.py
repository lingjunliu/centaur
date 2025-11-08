
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_maximum_inputs():
    list_of_inputs = []
    
    # Input 1: Two 1D tensors with float32 dtype
    x = np.array([0., 0., 0., 0.], dtype=np.float32)
    y = np.array([-2., 0., 2., 5.], dtype=np.float32)
    input_dict = {"name": "test1", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 2: Two 1D tensors with int32 dtype
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([-1, 0, 2, 3], dtype=np.int32)
    input_dict = {"name": "test2", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 3: Two 1D tensors with float64 dtype
    x = np.array([1.5, 2.7, 3.1], dtype=np.float64)
    y = np.array([2.1, 1.8, 3.9], dtype=np.float64)
    input_dict = {"name": "test3", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 4: Two 1D tensors with int16 dtype
    x = np.array([1, 2], dtype=np.int16)
    y = np.array([2, 3], dtype=np.int16)
    input_dict = {"name": "test4", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 5: Two 1D tensors with uint8 dtype
    x = np.array([255, 0, 128], dtype=np.uint8)
    y = np.array([128, 128, 0], dtype=np.uint8)
    input_dict = {"name": "test5", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 6: Two 1D tensors with int64 dtype
    x = np.array([9223372036854775807, 1], dtype=np.int64)
    y = np.array([-9223372036854775808, 0], dtype=np.int64)
    input_dict = {"name": "test6", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 7: Two 1D tensors with float32 dtype with negative values
    x = np.array([-1.5, -2.7, 3.1], dtype=np.float32)
    y = np.array([-2.1, 1.8, 3.9], dtype=np.float32)
    input_dict = {"name": "test7", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 8: Two 1D tensors with half dtype (float16)
    x = np.array([1.5, 2.7], dtype=np.float16)
    y = np.array([2.1, 1.8], dtype=np.float16)
    input_dict = {"name": "test8", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 9: Two 1D tensors with bfloat16 dtype
    x = np.array([1.5, 2.7], dtype=np.float32)
    y = np.array([2.1, 1.8], dtype=np.float32)
    input_dict = {"name": "test9", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict)

    # Input 10: Two 1D tensors with uint32 dtype
    x = np.array([4294967295, 1], dtype=np.uint32)
    y = np.array([1, 2], dtype=np.uint32)
    input_dict = {"name": "test10", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict)

    return list_of_inputs

generated_inputs["tf.raw_ops.Maximum"] = tf_raw_ops_maximum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Maximum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Maximum', generated_inputs['tf.raw_ops.Maximum'], lib="tf", suffix=0)
