
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_cross_inputs():
    list_of_inputs = []

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[2.0, 4.0, -6.0], [5.0, 7.0, 9.0]], dtype=np.float32)
    input_dict = {'name': 'cross_1', 'a': a, 'b': b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float64)
    b = np.array([[2.0, -4.0, 6.0], [-5.0, 7.0, -9.0]], dtype=np.float64)
    input_dict = {'name': 'cross_2', 'a': a, 'b': b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    b = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int32)
    input_dict = {'name': 'cross_3', 'a': a, 'b': b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    b = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int16)
    input_dict = {'name': 'cross_5', 'a': a, 'b': b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    b = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int8)
    input_dict = {'name': 'cross_6', 'a': a, 'b': b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    b = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int64)
    input_dict = {'name': 'cross_7', 'a': a, 'b': b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
    b = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.float16)
    input_dict = {'name': 'cross_8', 'a': a, 'b': b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Cross"] = tf_raw_ops_cross_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Cross'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Cross', generated_inputs['tf.raw_ops.Cross'], lib="tf", suffix=0)
