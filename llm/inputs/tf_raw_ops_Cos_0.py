
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_cos_inputs():
    list_of_inputs = []

    # Input 1: float32, simple array
    x = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, multi-dimensional array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"x": tf.constant(x), "name": "cos_op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16, negative values
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float16)
    input_dict = {"x": tf.constant(x, dtype=tf.bfloat16), "name": "cos_op_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex128
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
    input_dict = {"x": tf.constant(x), "name": "cos_op_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: half
    x = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    input_dict = {"x": tf.constant(x, dtype=tf.float16), "name": "cos_op_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, larger values
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, 3D tensor
    x = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict = {"x": tf.constant(x), "name": "cos_op_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: bfloat16, tensor with zeros
    x = np.array([0.0, 0.0, 0.0], dtype=np.float16)
    input_dict = {"x": tf.constant(x, dtype=tf.bfloat16), "name": "cos_op_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: half, large values
    x = np.array([1000.0, 2000.0, 3000.0], dtype=np.float16)
    input_dict = {"x": tf.constant(x, dtype=tf.float16), "name": "cos_op_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Cos"] = tf_raw_ops_cos_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Cos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Cos'.")

check_valid('tf.raw_ops.Cos', generated_inputs['tf.raw_ops.Cos'], lib="tf", suffix=0)
