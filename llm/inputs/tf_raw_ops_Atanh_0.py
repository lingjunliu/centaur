
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_raw_ops_atanh_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensor within range [-1, 1]
    x = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": None}
    list_of_inputs.append(input_dict)

    # Input 2: float64 tensor with edge cases and a zero
    x = np.array([-1.0, 1.0, 0.0, 0.75, -0.25], dtype=np.float64)
    input_dict = {"x": tf.constant(x), "name": "atanh_op_2"}
    list_of_inputs.append(input_dict)

    # Input 3: bfloat16 tensor
    x = np.array([-0.8, 0.2, 0.9, -0.1], dtype=np.float16)  # Note: Changed dtype to np.float16
    input_dict = {"x": tf.constant(x), "name": "atanh_op_3"}
    list_of_inputs.append(input_dict)

    # Input 4: half tensor
    x = np.array([0.3, -0.6, 0.4, -0.95], dtype=np.float16)
    input_dict = {"x": tf.constant(x), "name": "atanh_op_4"}
    list_of_inputs.append(input_dict)
    
    # Input 5: Complex64 tensor
    x = np.array([0.1 + 0.1j, -0.2 - 0.2j, 0.3 - 0.3j, -0.4 + 0.4j], dtype=np.complex64)
    input_dict = {"x": tf.constant(x), "name": "atanh_op_5"}
    list_of_inputs.append(input_dict)

    # Input 6: Complex128 tensor
    x = np.array([0.5 + 0.5j, -0.6 - 0.6j, 0.7 - 0.7j, -0.8 + 0.8j], dtype=np.complex128)
    input_dict = {"x": tf.constant(x), "name": "atanh_op_6"}
    list_of_inputs.append(input_dict)

    # Input 7: Multi-dimensional float32 tensor
    x = np.array([[0.1, 0.2], [-0.3, -0.4]], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": "atanh_op_7"}
    list_of_inputs.append(input_dict)

    # Input 8: Multi-dimensional float64 tensor with more elements
    x = np.array([[-0.9, 0.8, -0.7], [0.6, -0.5, 0.4]], dtype=np.float64)
    input_dict = {"x": tf.constant(x), "name": "atanh_op_8"}
    list_of_inputs.append(input_dict)

    # Input 9: float32 tensor with values near -1 and 1
    x = np.array([-0.99, 0.99, -0.01, 0.01], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": "atanh_op_9"}
    list_of_inputs.append(input_dict)

    # Input 10: Large float32 array
    x = np.linspace(-0.9, 0.9, num=100, dtype=np.float32)
    input_dict = {"x": tf.constant(x), "name": "atanh_op_10"}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
atanh_inputs = tf_raw_ops_atanh_inputs()
generated_inputs["tf.raw_ops.Atanh"] = atanh_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Atanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Atanh'.")

check_valid('tf.raw_ops.Atanh', generated_inputs['tf.raw_ops.Atanh'], lib="tf", suffix=0)
