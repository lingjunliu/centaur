
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_l2loss_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D float32
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"t": t, "name": "l2loss_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    input_dict = {"t": t, "name": "l2loss_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float32
    t = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"t": t, "name": "l2loss_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D half
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    input_dict = {"t": t, "name": "l2loss_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float32, replacing bfloat16
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {"t": t, "name": "l2loss_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 with negative values
    t = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    input_dict = {"t": t, "name": "l2loss_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float64 with zeros
    t = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    input_dict = {"t": t, "name": "l2loss_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D half with negative values
    t = np.array([[[1.0, -2.0], [3.0, 4.0]], [[-5.0, 6.0], [7.0, -8.0]]], dtype=np.float16)
    input_dict = {"t": t, "name": "l2loss_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 with mixed values, replacing bfloat16
    t = np.array([[-1.0, 0.0], [3.0, -4.0]], dtype=np.float32)
    input_dict = {"t": t, "name": "l2loss_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty 2D float32
    t = np.array([[]], dtype=np.float32)
    input_dict = {"t": t, "name": "l2loss_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.L2Loss"] = tf_raw_ops_l2loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.L2Loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.L2Loss'.")

check_valid('tf.raw_ops.L2Loss', generated_inputs['tf.raw_ops.L2Loss'], lib="tf", suffix=0)
