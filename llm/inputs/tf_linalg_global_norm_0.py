
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_global_norm_inputs():
    list_of_inputs = []

    # Input 1: List of tensors with different shapes
    t_list = [tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
              tf.constant(np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32))]
    name = "global_norm_1"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of tensors with None
    t_list = [tf.constant(np.array([1.0, 2.0], dtype=np.float32)), None, tf.constant(np.array([3.0], dtype=np.float32))]
    name = "global_norm_2"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single tensor in a list
    t_list = [tf.constant(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))]
    name = "global_norm_3"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List of tensors with negative values
    t_list = [tf.constant(np.array([-1.0, 2.0, -3.0], dtype=np.float32)),
              tf.constant(np.array([[4.0, -5.0], [-6.0, 7.0]], dtype=np.float32))]
    name = "global_norm_5"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: List of tensors with zero values
    t_list = [tf.constant(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
              tf.constant(np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32))]
    name = "global_norm_7"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: List of tensors with different dtypes (float32 and float64)
    t_list = [tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
              tf.constant(np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float64))]
    name = "global_norm_8"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: List of tensors with large values
    t_list = [tf.constant(np.array([1e5, 2e5, 3e5], dtype=np.float32)),
              tf.constant(np.array([[4e5, 5e5], [6e5, 7e5]], dtype=np.float32))]
    name = "global_norm_9"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: List of Tensors with rank 0
    t_list = [tf.constant(np.array(5.0, dtype=np.float32)), tf.constant(np.array(10.0, dtype=np.float32))]
    name = "global_norm_11"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: List with a single 2D tensor
    t_list = [tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))]
    name = "global_norm_12"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.global_norm"] = tf_linalg_global_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.global_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.global_norm'.")

check_valid('tf.linalg.global_norm', generated_inputs['tf.linalg.global_norm'], lib="tf", suffix=0)
