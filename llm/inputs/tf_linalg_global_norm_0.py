
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_global_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic list of tensors
    t_list = [tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32)), tf.constant(np.array([4.0, 5.0], dtype=np.float32))]
    name = "global_norm_1"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List with None (at the end, to avoid issues with shape inference).  Removing None from middle to avoid problems in earlier steps
    t_list = [tf.constant(np.array([1.0, 2.0], dtype=np.float32)), tf.constant(np.array([3.0], dtype=np.float32))]
    name = "global_norm_2"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List of tensors with different shapes
    t_list = [tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)), tf.constant(np.array([5.0], dtype=np.float32))]
    name = "global_norm_3"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty list
    t_list = []
    name = "global_norm_4"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List of integers (converted to float32)
    t_list = [tf.constant(np.array([1, 2, 3], dtype=np.float32)), tf.constant(np.array([4, 5], dtype=np.float32))]
    name = "global_norm_5"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: List with negative values
    t_list = [tf.constant(np.array([-1.0, 2.0], dtype=np.float32)), tf.constant(np.array([3.0, -4.0], dtype=np.float32))]
    name = "global_norm_6"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: List with zero values
    t_list = [tf.constant(np.array([0.0, 0.0], dtype=np.float32)), tf.constant(np.array([0.0, 0.0], dtype=np.float32))]
    name = "global_norm_7"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single tensor in the list
    t_list = [tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))]
    name = "global_norm_8"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: List of 3D tensors
    t_list = [tf.constant(np.random.rand(2, 3, 4).astype(np.float32)), tf.constant(np.random.rand(3, 2, 1).astype(np.float32))]
    name = "global_norm_9"
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
