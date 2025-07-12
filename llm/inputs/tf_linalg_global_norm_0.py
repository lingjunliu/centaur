
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_global_norm_inputs():
    list_of_inputs = []

    # Input 1: List of tensors with different shapes and dtypes.
    t_list = [tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
              tf.constant(np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float64)),
              tf.constant(np.array([8, 9], dtype=np.int32))]
    name = "global_norm_1"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List with None elements.
    t_list = [tf.constant(np.array([1.0, 2.0], dtype=np.float32)), None, tf.constant(np.array([3.0], dtype=np.float32))]
    name = "global_norm_2"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List with only one tensor.
    t_list = [tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))]
    name = "global_norm_3"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty list. Should still be valid and return 0.0.
    t_list = []
    name = "global_norm_4"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List with zero tensors
    t_list = [tf.constant(np.array([0.0, 0.0], dtype=np.float32)), tf.constant(np.array([0.0], dtype=np.float32))]
    name = "global_norm_5"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger tensors and mixed types
    t_list = [tf.constant(np.random.rand(100).astype(np.float32)),
              tf.constant(np.random.randint(0, 10, size=(50, 2)).astype(np.int32)),
              tf.constant(np.random.rand(20, 20).astype(np.float64))]
    name = "global_norm_6"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: More complex tensors
    t_list = [tf.constant(np.array([4.0, 5.0], dtype=np.float32)), tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))]
    name = "global_norm_7"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: More complex tensors
    t_list = [tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)), tf.constant(np.array([7.0, 8.0], dtype=np.float32))]
    name = "global_norm_8"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Mixed None values
    t_list = [None, tf.constant(np.array([4.0], dtype=np.float32))]
    name = "global_norm_9"
    input_dict = {"t_list": t_list, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Just one None value to fulfill the requirement of at least 10 inputs
    t_list = [None]
    name = "global_norm_10"
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
