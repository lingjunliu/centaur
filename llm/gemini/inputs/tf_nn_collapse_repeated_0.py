
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_collapse_repeated_inputs():
    list_of_inputs = []
    
    # Input 1
    labels = np.array([[1, 1, 2, 2, 1], [1, 2, 3, 3, 3]], dtype=np.int32)
    seq_length = np.array([5, 5], dtype=np.int32)
    name = "collapse_1"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 2
    labels = np.array([[1, 2, 2, 3], [4, 4, 4, 4]], dtype=np.int64)
    seq_length = np.array([3, 4], dtype=np.int64)
    name = "collapse_2"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 3
    labels = np.array([[9, 9, 9], [8, 8, 7], [1, 2, 3]], dtype=np.int32)
    seq_length = np.array([3, 2, 1], dtype=np.int32)
    name = "collapse_3"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 4
    labels = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
    seq_length = np.array([5], dtype=np.int32)
    name = "collapse_4"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 5
    labels = np.array([[1, 2, 1, 2, 1, 2]], dtype=np.int64)
    seq_length = np.array([6], dtype=np.int64)
    name = "collapse_5"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 6
    labels = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int32)
    seq_length = np.array([3, 3], dtype=np.int32)
    name = "collapse_6"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 7
    labels = np.array([[10, 10, 20, 20, 30, 30, 40, 40]], dtype=np.int32)
    seq_length = np.array([8], dtype=np.int32)
    name = "collapse_7"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 8
    labels = np.array([[-1, -1, -2, -2], [-3, -3, -3, -4]], dtype=np.int32)
    seq_length = np.array([4, 4], dtype=np.int32)
    name = "collapse_8"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 9
    labels = np.array([[100, 100], [200, 200], [300, 300]], dtype=np.int64)
    seq_length = np.array([2, 1, 2], dtype=np.int64)
    name = "collapse_9"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    # Input 10
    labels = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.int32)
    seq_length = np.array([10], dtype=np.int32)
    name = "collapse_10"
    list_of_inputs.append({
        "labels": labels,
        "seq_length": seq_length,
        "name": name
    })
    
    return list_of_inputs

generated_inputs["tf.nn.collapse_repeated"] = tf_nn_collapse_repeated_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.collapse_repeated' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.collapse_repeated'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.collapse_repeated', generated_inputs['tf.nn.collapse_repeated'], lib="tf", suffix=0)
