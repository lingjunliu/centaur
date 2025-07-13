
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_edit_distance_inputs():
    list_of_inputs = []

    # Input 1
    hypothesis_indices = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
    hypothesis_values = np.array([1, 2, 3], dtype=np.int64)
    hypothesis_shape = np.array([1, 3], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=np.int64)
    truth_values = np.array([1, 2, 4, 3], dtype=np.int64)
    truth_shape = np.array([1, 4], dtype=np.int64)
    normalize = True
    name = "edit_distance_1"

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": truth_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    hypothesis_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    hypothesis_values = np.array([1, 2], dtype=np.int64)
    hypothesis_shape = np.array([1, 2], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
    truth_values = np.array([1, 3, 2], dtype=np.int64)
    truth_shape = np.array([1, 3], dtype=np.int64)
    normalize = False
    name = "edit_distance_2"

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": truth_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    hypothesis_indices = np.array([[0, 0]], dtype=np.int64)
    hypothesis_values = np.array([1], dtype=np.int64)
    hypothesis_shape = np.array([1, 1], dtype=np.int64)
    truth_indices = np.array([[0, 0]], dtype=np.int64)
    truth_values = np.array([1], dtype=np.int64)
    truth_shape = np.array([1, 1], dtype=np.int64)
    normalize = True
    name = "edit_distance_3"

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": truth_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    hypothesis_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    hypothesis_values = np.array([1, 2], dtype=np.int64)
    hypothesis_shape = np.array([1, 2], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    truth_values = np.array([1, 2], dtype=np.int64)
    truth_shape = np.array([1, 2], dtype=np.int64)
    normalize = False
    name = "edit_distance_4"

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": truth_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    hypothesis_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    hypothesis_values = np.array([1, 2], dtype=np.int64)
    hypothesis_shape = np.array([1, 2], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
    truth_values = np.array([1, 2, 3], dtype=np.int64)
    truth_shape = np.array([1, 3], dtype=np.int64)
    normalize = True
    name = "edit_distance_5"

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": truth_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    hypothesis_indices = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int64)
    hypothesis_values = np.array([1, 2], dtype=np.int64)
    hypothesis_shape = np.array([1, 1, 2], dtype=np.int64)
    truth_indices = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=np.int64)
    truth_values = np.array([1, 2, 3], dtype=np.int64)
    truth_shape = np.array([1, 1, 3], dtype=np.int64)
    normalize = True
    name = "edit_distance_6"

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": truth_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    hypothesis_indices = np.array([[0, 0], [1, 0]], dtype=np.int64)
    hypothesis_values = np.array([1, 2], dtype=np.int64)
    hypothesis_shape = np.array([2, 1], dtype=np.int64)
    truth_indices = np.array([[0, 0], [1, 0]], dtype=np.int64)
    truth_values = np.array([1, 2], dtype=np.int64)
    truth_shape = np.array([2, 1], dtype=np.int64)
    normalize = False
    name = "edit_distance_7"

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": truth_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    hypothesis_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    hypothesis_values = np.array([1, 2], dtype=np.int64)
    hypothesis_shape = np.array([1, 2], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    truth_values = np.array([1, 3], dtype=np.int64)
    truth_shape = np.array([1, 2], dtype=np.int64)
    normalize = True
    name = "edit_distance_8"

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": truth_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    hypothesis_indices = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
    hypothesis_values = np.array([1, 2, 3], dtype=np.int64)
    hypothesis_shape = np.array([1, 3], dtype=np.int64)
    truth_indices = np.array([[0, 0]], dtype=np.int64)
    truth_values = np.array([1], dtype=np.int64)
    truth_shape = np.array([1, 1], dtype=np.int64)
    normalize = False
    name = "edit_distance_9"

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": truth_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    hypothesis_indices = np.array([[0, 0]], dtype=np.int64)
    hypothesis_values = np.array([1], dtype=np.int64)
    hypothesis_shape = np.array([1, 1], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
    truth_values = np.array([1, 2, 3], dtype=np.int64)
    truth_shape = np.array([1, 3], dtype=np.int64)
    normalize = True
    name = "edit_distance_10"

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": truth_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_edit_distance_inputs()
generated_inputs["tf.raw_ops.EditDistance"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.EditDistance"].append({
        'hypothesis_indices': tf.constant(input_dict["hypothesis_indices"]),
        'hypothesis_values': tf.constant(input_dict["hypothesis_values"]),
        'hypothesis_shape': tf.constant(input_dict["hypothesis_shape"]),
        'truth_indices': tf.constant(input_dict["truth_indices"]),
        'truth_values': tf.constant(input_dict["truth_values"]),
        'truth_shape': tf.constant(input_dict["truth_shape"]),
        'normalize': input_dict["normalize"],
        'name': input_dict["name"]
    })

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.EditDistance' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EditDistance'.")

check_valid('tf.raw_ops.EditDistance', generated_inputs['tf.raw_ops.EditDistance'], lib="tf", suffix=0)
