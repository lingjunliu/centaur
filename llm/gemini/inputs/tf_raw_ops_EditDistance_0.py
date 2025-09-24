
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_editdistance_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    hypothesis_indices = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
    hypothesis_values = np.array([1, 2, 3], dtype=np.int64)
    hypothesis_shape = np.array([1, 3], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    truth_values = np.array([1, 2], dtype=np.int64)
    truth_shape = np.array([1, 2], dtype=np.int64)
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

    # Input 2: No normalization
    hypothesis_indices = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
    hypothesis_values = np.array([1, 2, 3], dtype=np.int64)
    hypothesis_shape = np.array([1, 3], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    truth_values = np.array([1, 2], dtype=np.int64)
    truth_shape = np.array([1, 2], dtype=np.int64)
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

    # Input 3: Empty hypothesis
    hypothesis_indices = np.array([], dtype=np.int64).reshape(0, 2)
    hypothesis_values = np.array([], dtype=np.int64)
    hypothesis_shape = np.array([1, 0], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    truth_values = np.array([1, 2], dtype=np.int64)
    truth_shape = np.array([1, 2], dtype=np.int64)
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

    # Input 4: Empty truth
    hypothesis_indices = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
    hypothesis_values = np.array([1, 2, 3], dtype=np.int64)
    hypothesis_shape = np.array([1, 3], dtype=np.int64)
    truth_indices = np.array([], dtype=np.int64).reshape(0, 2)
    truth_values = np.array([], dtype=np.int64)
    truth_shape = np.array([1, 0], dtype=np.int64)
    normalize = True
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

    # Input 5: Identical hypothesis and truth
    hypothesis_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    hypothesis_values = np.array([1, 2], dtype=np.int64)
    hypothesis_shape = np.array([1, 2], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    truth_values = np.array([1, 2], dtype=np.int64)
    truth_shape = np.array([1, 2], dtype=np.int64)
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

    # Input 6: Different values type (float32)
    hypothesis_indices = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
    hypothesis_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    hypothesis_shape = np.array([1, 3], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    truth_values = np.array([1.0, 2.0], dtype=np.float32)
    truth_shape = np.array([1, 2], dtype=np.int64)
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

     # Input 7: Different values type (int64), different shape
    hypothesis_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    hypothesis_values = np.array([1, 2, 4], dtype=np.int64)
    hypothesis_shape = np.array([2, 2], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0, 1], [1,0], [1,1]], dtype=np.int64)
    truth_values = np.array([1, 2, 5, 7], dtype=np.int64)
    truth_shape = np.array([2, 2], dtype=np.int64)
    normalize = True
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

    # Input 8: Multiple batches (shape [2,3])
    hypothesis_indices = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1,1]], dtype=np.int64)
    hypothesis_values = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    hypothesis_shape = np.array([2, 3], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0, 1], [1,0], [1,1], [1,2]], dtype=np.int64)
    truth_values = np.array([1, 2, 4, 5, 6], dtype=np.int64)
    truth_shape = np.array([2, 3], dtype=np.int64)
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

    # Input 9: Multiple batches, no normalize
    hypothesis_indices = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1,1]], dtype=np.int64)
    hypothesis_values = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    hypothesis_shape = np.array([2, 3], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0, 1], [1,0], [1,1], [1,2]], dtype=np.int64)
    truth_values = np.array([1, 2, 4, 5, 6], dtype=np.int64)
    truth_shape = np.array([2, 3], dtype=np.int64)
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

    # Input 10: hypothesis longer than truth
    hypothesis_indices = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=np.int64)
    hypothesis_values = np.array([1, 2, 3, 7], dtype=np.int64)
    hypothesis_shape = np.array([1, 4], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    truth_values = np.array([1, 2], dtype=np.int64)
    truth_shape = np.array([1, 2], dtype=np.int64)
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
generated_inputs["tf.raw_ops.EditDistance"] = tf_raw_ops_editdistance_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.EditDistance' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EditDistance'.")

check_valid('tf.raw_ops.EditDistance', generated_inputs['tf.raw_ops.EditDistance'], lib="tf", suffix=0)
