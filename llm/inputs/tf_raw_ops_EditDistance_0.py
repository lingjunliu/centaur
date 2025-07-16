
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
    name = None

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": hypothesis_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    hypothesis_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    hypothesis_values = np.array([5, 6], dtype=np.int64)
    hypothesis_shape = np.array([1, 2], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
    truth_values = np.array([5, 7, 6], dtype=np.int64)
    truth_shape = np.array([1, 3], dtype=np.int64)
    normalize = False
    name = "edit_distance_op"

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": hypothesis_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    hypothesis_indices = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int64)
    hypothesis_values = np.array([7, 8], dtype=np.int64)
    hypothesis_shape = np.array([1, 1, 2], dtype=np.int64)
    truth_indices = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=np.int64)
    truth_values = np.array([7, 9, 8], dtype=np.int64)
    truth_shape = np.array([1, 1, 3], dtype=np.int64)
    normalize = True
    name = None

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": hypothesis_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    hypothesis_indices = np.array([[0, 0]], dtype=np.int64)
    hypothesis_values = np.array([10], dtype=np.int64)
    hypothesis_shape = np.array([1, 1], dtype=np.int64)
    truth_indices = np.array([[0, 0]], dtype=np.int64)
    truth_values = np.array([10], dtype=np.int64)
    truth_shape = np.array([1, 1], dtype=np.int64)
    normalize = True
    name = None

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": hypothesis_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    hypothesis_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    hypothesis_values = np.array([11, 12], dtype=np.int64)
    hypothesis_shape = np.array([1, 2], dtype=np.int64)
    truth_indices = np.array([[0, 0]], dtype=np.int64)
    truth_values = np.array([11], dtype=np.int64)
    truth_shape = np.array([1, 1], dtype=np.int64)
    normalize = False
    name = None

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": hypothesis_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    hypothesis_indices = np.array([[0, 0], [1, 0]], dtype=np.int64)
    hypothesis_values = np.array([13, 14], dtype=np.int64)
    hypothesis_shape = np.array([2, 1], dtype=np.int64)
    truth_indices = np.array([[0, 0], [1, 0]], dtype=np.int64)
    truth_values = np.array([13, 15], dtype=np.int64)
    truth_shape = np.array([2, 1], dtype=np.int64)
    normalize = True
    name = None

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": hypothesis_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    hypothesis_indices = np.array([[0, 0]], dtype=np.int64)
    hypothesis_values = np.array([16], dtype=np.int64)
    hypothesis_shape = np.array([1, 1], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0,1]], dtype=np.int64)
    truth_values = np.array([16, 17], dtype=np.int64)
    truth_shape = np.array([1, 2], dtype=np.int64)
    normalize = False
    name = "another_edit_distance"

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": hypothesis_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    hypothesis_indices = np.array([[0, 0, 0]], dtype=np.int64)
    hypothesis_values = np.array([18], dtype=np.int64)
    hypothesis_shape = np.array([1, 1, 1], dtype=np.int64)
    truth_indices = np.array([[0, 0, 0]], dtype=np.int64)
    truth_values = np.array([18], dtype=np.int64)
    truth_shape = np.array([1, 1, 1], dtype=np.int64)
    normalize = True
    name = None

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": hypothesis_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    hypothesis_indices = np.array([[0, 0], [0,1]], dtype=np.int64)
    hypothesis_values = np.array([19, 20], dtype=np.int64)
    hypothesis_shape = np.array([1, 2], dtype=np.int64)
    truth_indices = np.array([[0, 0], [0,1], [0,2]], dtype=np.int64)
    truth_values = np.array([19, 20, 21], dtype=np.int64)
    truth_shape = np.array([1, 3], dtype=np.int64)
    normalize = True
    name = None

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": hypothesis_values,
        "truth_shape": truth_shape,
        "normalize": normalize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    hypothesis_indices = np.array([[0, 0], [1, 0]], dtype=np.int64)
    hypothesis_values = np.array([22, 23], dtype=np.int64)
    hypothesis_shape = np.array([2, 1], dtype=np.int64)
    truth_indices = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.int64)
    truth_values = np.array([22, 24, 23], dtype=np.int64)
    truth_shape = np.array([3, 1], dtype=np.int64)
    normalize = False
    name = None

    input_dict = {
        "hypothesis_indices": hypothesis_indices,
        "hypothesis_values": hypothesis_values,
        "hypothesis_shape": hypothesis_shape,
        "truth_indices": truth_indices,
        "truth_values": hypothesis_values,
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
        "args": [],
        "kwargs": {k: tf.convert_to_tensor(v, dtype=tf.int64) if k.endswith(('indices', 'values', 'shape')) and not isinstance(v, bool) else v for k, v in input_dict.items()}
    })

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.EditDistance' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EditDistance'.")

check_valid('tf.raw_ops.EditDistance', generated_inputs['tf.raw_ops.EditDistance'], lib="tf", suffix=0)
