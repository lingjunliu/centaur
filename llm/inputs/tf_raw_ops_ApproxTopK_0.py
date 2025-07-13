
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_approx_top_k_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    k = 3
    reduction_dimension = -1
    recall_target = 0.9
    is_max_k = True
    reduction_input_size_override = -1
    aggregate_to_topk = True

    input_dict = {
        "input": input_tensor,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "is_max_k": is_max_k,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    k = 2
    reduction_dimension = 0
    recall_target = 0.95
    is_max_k = False
    reduction_input_size_override = 5
    aggregate_to_topk = False

    input_dict = {
        "input": input_tensor,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "is_max_k": is_max_k,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": "approx_top_k_min"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    k = 1
    reduction_dimension = 0
    recall_target = 0.8
    is_max_k = True
    reduction_input_size_override = -1
    aggregate_to_topk = True

    input_dict = {
        "input": input_tensor,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "is_max_k": is_max_k,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    k = 2
    reduction_dimension = 1
    recall_target = 0.7
    is_max_k = False
    reduction_input_size_override = -1
    aggregate_to_topk = False

    input_dict = {
        "input": input_tensor,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "is_max_k": is_max_k,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": "approx_top_k_max"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    k = 1
    reduction_dimension = 0
    recall_target = 0.99
    is_max_k = True
    reduction_input_size_override = 10
    aggregate_to_topk = True

    input_dict = {
        "input": input_tensor,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "is_max_k": is_max_k,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    k = 0
    reduction_dimension = 0
    recall_target = 0.99
    is_max_k = True
    reduction_input_size_override = 10
    aggregate_to_topk = True

    input_dict = {
        "input": input_tensor,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "is_max_k": is_max_k,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32)
    k = 3
    reduction_dimension = -1
    recall_target = 0.9
    is_max_k = True
    reduction_input_size_override = -1
    aggregate_to_topk = True

    input_dict = {
        "input": input_tensor,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "is_max_k": is_max_k,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    input_tensor = np.array([[1.0, 2.0,3.0], [4.0, 5.0,6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    k = 2
    reduction_dimension = 1
    recall_target = 0.95
    is_max_k = False
    reduction_input_size_override = -1
    aggregate_to_topk = False

    input_dict = {
        "input": input_tensor,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "is_max_k": is_max_k,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": "approx_top_k_min"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[1.0, 2.0,3.0], [4.0, 5.0,6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    k = 2
    reduction_dimension = 0
    recall_target = 0.95
    is_max_k = True
    reduction_input_size_override = -1
    aggregate_to_topk = True

    input_dict = {
        "input": input_tensor,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "is_max_k": is_max_k,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    k = 3
    reduction_dimension = -1
    recall_target = 0.5
    is_max_k = True
    reduction_input_size_override = -1
    aggregate_to_topk = True

    input_dict = {
        "input": input_tensor,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "is_max_k": is_max_k,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.ApproxTopK"] = tf_raw_ops_approx_top_k_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.ApproxTopK' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ApproxTopK'.")

check_valid('tf.raw_ops.ApproxTopK', generated_inputs['tf.raw_ops.ApproxTopK'], lib="tf", suffix=0)
