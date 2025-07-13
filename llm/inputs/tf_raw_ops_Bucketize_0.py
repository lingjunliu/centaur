
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_bucketize_inputs():
    list_of_inputs = []

    # Input 1: Basic test
    input_val = np.array([-5, 5, 15, 105]).astype(np.int32)
    boundaries_val = [0, 10, 100]
    input_dict = {"input": input_val, "boundaries": boundaries_val, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float input
    input_val = np.array([-5.0, 5.0, 15.0, 105.0]).astype(np.float32)
    boundaries_val = [0.0, 10.0, 100.0]
    input_dict = {"input": input_val, "boundaries": boundaries_val, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D input
    input_val = np.array([[-5, 5], [15, 105]]).astype(np.int64)
    boundaries_val = [0, 10, 100]
    input_dict = {"input": input_val, "boundaries": boundaries_val, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty boundaries
    input_val = np.array([1, 2, 3]).astype(np.int32)
    boundaries_val = []
    input_dict = {"input": input_val, "boundaries": boundaries_val, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: One boundary
    input_val = np.array([1, 2, 3]).astype(np.float64)
    boundaries_val = [2]
    input_dict = {"input": input_val, "boundaries": boundaries_val, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All negative input
    input_val = np.array([-1, -2, -3]).astype(np.int32)
    boundaries_val = [-4, -2]
    input_dict = {"input": input_val, "boundaries": boundaries_val, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Identical boundaries
    input_val = np.array([1, 2, 3]).astype(np.int64)
    boundaries_val = [2, 2]
    input_dict = {"input": input_val, "boundaries": boundaries_val, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D input
    input_val = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.float32)
    boundaries_val = [2, 5, 7]
    input_dict = {"input": input_val, "boundaries": boundaries_val, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large boundaries
    input_val = np.array([1000, 2000, 3000]).astype(np.int32)
    boundaries_val = [500, 1500, 2500]
    input_dict = {"input": input_val, "boundaries": boundaries_val, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Same value in the input
    input_val = np.array([5, 5, 5]).astype(np.int32)
    boundaries_val = [0, 5, 10]
    input_dict = {"input": input_val, "boundaries": boundaries_val, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Bucketize"] = tf_raw_ops_bucketize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Bucketize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Bucketize'.")

check_valid('tf.raw_ops.Bucketize', generated_inputs['tf.raw_ops.Bucketize'], lib="tf", suffix=0)
