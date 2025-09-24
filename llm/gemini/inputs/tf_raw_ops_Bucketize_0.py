
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_bucketize_inputs():
    list_of_inputs = []

    # Input 1: Basic case with integers and boundaries
    input_np = np.array([-5, 5, 15, 25, 35], dtype=np.int32)
    boundaries = [0.0, 10.0, 20.0, 30.0]
    input_dict = {"input": input_np, "boundaries": boundaries, "name": "bucketize_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float input and boundaries
    input_np = np.array([-2.5, 2.5, 7.5, 12.5, 17.5], dtype=np.float32)
    boundaries = [0.0, 5.0, 10.0, 15.0]
    input_dict = {"input": input_np, "boundaries": boundaries, "name": "bucketize_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative boundaries
    input_np = np.array([-15, -5, 5, 15], dtype=np.int64)
    boundaries = [-10.0, 0.0, 10.0]
    input_dict = {"input": input_np, "boundaries": boundaries, "name": "bucketize_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D input
    input_np = np.array([[-5, 5], [15, 25]], dtype=np.float64)
    boundaries = [0.0, 10.0, 20.0]
    input_dict = {"input": input_np, "boundaries": boundaries, "name": "bucketize_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty boundaries list
    input_np = np.array([1, 2, 3], dtype=np.int32)
    boundaries = []
    input_dict = {"input": input_np, "boundaries": boundaries, "name": "bucketize_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All values greater than largest boundary
    input_np = np.array([10, 20, 30], dtype=np.int32)
    boundaries = [1.0, 5.0]
    input_dict = {"input": input_np, "boundaries": boundaries, "name": "bucketize_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: All values less than smallest boundary
    input_np = np.array([-10, -20, -30], dtype=np.float32)
    boundaries = [-5.0, -1.0]
    input_dict = {"input": input_np, "boundaries": boundaries, "name": "bucketize_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: One boundary value
    input_np = np.array([-1, 0, 1], dtype=np.int64)
    boundaries = [0.0]
    input_dict = {"input": input_np, "boundaries": boundaries, "name": "bucketize_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Repeated boundary values
    input_np = np.array([4, 5, 6], dtype=np.float64)
    boundaries = [4.0, 4.0, 5.0]
    input_dict = {"input": input_np, "boundaries": boundaries, "name": "bucketize_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D input with negative and positive values
    input_np = np.array([[[ -1, 1], [ 2, -2]], [[-3, 3], [4, -4]]], dtype=np.int32)
    boundaries = [-2.0, 0.0, 2.0]
    input_dict = {"input": input_np, "boundaries": boundaries, "name": "bucketize_10"}
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
