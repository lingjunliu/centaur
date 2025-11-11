
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_bucketize_inputs():
    list_of_inputs = []

    input1 = np.array([-5, 10000], dtype=np.int32)
    boundaries1 = [0, 10, 100]
    input_dict1 = {'name': 'bucketize1', 'input': input1, 'boundaries': boundaries1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([150, 10], dtype=np.int64)
    boundaries2 = [0, 10, 100, 200]
    input_dict2 = {'name': 'bucketize2', 'input': input2, 'boundaries': boundaries2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([5, 100], dtype=np.float32)
    boundaries3 = [0, 10, 100]
    input_dict3 = {'name': 'bucketize3', 'input': input3, 'boundaries': boundaries3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([-1.5, 2.7, 5.1], dtype=np.float64)
    boundaries4 = [-2, 0, 3, 6]
    input_dict4 = {'name': 'bucketize4', 'input': input4, 'boundaries': boundaries4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    boundaries5 = [0, 2, 4]
    input_dict5 = {'name': 'bucketize5', 'input': input5, 'boundaries': boundaries5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    boundaries6 = [0, 3, 6]
    input_dict6 = {'name': 'bucketize6', 'input': input6, 'boundaries': boundaries6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([10.5, 20.2, 30.8], dtype=np.float32)
    boundaries7 = [5.0, 15.0, 25.0, 35.0]
    input_dict7 = {'name': 'bucketize7', 'input': input7, 'boundaries': boundaries7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    boundaries8 = [-8, -4, 0, 4, 8]
    input_dict8 = {'name': 'bucketize8', 'input': input8, 'boundaries': boundaries8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[1.1, 2.2], [-3.3, 4.4]], dtype=np.float64)
    boundaries9 = [-2.0, 0.0, 2.0]
    input_dict9 = {'name': 'bucketize9', 'input': input9, 'boundaries': boundaries9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([100, 200, 300], dtype=np.int64)
    boundaries10 = [50, 150, 250]
    input_dict10 = {'name': 'bucketize10', 'input': input10, 'boundaries': boundaries10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.raw_ops.Bucketize"] = tf_raw_ops_bucketize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Bucketize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Bucketize'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Bucketize', generated_inputs['tf.raw_ops.Bucketize'], lib="tf", suffix=0)
