
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_counter_inputs():
    list_of_inputs = []

    # Input 1
    start = np.int64(0)
    step = np.int64(1)
    dtype = tf.int64
    input_dict = {"start": start, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    start = np.int32(5)
    step = np.int32(2)
    dtype = tf.int32
    input_dict = {"start": start, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    start = np.int64(-5)
    step = np.int64(-1)
    dtype = tf.int64
    input_dict = {"start": start, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    start = np.int32(10)
    step = np.int32(-2)
    dtype = tf.int32
    input_dict = {"start": start, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    start = np.int64(0)
    step = np.int64(5)
    dtype = tf.int64
    input_dict = {"start": start, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    start = np.int32(-10)
    step = np.int32(1)
    dtype = tf.int32
    input_dict = {"start": start, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    start = np.int64(100)
    step = np.int64(10)
    dtype = tf.int64
    input_dict = {"start": start, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    start = np.int32(20)
    step = np.int32(3)
    dtype = tf.int32
    input_dict = {"start": start, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    start = np.int64(-20)
    step = np.int64(-3)
    dtype = tf.int64
    input_dict = {"start": start, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    start = np.int32(1)
    step = np.int32(100)
    dtype = tf.int32
    input_dict = {"start": start, "step": step, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.Counter"] = tf_data_experimental_counter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.Counter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.Counter'.")

check_valid('tf.data.experimental.Counter', generated_inputs['tf.data.experimental.Counter'], lib="tf", suffix=0)
