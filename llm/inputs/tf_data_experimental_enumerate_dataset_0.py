
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_enumerate_dataset_inputs():
    list_of_inputs = []

    # Input 1: Scalar Tensor
    start = tf.constant(0, dtype=tf.int64)
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar Tensor - different value
    start = tf.constant(10, dtype=tf.int64)
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar Tensor - negative value
    start = tf.constant(-5, dtype=tf.int64)
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rank 0 Tensor with a different dtype
    start = tf.constant(2, dtype=tf.int32)
    start = tf.cast(start, tf.int64) #Explicit cast to int64 to adhere to signature
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Rank 1 Tensor, but should still work since only scalar tensors are expected
    start_np = np.array(5, dtype=np.int64)
    start = tf.convert_to_tensor(start_np, dtype=tf.int64)
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Rank 1 Tensor, with different value
    start_np = np.array(-10, dtype=np.int64)
    start = tf.convert_to_tensor(start_np, dtype=tf.int64)
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: tf.Variable
    start = tf.Variable(5, dtype=tf.int64)
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: very large number
    start = tf.constant(2**8, dtype=tf.int64) # reduced the exponent further
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: zero
    start = tf.constant(0, dtype=tf.int64)
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Very small negative number
    start = tf.constant(-(2**8), dtype=tf.int64) # reduced the exponent further
    input_dict = {"start": start}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.enumerate_dataset"] = tf_data_experimental_enumerate_dataset_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.enumerate_dataset' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.enumerate_dataset'.")

check_valid('tf.data.experimental.enumerate_dataset', generated_inputs['tf.data.experimental.enumerate_dataset'], lib="tf", suffix=0)
