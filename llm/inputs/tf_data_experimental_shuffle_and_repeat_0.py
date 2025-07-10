
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_shuffle_and_repeat_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    buffer_size = np.array(10, dtype=np.int64)
    count = np.array(2, dtype=np.int64)
    seed = np.array(42, dtype=np.int64)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Infinite repeat
    buffer_size = np.array(5, dtype=np.int64)
    count = None
    seed = np.array(123, dtype=np.int64)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero count
    buffer_size = np.array(100, dtype=np.int64)
    count = np.array(0, dtype=np.int64)
    seed = np.array(7, dtype=np.int64)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Large buffer size
    buffer_size = np.array(1000, dtype=np.int64)
    count = np.array(1, dtype=np.int64)
    seed = np.array(99, dtype=np.int64)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small buffer size
    buffer_size = np.array(2, dtype=np.int64)
    count = np.array(3, dtype=np.int64)
    seed = np.array(1, dtype=np.int64)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative count
    buffer_size = np.array(15, dtype=np.int64)
    count = np.array(-1, dtype=np.int64)
    seed = np.array(55, dtype=np.int64)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different seed
    buffer_size = np.array(20, dtype=np.int64)
    count = np.array(2, dtype=np.int64)
    seed = np.array(1000, dtype=np.int64)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large count
    buffer_size = np.array(50, dtype=np.int64)
    count = np.array(10, dtype=np.int64)
    seed = np.array(2023, dtype=np.int64)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: buffer_size is a tensor with shape [1]
    buffer_size = np.array([25], dtype=np.int64)
    count = np.array(5, dtype=np.int64)
    seed = np.array(4, dtype=np.int64)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: seed is a tensor with shape [1]
    buffer_size = np.array(30, dtype=np.int64)
    count = np.array(3, dtype=np.int64)
    seed = np.array([6], dtype=np.int64)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.shuffle_and_repeat"] = tf_data_experimental_shuffle_and_repeat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.shuffle_and_repeat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.shuffle_and_repeat'.")

check_valid('tf.data.experimental.shuffle_and_repeat', generated_inputs['tf.data.experimental.shuffle_and_repeat'], lib="tf", suffix=0)
