
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_shuffle_and_repeat_inputs():
    list_of_inputs = []

    # Input 1: Basic case with count and seed
    buffer_size = np.int64(10)
    count = np.int64(2)
    seed = np.int64(42)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(input_dict)

    # Input 2: No count (infinite repeat)
    buffer_size = np.int64(5)
    count = None
    seed = np.int64(123)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(input_dict)

    # Input 3: Zero buffer size (should still work)
    buffer_size = np.int64(0)
    count = np.int64(3)
    seed = np.int64(0)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(input_dict)

    # Input 4: Large buffer size
    buffer_size = np.int64(1000)
    count = np.int64(1)
    seed = np.int64(999)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(input_dict)

    # Input 5: Negative count (treated as None/infinite)
    buffer_size = np.int64(7)
    count = np.int64(-1)
    seed = np.int64(55)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(input_dict)

    # Input 6: Different seed value
    buffer_size = np.int64(8)
    count = np.int64(2)
    seed = np.int64(678)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(input_dict)

     # Input 7: Very small buffer
    buffer_size = np.int64(1)
    count = np.int64(2)
    seed = np.int64(101)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(input_dict)

    # Input 8: Another seed
    buffer_size = np.int64(3)
    count = np.int64(4)
    seed = np.int64(222)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(input_dict)

    # Input 9: Edge case with 1 repeat
    buffer_size = np.int64(6)
    count = np.int64(1)
    seed = np.int64(777)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(input_dict)

    # Input 10: Larger count
    buffer_size = np.int64(9)
    count = np.int64(5)
    seed = np.int64(444)
    input_dict = {"buffer_size": buffer_size, "count": count, "seed": seed}
    list_of_inputs.append(input_dict)

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
