
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_shuffle_and_repeat_inputs():
    list_of_inputs = []

    # Input 1: Basic positive values
    buffer_size = np.int64(10)
    count = np.int64(2)
    seed = np.int64(42)

    input_dict = {
        "buffer_size": tf.constant(buffer_size),
        "count": tf.constant(count),
        "seed": tf.constant(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Indefinite repeat (count=None represented by -1)
    buffer_size = np.int64(5)
    count = np.int64(-1)
    seed = np.int64(123)

    input_dict = {
        "buffer_size": tf.constant(buffer_size),
        "count": tf.constant(count),
        "seed": tf.constant(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero buffer size (should still be valid)
    buffer_size = np.int64(0)
    count = np.int64(3)
    seed = np.int64(0)

    input_dict = {
        "buffer_size": tf.constant(buffer_size),
        "count": tf.constant(count),
        "seed": tf.constant(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large buffer size and count
    buffer_size = np.int64(1000)
    count = np.int64(10)
    seed = np.int64(999)

    input_dict = {
        "buffer_size": tf.constant(buffer_size),
        "count": tf.constant(count),
        "seed": tf.constant(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small buffer size and count
    buffer_size = np.int64(2)
    count = np.int64(1)
    seed = np.int64(1)

    input_dict = {
        "buffer_size": tf.constant(buffer_size),
        "count": tf.constant(count),
        "seed": tf.constant(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Another seed
    buffer_size = np.int64(15)
    count = np.int64(4)
    seed = np.int64(666)

    input_dict = {
        "buffer_size": tf.constant(buffer_size),
        "count": tf.constant(count),
        "seed": tf.constant(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large seed value
    buffer_size = np.int64(7)
    count = np.int64(2)
    seed = np.int64(2**31 - 1)

    input_dict = {
        "buffer_size": tf.constant(buffer_size),
        "count": tf.constant(count),
        "seed": tf.constant(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different buffer_size
    buffer_size = np.int64(30)
    count = np.int64(5)
    seed = np.int64(100)

    input_dict = {
        "buffer_size": tf.constant(buffer_size),
        "count": tf.constant(count),
        "seed": tf.constant(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Zero count
    buffer_size = np.int64(10)
    count = np.int64(0)
    seed = np.int64(50)

    input_dict = {
        "buffer_size": tf.constant(buffer_size),
        "count": tf.constant(count),
        "seed": tf.constant(seed)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All small values
    buffer_size = np.int64(1)
    count = np.int64(1)
    seed = np.int64(1)

    input_dict = {
        "buffer_size": tf.constant(buffer_size),
        "count": tf.constant(count),
        "seed": tf.constant(seed)
    }
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
