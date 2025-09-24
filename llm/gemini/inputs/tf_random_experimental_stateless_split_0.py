
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_experimental_stateless_split_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    seed = np.array([1, 2], dtype=np.int32)
    num = 2
    alg = 'auto_select'
    input_dict = {'seed': seed, 'num': num, 'alg': alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different num value
    seed = np.array([3, 4], dtype=np.int32)
    num = 5
    alg = 'auto_select'
    input_dict = {'seed': seed, 'num': num, 'alg': alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different seed value and alg
    seed = np.array([5, 6], dtype=np.int32)
    num = 3
    alg = 'philox'
    input_dict = {'seed': seed, 'num': num, 'alg': alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64 seed
    seed = np.array([7, 8], dtype=np.int64)
    num = 2
    alg = 'auto_select'
    input_dict = {'seed': seed, 'num': num, 'alg': alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger seed values (within int32 range)
    seed = np.array([2**15, 2**16], dtype=np.int32)
    num = 4
    alg = 'auto_select'
    input_dict = {'seed': seed, 'num': num, 'alg': alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small seed values
    seed = np.array([-1, -2], dtype=np.int32)
    num = 3
    alg = 'auto_select'
    input_dict = {'seed': seed, 'num': num, 'alg': alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Zero seed
    seed = np.array([0, 0], dtype=np.int32)
    num = 2
    alg = 'auto_select'
    input_dict = {'seed': seed, 'num': num, 'alg': alg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.experimental.stateless_split"] = tf_random_experimental_stateless_split_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.experimental.stateless_split' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.experimental.stateless_split'.")

check_valid('tf.random.experimental.stateless_split', generated_inputs['tf.random.experimental.stateless_split'], lib="tf", suffix=0)
