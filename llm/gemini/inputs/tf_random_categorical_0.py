
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_categorical_inputs():
    list_of_inputs = []

    # Input 1
    logits = np.array([[0.1, 0.9]], dtype=np.float32)
    num_samples = np.int32(5)
    dtype = tf.int32
    seed = np.int32(123)
    name = "categorical_sample_1"
    input_dict = {"logits": logits, "num_samples": num_samples, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    logits = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32)
    num_samples = np.int32(1)
    dtype = tf.int64
    seed = np.int32(456)
    name = "categorical_sample_2"
    input_dict = {"logits": logits, "num_samples": num_samples, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    logits = np.array([[0.2, 0.3, 0.5]], dtype=np.float64)
    num_samples = np.int32(10)
    dtype = tf.int32
    seed = np.int32(789)
    name = "categorical_sample_3"
    input_dict = {"logits": logits, "num_samples": num_samples, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    logits = np.array([[-0.1, 0.9], [1.2, -0.8]], dtype=np.float32)
    num_samples = np.int32(3)
    dtype = tf.int64
    seed = np.int32(101)
    name = "categorical_sample_4"
    input_dict = {"logits": logits, "num_samples": num_samples, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    logits = np.array([[10.0, 0.0, -10.0]], dtype=np.float32)
    num_samples = np.int32(7)
    dtype = tf.int32
    seed = np.int32(202)
    name = "categorical_sample_5"
    input_dict = {"logits": logits, "num_samples": num_samples, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    logits = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    num_samples = np.int32(2)
    dtype = tf.int64
    seed = np.int32(303)
    name = "categorical_sample_6"
    input_dict = {"logits": logits, "num_samples": num_samples, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    logits = np.array([[0.6, 0.4], [0.4, 0.6], [0.5, 0.5]], dtype=np.float32)
    num_samples = np.int32(4)
    dtype = tf.int32
    seed = np.int32(404)
    name = "categorical_sample_7"
    input_dict = {"logits": logits, "num_samples": num_samples, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    logits = np.array([[-1.0, 2.0, 0.0], [3.0, -2.0, 1.0]], dtype=np.float32)
    num_samples = np.int32(6)
    dtype = tf.int64
    seed = np.int32(505)
    name = "categorical_sample_8"
    input_dict = {"logits": logits, "num_samples": num_samples, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    logits = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]], dtype=np.float32)
    num_samples = np.int32(8)
    dtype = tf.int32
    seed = np.int32(606)
    name = "categorical_sample_9"
    input_dict = {"logits": logits, "num_samples": num_samples, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    logits = np.array([[-5.0, 5.0], [5.0, -5.0]], dtype=np.float32)
    num_samples = np.int32(9)
    dtype = tf.int64
    seed = np.int32(707)
    name = "categorical_sample_10"
    input_dict = {"logits": logits, "num_samples": num_samples, "dtype": dtype, "seed": seed, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.random.categorical"] = tf_random_categorical_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.random.categorical' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.categorical'.")

check_valid('tf.random.categorical', generated_inputs['tf.random.categorical'], lib="tf", suffix=0)
