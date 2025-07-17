
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_multinomial_inputs():
    list_of_inputs = []

    # Input 1
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    num_samples = np.array(2, dtype=np.int32)
    seed = 123
    seed2 = 456
    output_dtype = tf.int64
    name = "multinomial_1"
    input_dict = {"seed": seed, "seed2": seed2, "output_dtype": output_dtype, "name": name, "logits": logits, "num_samples": num_samples}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    logits = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]], dtype=np.float64)
    num_samples = np.array(5, dtype=np.int32)
    seed = 1
    seed2 = 2
    output_dtype = tf.int32
    name = "multinomial_2"
    input_dict = {"seed": seed, "seed2": seed2, "output_dtype": output_dtype, "name": name, "logits": logits, "num_samples": num_samples}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    logits = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    num_samples = np.array(3, dtype=np.int32)
    seed = 789
    seed2 = 101
    output_dtype = tf.int64
    name = "multinomial_3"
    input_dict = {"seed": seed, "seed2": seed2, "output_dtype": output_dtype, "name": name, "logits": logits, "num_samples": num_samples}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    logits = np.array([[10.0, 5.0, 1.0]], dtype=np.float32)
    num_samples = np.array(10, dtype=np.int32)
    seed = 112
    seed2 = 131
    output_dtype = tf.int32
    name = "multinomial_4"
    input_dict = {"seed": seed, "seed2": seed2, "output_dtype": output_dtype, "name": name, "logits": logits, "num_samples": num_samples}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    logits = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]], dtype=np.float64)
    num_samples = np.array(1, dtype=np.int32)
    seed = 3
    seed2 = 4
    output_dtype = tf.int64
    name = "multinomial_5"
    input_dict = {"seed": seed, "seed2": seed2, "output_dtype": output_dtype, "name": name, "logits": logits, "num_samples": num_samples}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    logits = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    num_samples = np.array(4, dtype=np.int32)
    seed = 456
    seed2 = 789
    output_dtype = tf.int32
    name = "multinomial_6"
    input_dict = {"seed": seed, "seed2": seed2, "output_dtype": output_dtype, "name": name, "logits": logits, "num_samples": num_samples}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    logits = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    num_samples = np.array(7, dtype=np.int32)
    seed = 5
    seed2 = 6
    output_dtype = tf.int64
    name = "multinomial_7"
    input_dict = {"seed": seed, "seed2": seed2, "output_dtype": output_dtype, "name": name, "logits": logits, "num_samples": num_samples}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8
    logits = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    num_samples = np.array(3, dtype=np.int32)
    seed = 131
    seed2 = 141
    output_dtype = tf.int32
    name = "multinomial_8"
    input_dict = {"seed": seed, "seed2": seed2, "output_dtype": output_dtype, "name": name, "logits": logits, "num_samples": num_samples}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    logits = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float64)
    num_samples = np.array(6, dtype=np.int32)
    seed = 7
    seed2 = 8
    output_dtype = tf.int64
    name = "multinomial_9"
    input_dict = {"seed": seed, "seed2": seed2, "output_dtype": output_dtype, "name": name, "logits": logits, "num_samples": num_samples}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    logits = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]], dtype=np.float32)
    num_samples = np.array(2, dtype=np.int32)
    seed = 171
    seed2 = 181
    output_dtype = tf.int32
    name = "multinomial_10"
    input_dict = {"seed": seed, "seed2": seed2, "output_dtype": output_dtype, "name": name, "logits": logits, "num_samples": num_samples}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Multinomial"] = tf_raw_ops_multinomial_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Multinomial' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Multinomial'.")

check_valid('tf.raw_ops.Multinomial', generated_inputs['tf.raw_ops.Multinomial'], lib="tf", suffix=0)
