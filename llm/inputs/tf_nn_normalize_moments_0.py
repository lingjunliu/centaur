
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_normalize_moments_inputs():
    list_of_inputs = []

    # Input 1
    counts = np.array(10.0, dtype=np.float32)
    mean_ss = np.array(50.0, dtype=np.float32)
    variance_ss = np.array(300.0, dtype=np.float32)
    shift = np.array(0.0, dtype=np.float32)
    name = "moments1"
    input_dict = {"counts": tf.convert_to_tensor(counts), "mean_ss": tf.convert_to_tensor(mean_ss), "variance_ss": tf.convert_to_tensor(variance_ss), "shift": tf.convert_to_tensor(shift), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    counts = np.array(5.0, dtype=np.float32)
    mean_ss = np.array(20.0, dtype=np.float32)
    variance_ss = np.array(100.0, dtype=np.float32)
    shift = np.array(2.0, dtype=np.float32)
    name = "moments2"
    input_dict = {"counts": tf.convert_to_tensor(counts), "mean_ss": tf.convert_to_tensor(mean_ss), "variance_ss": tf.convert_to_tensor(variance_ss), "shift": tf.convert_to_tensor(shift), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    counts = np.array(100.0, dtype=np.float32)
    mean_ss = np.array(450.0, dtype=np.float32)
    variance_ss = np.array(2100.0, dtype=np.float32)
    shift = np.array(-1.0, dtype=np.float32)
    name = "moments3"
    input_dict = {"counts": tf.convert_to_tensor(counts), "mean_ss": tf.convert_to_tensor(mean_ss), "variance_ss": tf.convert_to_tensor(variance_ss), "shift": tf.convert_to_tensor(shift), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    counts = np.array(2.0, dtype=np.float32)
    mean_ss = np.array(8.0, dtype=np.float32)
    variance_ss = np.array(30.0, dtype=np.float32)
    shift = np.array(-2.0, dtype=np.float32)
    name = "moments4"
    input_dict = {"counts": tf.convert_to_tensor(counts), "mean_ss": tf.convert_to_tensor(mean_ss), "variance_ss": tf.convert_to_tensor(variance_ss), "shift": tf.convert_to_tensor(shift), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    counts = np.array(15.0, dtype=np.float32)
    mean_ss = np.array(70.0, dtype=np.float32)
    variance_ss = np.array(400.0, dtype=np.float32)
    shift = np.array(1.0, dtype=np.float32)
    name = "moments5"
    input_dict = {"counts": tf.convert_to_tensor(counts), "mean_ss": tf.convert_to_tensor(mean_ss), "variance_ss": tf.convert_to_tensor(variance_ss), "shift": tf.convert_to_tensor(shift), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    counts = np.array(1.0, dtype=np.float32)
    mean_ss = np.array(3.0, dtype=np.float32)
    variance_ss = np.array(10.0, dtype=np.float32)
    shift = np.array(0.5, dtype=np.float32)
    name = "moments6"
    input_dict = {"counts": tf.convert_to_tensor(counts), "mean_ss": tf.convert_to_tensor(mean_ss), "variance_ss": tf.convert_to_tensor(variance_ss), "shift": tf.convert_to_tensor(shift), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    counts = np.array(20.0, dtype=np.float32)
    mean_ss = np.array(90.0, dtype=np.float32)
    variance_ss = np.array(500.0, dtype=np.float32)
    shift = np.array(-0.5, dtype=np.float32)
    name = "moments7"
    input_dict = {"counts": tf.convert_to_tensor(counts), "mean_ss": tf.convert_to_tensor(mean_ss), "variance_ss": tf.convert_to_tensor(variance_ss), "shift": tf.convert_to_tensor(shift), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    counts = np.array(30.0, dtype=np.float32)
    mean_ss = np.array(110.0, dtype=np.float32)
    variance_ss = np.array(600.0, dtype=np.float32)
    shift = np.array(0.0, dtype=np.float32)
    name = "moments8"
    input_dict = {"counts": tf.convert_to_tensor(counts), "mean_ss": tf.convert_to_tensor(mean_ss), "variance_ss": tf.convert_to_tensor(variance_ss), "shift": tf.convert_to_tensor(shift), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    counts = np.array(7.0, dtype=np.float32)
    mean_ss = np.array(30.0, dtype=np.float32)
    variance_ss = np.array(150.0, dtype=np.float32)
    shift = np.array(3.0, dtype=np.float32)
    name = "moments9"
    input_dict = {"counts": tf.convert_to_tensor(counts), "mean_ss": tf.convert_to_tensor(mean_ss), "variance_ss": tf.convert_to_tensor(variance_ss), "shift": tf.convert_to_tensor(shift), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    counts = np.array(3.0, dtype=np.float32)
    mean_ss = np.array(10.0, dtype=np.float32)
    variance_ss = np.array(40.0, dtype=np.float32)
    shift = np.array(-3.0, dtype=np.float32)
    name = "moments10"
    input_dict = {"counts": tf.convert_to_tensor(counts), "mean_ss": tf.convert_to_tensor(mean_ss), "variance_ss": tf.convert_to_tensor(variance_ss), "shift": tf.convert_to_tensor(shift), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.normalize_moments"] = tf_nn_normalize_moments_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.normalize_moments' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.normalize_moments'.")

check_valid('tf.nn.normalize_moments', generated_inputs['tf.nn.normalize_moments'], lib="tf", suffix=0)
