
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_xdivy_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array(1.0, dtype=np.float32)
    y = np.array(2.0, dtype=np.float32)
    name = "div1"
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "y": tf.convert_to_tensor(y).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array(0.0, dtype=np.float32)
    y = np.array(1.0, dtype=np.float32)
    name = "div2"
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "y": tf.convert_to_tensor(y).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array(0.0, dtype=np.float32)
    y = np.array(0.0, dtype=np.float32)
    name = "div3"
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "y": tf.convert_to_tensor(y).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array(1.0, dtype=np.float32)
    y = np.array(0.0, dtype=np.float32)
    name = "div4"
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "y": tf.convert_to_tensor(y).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 0.0], dtype=np.float32)
    name = "div5"
    input_dict = {"x": tf.convert_to_tensor(x).numpy(), "y": tf.convert_to_tensor(y).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.xdivy"] = tf_math_xdivy_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.xdivy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.xdivy'.")

check_valid('tf.math.xdivy', generated_inputs['tf.math.xdivy'], lib="tf", suffix=0)
