
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_polygamma_inputs():
    list_of_inputs = []
    
    a = np.array(0.0, dtype=np.float32)
    x = np.array(1.0, dtype=np.float32)
    name = "polygamma_0"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([1.0], dtype=np.float32)
    x = np.array([2.5], dtype=np.float32)
    name = "polygamma_1"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    name = "polygamma_2"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float32)
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    name = "polygamma_3"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    x = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    name = "polygamma_large"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array(4.0, dtype=np.float32)
    x = np.array(0.5, dtype=np.float32)
    name = "polygamma_order4"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.ones((2, 2, 2), dtype=np.float32)
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    name = "polygamma_3d"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([5.0, 5.0], dtype=np.float64)
    x = np.array([2.0, 3.0], dtype=np.float64)
    name = "polygamma_order5"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.polygamma"] = tf_math_polygamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.polygamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.polygamma'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.polygamma', generated_inputs['tf.math.polygamma'], lib="tf", suffix=0)
