
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_math_floor_inputs():
    list_of_inputs = []
    
    x = np.array([1.3324, 2.7, 3.14159], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-1.5, -2.532, -0.99], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.3324, -1.5, 5.555, -2.532, 0.99], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1.2, 3.4], [5.6, 7.8]], dtype=np.float32)
    input_dict = {"x": x, "name": "floor_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([float("inf"), float("-inf"), 1.5], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([9.999], dtype=np.float64)
    input_dict = {"x": x, "name": "single_floor"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[-10.1, -20.2, -30.3], [-40.4, -50.5, -60.6]], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0001, 1.9999, 2.00001, -1.0001], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.floor"] = tf_math_floor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.floor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.floor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.floor', generated_inputs['tf.math.floor'], lib="tf", suffix=0)
