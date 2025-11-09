
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_math_is_inf_inputs():
    list_of_inputs = []
    
    x = np.array([5.0, np.inf, 6.8, np.inf], dtype=np.float32)
    name = "test1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0, -np.inf, 3.0, np.inf], dtype=np.float32)
    name = "test2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1.5, np.inf], [np.inf, -np.inf]], dtype=np.float64)
    name = "test3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    name = "test4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[1.0, np.inf], [2.0, 3.0]], [[np.inf, 5.0], [6.0, -np.inf]]], dtype=np.float16)
    name = "test5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([np.inf], dtype=np.float32)
    name = "test6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float64)
    name = "test7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.0, -1.5, 2.3, np.inf, -np.inf, 100.0], dtype=np.float32)
    name = "test8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[np.inf, np.inf], [np.inf, np.inf]], dtype=np.float32)
    name = "test9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-5.0, -10.0, -np.inf, -0.5], dtype=np.float64)
    name = "test10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.is_inf"] = tf_math_is_inf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.is_inf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.is_inf'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.is_inf', generated_inputs['tf.math.is_inf'], lib="tf", suffix=0)
