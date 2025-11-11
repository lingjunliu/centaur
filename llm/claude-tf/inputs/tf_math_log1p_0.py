
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_log1p_inputs():
    list_of_inputs = []
    
    x = np.array([0, 0.5, 1, 5], dtype=np.float32)
    name = "log1p_op1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    name = "log1p_op2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(2.0, dtype=np.float32)
    name = "log1p_op3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    name = "log1p_op4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-0.5, -0.3, -0.1, 0.0], dtype=np.float32)
    name = "log1p_op5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([10.0, 100.0, 1000.0], dtype=np.float64)
    name = "log1p_op6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1e-10, 1e-8, 1e-6, 1e-4], dtype=np.float64)
    name = "log1p_op7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1+2j, 3+4j, 0+1j], dtype=np.complex64)
    name = "log1p_op8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    name = "log1p_op9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.random.rand(2, 2, 2, 2).astype(np.float32)
    name = "log1p_op10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.5, 1.5, 2.5], dtype=np.float16)
    name = "log1p_op11"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.log1p"] = tf_math_log1p_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.log1p'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.log1p', generated_inputs['tf.math.log1p'], lib="tf", suffix=0)
