
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_math_cos_inputs():
    list_of_inputs = []
    
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "name": "cos_op1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32)
    input_dict = {"x": x, "name": "cos_op2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[0.0, 1.57], [-1.57, 3.14]], dtype=np.float32)
    input_dict = {"x": x, "name": "cos_op3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]], dtype=np.float32)
    input_dict = {"x": x, "name": "cos_op4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(1.5, dtype=np.float32)
    input_dict = {"x": x, "name": "cos_op5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    input_dict = {"x": x, "name": "cos_op6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([100.0, 200.0, 1000.0], dtype=np.float32)
    input_dict = {"x": x, "name": "cos_op7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.001, -0.001, 0.0001], dtype=np.float32)
    input_dict = {"x": x, "name": "cos_op8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0+2.0j, 3.0+4.0j, 5.0+6.0j], dtype=np.complex64)
    input_dict = {"x": x, "name": "cos_op9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.5+0.5j, -1.0-1.0j], dtype=np.complex128)
    input_dict = {"x": x, "name": "cos_op10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.0, 1.5, -1.5], dtype=np.float16)
    input_dict = {"x": x, "name": "cos_op11"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.5, 1.0, -0.5], dtype=np.float32)
    input_dict = {"x": x, "name": "cos_op12"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.cos"] = tf_math_cos_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.cos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.cos'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.cos', generated_inputs['tf.math.cos'], lib="tf", suffix=0)
