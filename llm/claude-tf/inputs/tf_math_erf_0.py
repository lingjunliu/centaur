
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_math_erf_inputs():
    list_of_inputs = []
    
    x = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, -2.0]], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.5, -0.5, 1.5, -1.5], dtype=np.float32)
    input_dict = {"x": x, "name": "erf_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(1.0, dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float64)
    input_dict = {"x": x, "name": "erf_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([5.0, 10.0, 15.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-5.0, -10.0, -15.0], dtype=np.float32)
    input_dict = {"x": x, "name": "erf_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.001, -0.001, 0.0001, -0.0001], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "name": "erf_zeros"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1.5, -1.5], [2.5, -2.5]], dtype=np.float16)
    input_dict = {"x": x, "name": "erf_float16"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.erf"] = tf_math_erf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.erf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.erf'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.erf', generated_inputs['tf.math.erf'], lib="tf", suffix=0)
