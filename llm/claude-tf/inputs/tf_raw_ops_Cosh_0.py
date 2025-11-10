
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_cosh_inputs():
    list_of_inputs = []
    
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"name": "cosh_op1", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {"name": "cosh_op2", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-0.5, 1.0, -1.2, 2.0], dtype=np.float32)
    input_dict = {"name": "cosh_op3", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.0, 1.5, -2.5, 3.0], dtype=np.float64)
    input_dict = {"name": "cosh_op4", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"name": "cosh_op5", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[0.5, -0.5], [1.0, -1.0]], [[2.0, -2.0], [3.0, -3.0]]], dtype=np.float32)
    input_dict = {"name": "cosh_op6", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(1.5, dtype=np.float32)
    input_dict = {"name": "cosh_op7", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.0], dtype=np.float32)
    input_dict = {"name": "cosh_op8", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([10.0, 15.0, 20.0], dtype=np.float32)
    input_dict = {"name": "cosh_op9", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.1, 0.5, 1.0], dtype=np.float16)
    input_dict = {"name": "cosh_op10", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1+2j, 2-1j, 0+1j], dtype=np.complex64)
    input_dict = {"name": "cosh_op11", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.5+0.5j, -1.0+2.0j], dtype=np.complex128)
    input_dict = {"name": "cosh_op12", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Cosh"] = tf_raw_ops_cosh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Cosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Cosh'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Cosh', generated_inputs['tf.raw_ops.Cosh'], lib="tf", suffix=0)
