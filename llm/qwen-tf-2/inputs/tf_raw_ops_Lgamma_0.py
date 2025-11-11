
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def lgamma_inputs():
    list_of_inputs = []
    
    # Input 1: Scalar tensor
    x = np.array(5.0, dtype=np.float32)
    input_dict = {"name": "lgamma_1", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 2: 1D array with negative values
    x = np.array([-4.0, -5.6], dtype=np.float32)
    input_dict = {"name": "lgamma_2", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 3: 1D array with positive values
    x = np.array([0.5, 1.0, 4.5], dtype=np.float32)
    input_dict = {"name": "lgamma_3", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 4: 2D array with mixed values
    x = np.array([[0, 0.5], [1, 4.5]], dtype=np.float32)
    input_dict = {"name": "lgamma_4", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 5: Float64 array
    x = np.array([0.5, 1.0, 4.5], dtype=np.float64)
    input_dict = {"name": "lgamma_5", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 6: Half array
    x = np.array([0.5, 1.0, 4.5], dtype=np.float16)
    input_dict = {"name": "lgamma_6", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 7: Scalar tensor with negative value
    x = np.array(-4.0, dtype=np.float32)
    input_dict = {"name": "lgamma_7", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 8: Array with zero values
    x = np.array([0, 0.5], dtype=np.float32)
    input_dict = {"name": "lgamma_8", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 9: 1D array with mixed positive and negative values
    x = np.array([0.5, -4.0, 1.0], dtype=np.float32)
    input_dict = {"name": "lgamma_9", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 10: 1D array with float64 values
    x = np.array([0.5, 1.0, 4.5], dtype=np.float64)
    input_dict = {"name": "lgamma_10", "x": x}
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Lgamma"] = lgamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Lgamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Lgamma'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Lgamma', generated_inputs['tf.raw_ops.Lgamma'], lib="tf", suffix=0)
