
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def tf_raw_ops_L2Loss_inputs():
    list_of_inputs = []
    
    # Input 1 - 2D tensor with positive values
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"name": "test1", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 2 - 2D tensor with negative values
    t = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {"name": "test2", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 3 - 1D tensor with mixed values
    t = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    input_dict = {"name": "test3", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 4 - 3D tensor
    t = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"name": "test4", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 5 - 1D tensor with float64
    t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"name": "test5", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 6 - 2D tensor with bfloat16
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"name": "test6", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 7 - 2D tensor with half
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"name": "test7", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 8 - 2D tensor with zero values
    t = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    input_dict = {"name": "test8", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 9 - 1D tensor with large values
    t = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    input_dict = {"name": "test9", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 10 - 2D tensor with decimal values
    t = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    input_dict = {"name": "test10", "t": t}
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.raw_ops.L2Loss"] = tf_raw_ops_L2Loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.L2Loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.L2Loss'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.L2Loss', generated_inputs['tf.raw_ops.L2Loss'], lib="tf", suffix=0)
