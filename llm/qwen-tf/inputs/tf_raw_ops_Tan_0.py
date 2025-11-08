
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def generate_tan_inputs():
    list_of_inputs = []
    
    # Input 1: scalar tensor
    input_tensor = np.array(1.0, dtype=np.float32)
    input_dict = {"name": "tan_1", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 2: 1D tensor with negative values
    input_tensor = np.array([-1.5, -0.5, 0.5], dtype=np.float32)
    input_dict = {"name": "tan_2", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 3: 2D tensor
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"name": "tan_3", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 4: 3D tensor
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"name": "tan_4", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 5: tensor with inf values
    input_tensor = np.array([np.inf, -np.inf, 0.0], dtype=np.float32)
    input_dict = {"name": "tan_5", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 6: tensor with nan values
    input_tensor = np.array([np.nan, np.nan, 1.0], dtype=np.float32)
    input_dict = {"name": "tan_6", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 7: complex tensor
    input_tensor = np.array([1+2j, 2+1j, 3+4j], dtype=np.complex64)
    input_dict = {"name": "tan_7", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 8: half precision tensor
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input_dict = {"name": "tan_8", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 9: bfloat16 tensor
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # Note: numpy doesn't support bfloat16 directly, but using float32 for simulation
    input_dict = {"name": "tan_9", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 10: tensor with mixed values including zero
    input_tensor = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    input_dict = {"name": "tan_10", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Tan"] = generate_tan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Tan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Tan'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Tan', generated_inputs['tf.raw_ops.Tan'], lib="tf", suffix=0)
