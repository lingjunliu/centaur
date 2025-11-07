
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Elu_inputs():
    list_of_inputs = []
    
    # Input 1: scalar tensor
    features = np.array(1.0, dtype=np.float32)
    input_dict = {"name": "elu_scalar", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: scalar tensor with negative value
    features = np.array(-1.0, dtype=np.float32)
    input_dict = {"name": "elu_negative_scalar", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D tensor
    features = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    input_dict = {"name": "elu_1d", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 1D tensor with negative values
    features = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {"name": "elu_1d_negative", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor
    features = np.array([[1.0, -1.0], [0.0, -2.0]], dtype=np.float32)
    input_dict = {"name": "elu_2d", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensor with negative values
    features = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {"name": "elu_2d_negative", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D tensor
    features = np.array([[[1.0, -1.0], [0.0, -2.0]], [[-3.0, -4.0], [-5.0, -6.0]]], dtype=np.float32)
    input_dict = {"name": "elu_3d", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 3D tensor with negative values
    features = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {"name": "elu_3d_negative", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: float64 tensor
    features = np.array([1.0, -1.0], dtype=np.float64)
    input_dict = {"name": "elu_float64", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: bfloat16 tensor
    features = np.array([1.0, -1.0], dtype=np.float32)  # bfloat16 is not directly supported in numpy so we use float32
    input_dict = {"name": "elu_bfloat16", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Elu"] = tf_raw_ops_Elu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Elu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Elu'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Elu', generated_inputs['tf.raw_ops.Elu'], lib="tf", suffix=0)
