
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_softplus_inputs():
    list_of_inputs = []
    
    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "softplus_1"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    name = "softplus_2"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    name = "softplus_3"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    name = "softplus_4"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([-5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    name = "softplus_5"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    name = "softplus_6"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    name = "softplus_7"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array(5.0, dtype=np.float32)
    name = "softplus_8"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    name = "softplus_9"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float64)
    name = "softplus_10"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = "softplus_11"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Softplus"] = tf_raw_ops_softplus_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Softplus' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Softplus'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Softplus', generated_inputs['tf.raw_ops.Softplus'], lib="tf", suffix=0)
