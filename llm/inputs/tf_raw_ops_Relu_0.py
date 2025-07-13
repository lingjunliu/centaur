
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_relu_inputs():
    list_of_inputs = []

    # Input 1: float32, positive values
    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "relu_op_1"
    input_dict = {"features": tf.convert_to_tensor(features).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, negative and positive values
    features = np.array([-1.0, 0.0, 2.0, -3.0], dtype=np.float32)
    name = "relu_op_2"
    input_dict = {"features": tf.convert_to_tensor(features).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32, positive values
    features = np.array([1, 2, 3], dtype=np.int32)
    name = "relu_op_3"
    input_dict = {"features": tf.convert_to_tensor(features).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int32, negative and positive values
    features = np.array([-1, 0, 2, -3], dtype=np.int32)
    name = "relu_op_4"
    input_dict = {"features": tf.convert_to_tensor(features).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64, positive values, multi-dimensional
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    name = "relu_op_5"
    input_dict = {"features": tf.convert_to_tensor(features).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int64, negative values
    features = np.array([-1, -2, -3], dtype=np.int64)
    name = "relu_op_6"
    input_dict = {"features":  tf.convert_to_tensor(features).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8, positive values
    features = np.array([1, 2, 3], dtype=np.uint8)
    name = "relu_op_7"
    input_dict = {"features": tf.convert_to_tensor(features).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: uint32
    features = np.array([1, 2, 3], dtype=np.uint32)
    name = "relu_op_9"
    input_dict = {"features": tf.convert_to_tensor(features).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: uint64
    features = np.array([1, 2, 3], dtype=np.uint64)
    name = "relu_op_10"
    input_dict = {"features": tf.convert_to_tensor(features).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Relu"] = tf_raw_ops_relu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Relu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Relu'.")

check_valid('tf.raw_ops.Relu', generated_inputs['tf.raw_ops.Relu'], lib="tf", suffix=0)
