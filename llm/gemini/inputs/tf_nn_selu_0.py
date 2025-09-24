
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_selu_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tensor with positive and negative values
    features = tf.constant(np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)).numpy()
    name = "selu_1"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor with mixed values
    features = tf.constant(np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)).numpy()
    name = "selu_2"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    features = tf.constant(np.random.rand(2, 3, 4).astype(np.float32)).numpy() - 0.5
    name = "selu_3"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with all positive values
    features = tf.constant(np.array([[0.1, 0.5], [1.0, 2.0]], dtype=np.float32)).numpy()
    name = "selu_4"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with all negative values
    features = tf.constant(np.array([[-0.1, -0.5], [-1.0, -2.0]], dtype=np.float32)).numpy()
    name = "selu_5"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large values
    features = tf.constant(np.array([[-100.0, 0.0], [100.0, 200.0]], dtype=np.float32)).numpy()
    name = "selu_6"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Half type
    features = tf.constant(np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float16)).numpy()
    name = "selu_7"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 type. Removing bfloat16 due to compatibility issues
    features = tf.constant(np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)).numpy()
    name = "selu_8"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty tensor
    features = tf.constant(np.array([], dtype=np.float32)).numpy()
    name = "selu_9"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Simple tensor
    features = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32)).numpy()
    name = "selu_10"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.selu"] = tf_nn_selu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.selu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.selu'.")

check_valid('tf.nn.selu', generated_inputs['tf.nn.selu'], lib="tf", suffix=0)
