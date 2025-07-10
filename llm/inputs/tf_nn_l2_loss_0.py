
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_l2_loss_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D tensor
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    name = "l2_loss_1"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor
    t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "l2_loss_2"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    t = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    name = "l2_loss_3"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with negative values
    t = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    name = "l2_loss_4"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with zeros
    t = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    name = "l2_loss_5"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float64 tensor
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    name = "l2_loss_6"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float16 tensor
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    name = "l2_loss_7"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large values
    t = np.array([[1000.0, 2000.0], [3000.0, 4000.0]], dtype=np.float32)
    name = "l2_loss_8"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Half tensor
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    name = "l2_loss_9"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty string name
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    name = ""
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.l2_loss"] = tf_nn_l2_loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.l2_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.l2_loss'.")

check_valid('tf.nn.l2_loss', generated_inputs['tf.nn.l2_loss'], lib="tf", suffix=0)
