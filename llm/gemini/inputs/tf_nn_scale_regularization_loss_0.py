
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_scale_regularization_loss_inputs():
    list_of_inputs = []

    # Input 1: Scalar loss
    regularization_loss = np.float32(0.1)
    input_dict = {"regularization_loss": regularization_loss}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Small positive loss
    regularization_loss = np.float32(0.0001)
    input_dict = {"regularization_loss": regularization_loss}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero loss
    regularization_loss = np.float32(0.0)
    input_dict = {"regularization_loss": regularization_loss}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative loss (although regularization loss should ideally be non-negative)
    regularization_loss = np.float32(-0.1)
    input_dict = {"regularization_loss": regularization_loss}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger loss
    regularization_loss = np.float32(10.0)
    input_dict = {"regularization_loss": regularization_loss}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Loss with a different dtype
    regularization_loss = np.float64(0.1)
    input_dict = {"regularization_loss": regularization_loss}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Loss with a very small value
    regularization_loss = np.float32(1e-8)
    input_dict = {"regularization_loss": regularization_loss}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large value
    regularization_loss = np.float32(1e8)
    input_dict = {"regularization_loss": regularization_loss}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  A different small value
    regularization_loss = np.float32(1e-4)
    input_dict = {"regularization_loss": regularization_loss}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: A different medium value
    regularization_loss = np.float32(5.0)
    input_dict = {"regularization_loss": regularization_loss}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.scale_regularization_loss"] = tf_nn_scale_regularization_loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.scale_regularization_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.scale_regularization_loss'.")

check_valid('tf.nn.scale_regularization_loss', generated_inputs['tf.nn.scale_regularization_loss'], lib="tf", suffix=0)
