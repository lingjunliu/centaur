
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_scale_regularization_loss_inputs():
    list_of_inputs = []

    # Input 1: Scalar tensor
    regularization_loss = tf.constant(1.0)
    input_dict = {"regularization_loss": regularization_loss}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive value
    regularization_loss = tf.constant(5.0)
    input_dict = {"regularization_loss": regularization_loss}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative value
    regularization_loss = tf.constant(-2.0)
    input_dict = {"regularization_loss": regularization_loss}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero value
    regularization_loss = tf.constant(0.0)
    input_dict = {"regularization_loss": regularization_loss}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rank 1 tensor
    regularization_loss = tf.constant([1.0, 2.0, 3.0])
    input_dict = {"regularization_loss": regularization_loss}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Rank 2 tensor
    regularization_loss = tf.constant([[1.0, 2.0], [3.0, 4.0]])
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
