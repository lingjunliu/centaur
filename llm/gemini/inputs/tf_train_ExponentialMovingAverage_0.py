
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_train_exponentialmovingaverage_inputs():
    list_of_inputs = []

    # Input 1
    decay = 0.999
    num_updates = 100
    zero_debias = False
    name = "ema1"
    input_dict = {"decay": decay, "num_updates": num_updates, "zero_debias": zero_debias, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    decay = 0.99
    num_updates = 500
    zero_debias = True
    name = "ema2"
    input_dict = {"decay": decay, "num_updates": num_updates, "zero_debias": zero_debias, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    decay = 0.9
    num_updates = 1000
    zero_debias = False
    name = "ema3"
    input_dict = {"decay": decay, "num_updates": num_updates, "zero_debias": zero_debias, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    decay = 0.9999
    num_updates = 10
    zero_debias = True
    name = "ema4"
    input_dict = {"decay": decay, "num_updates": num_updates, "zero_debias": zero_debias, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    decay = 0.5
    num_updates = 50
    zero_debias = False
    name = "ema5"
    input_dict = {"decay": decay, "num_updates": num_updates, "zero_debias": zero_debias, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    decay = 0.995
    num_updates = 250
    zero_debias = True
    name = "ema6"
    input_dict = {"decay": decay, "num_updates": num_updates, "zero_debias": zero_debias, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    decay = 0.99999
    num_updates = 1
    zero_debias = False
    name = "ema7"
    input_dict = {"decay": decay, "num_updates": num_updates, "zero_debias": zero_debias, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    decay = 0.0
    num_updates = 10000
    zero_debias = True
    name = "ema8"
    input_dict = {"decay": decay, "num_updates": num_updates, "zero_debias": zero_debias, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    decay = 1.0
    num_updates = 0
    zero_debias = False
    name = "ema9"
    input_dict = {"decay": decay, "num_updates": num_updates, "zero_debias": zero_debias, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    decay = 0.75
    num_updates = 123
    zero_debias = True
    name = "ema10"
    input_dict = {"decay": decay, "num_updates": num_updates, "zero_debias": zero_debias, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    decay = 0.9995
    num_updates = -1
    zero_debias = False
    name = "ema11"
    input_dict = {"decay": decay, "num_updates": num_updates, "zero_debias": zero_debias, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.train.ExponentialMovingAverage"] = tf_train_exponentialmovingaverage_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.train.ExponentialMovingAverage' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.train.ExponentialMovingAverage'.")

check_valid('tf.train.ExponentialMovingAverage', generated_inputs['tf.train.ExponentialMovingAverage'], lib="tf", suffix=0)
