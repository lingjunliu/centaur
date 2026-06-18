
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_categorical_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'logits': np.array([[0.1, 0.9]], dtype=np.float32),
        'num_samples': 5,
        'seed': np.array([7, 17], dtype=np.int32),
        'dtype': np.int64,
        'name': "stateless_cat_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'logits': np.array([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        'num_samples': 10,
        'seed': np.array([42, 43], dtype=np.int32),
        'dtype': np.int32,
        'name': "stateless_cat_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'logits': np.random.normal(size=(5, 5)).astype(np.float32),
        'num_samples': 1,
        'seed': np.array([123, 456], dtype=np.int64),
        'dtype': np.int64,
        'name': "stateless_cat_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'logits': np.array([[-10.0, -10.0], [10.0, 10.0]], dtype=np.float64),
        'num_samples': 3,
        'seed': np.array([9, 8], dtype=np.int32),
        'dtype': np.int32,
        'name': "stateless_cat_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'logits': np.zeros((10, 2), dtype=np.float32),
        'num_samples': 2,
        'seed': np.array([100, 200], dtype=np.int32),
        'dtype': np.int64,
        'name': "stateless_cat_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'logits': np.log(np.ones((1, 100)) / 100.0).astype(np.float32),
        'num_samples': 50,
        'seed': np.array([1, 1], dtype=np.int32),
        'dtype': np.int32,
        'name': "stateless_cat_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'logits': np.array([[-1.0, 0.0, 1.0]], dtype=np.float32),
        'num_samples': 20,
        'seed': np.array([999, 999], dtype=np.int64),
        'dtype': np.int64,
        'name': "stateless_cat_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'logits': np.array([[100.0, 0.0], [0.0, 100.0]], dtype=np.float32),
        'num_samples': 15,
        'seed': np.array([777, 888], dtype=np.int32),
        'dtype': np.int32,
        'name': "stateless_cat_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'logits': np.array([[-0.5, -0.5, -0.5, -0.5]], dtype=np.float64),
        'num_samples': 8,
        'seed': np.array([3, 4], dtype=np.int64),
        'dtype': np.int64,
        'name': "stateless_cat_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'logits': np.random.uniform(-5.0, 5.0, size=(3, 10)).astype(np.float32),
        'num_samples': 4,
        'seed': np.array([5, 12], dtype=np.int32),
        'dtype': np.int32,
        'name': "stateless_cat_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_categorical"] = tf_random_stateless_categorical_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_categorical' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_categorical'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_categorical', generated_inputs['tf.random.stateless_categorical'], lib="tf", suffix=0)
