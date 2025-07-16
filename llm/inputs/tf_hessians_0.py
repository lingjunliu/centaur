
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_hessians_inputs():
    list_of_inputs = []

    # Input 1: Basic case with single input and output
    x = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    y = tf.square(x)
    input_dict = {
        "ys": [tf.reduce_sum(y)],
        "xs": [x],
        "gate_gradients": False,
        "aggregation_method": None,
        "name": "hessians_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple inputs and outputs
    x1 = tf.constant(np.array([[1.0, 2.0]], dtype=np.float32))
    x2 = tf.constant(np.array([[3.0], [4.0]], dtype=np.float32))
    y1 = tf.square(x1)
    y2 = tf.multiply(x1, x2)
    input_dict = {
        "ys": [tf.reduce_sum(y1), tf.reduce_sum(y2)],
        "xs": [x1, x2],
        "gate_gradients": True,
        "aggregation_method": "sum",
        "name": "hessians_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: ys as a single tensor
    x = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    y = tf.square(x) + tf.constant(1.0)
    input_dict = {
        "ys": [tf.reduce_sum(y)],
        "xs": [x],
        "gate_gradients": False,
        "aggregation_method": "mean",
        "name": "hessians_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Higher dimension tensor
    x = tf.constant(np.random.rand(2, 3, 4).astype(np.float32))
    y = tf.reduce_sum(tf.square(x))
    input_dict = {
        "ys": [y],
        "xs": [x],
        "gate_gradients": True,
        "aggregation_method": "experimental_accumulate_n",
        "name": "hessians_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5: xs as a single tensor
    x = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    y = tf.square(x)
    input_dict = {
        "ys": [tf.reduce_sum(y)],
        "xs": [x],
        "gate_gradients": False,
        "aggregation_method": None,
        "name": "hessians_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different aggregation method
    x = tf.constant(np.array([[1.0, 2.0]], dtype=np.float32))
    y = tf.square(x)
    input_dict = {
        "ys": [tf.reduce_sum(y)],
        "xs": [x],
        "gate_gradients": False,
        "aggregation_method": "tree",
        "name": "hessians_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: More complex expression
    x = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    y = tf.sin(x) * x
    input_dict = {
        "ys": [tf.reduce_sum(y)],
        "xs": [x],
        "gate_gradients": True,
        "aggregation_method": None,
        "name": "hessians_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: List of xs tensors
    x1 = tf.constant(np.array([[1.0, 2.0]], dtype=np.float32))
    x2 = tf.constant(np.array([[3.0, 4.0]], dtype=np.float32))
    y = tf.square(x1) + tf.square(x2)
    input_dict = {
        "ys": [tf.reduce_sum(y)],
        "xs": [x1, x2],
        "gate_gradients": False,
        "aggregation_method": "sum",
        "name": "hessians_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: More complex ys
    x = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    y1 = tf.square(x)
    y2 = tf.exp(x)
    input_dict = {
        "ys": [tf.reduce_sum(y1), tf.reduce_sum(y2)],
        "xs": [x],
        "gate_gradients": True,
        "aggregation_method": "mean",
        "name": "hessians_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: With negative values
    x = tf.constant(np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32))
    y = tf.square(x)
    input_dict = {
        "ys": [tf.reduce_sum(y)],
        "xs": [x],
        "gate_gradients": False,
        "aggregation_method": None,
        "name": "hessians_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: scalar value for ys
    x = tf.constant(np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32))
    y = tf.constant(5.0, dtype=np.float32)

    input_dict = {
        "ys": [y],
        "xs": [x],
        "gate_gradients": False,
        "aggregation_method": None,
        "name": "hessians_11"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.hessians"] = tf_hessians_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.hessians' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.hessians'.")

check_valid('tf.hessians', generated_inputs['tf.hessians'], lib="tf", suffix=0)
