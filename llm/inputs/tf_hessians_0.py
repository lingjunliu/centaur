
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_hessians_inputs():
    list_of_inputs = []

    # Input 1
    ys = [tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))]
    xs = [tf.constant(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))]
    gate_gradients = False
    aggregation_method = None
    name = "hessian_1"
    input_dict = {"ys": ys, "xs": xs, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ys = [tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))]
    xs = [tf.constant(np.array([4.0, 5.0, 6.0], dtype=np.float32))]
    gate_gradients = True
    aggregation_method = "sum"
    name = "hessian_2"
    input_dict = {"ys": ys, "xs": xs, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ys = [tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32))]
    xs = [tf.constant(np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]], dtype=np.float32))]
    gate_gradients = False
    aggregation_method = "mean"
    name = "hessian_3"
    input_dict = {"ys": ys, "xs": xs, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
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
