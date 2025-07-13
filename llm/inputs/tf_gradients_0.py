
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_gradients_inputs():
    list_of_inputs = []

    # Input 1
    ys = [tf.constant(1.0)]
    xs = [tf.constant(2.0)]
    grad_ys = [tf.constant(1.0)]
    name = "gradients1"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = [tf.constant(2.0)]
    unconnected_gradients = "none"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "stop_gradients": stop_gradients, "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ys = [tf.constant([1.0, 2.0])]
    xs = [tf.constant([3.0, 4.0])]
    grad_ys = [tf.constant([1.0, 1.0])]
    name = "gradients2"
    gate_gradients = True
    aggregation_method = None
    stop_gradients = [tf.constant([3.0, 4.0])]
    unconnected_gradients = "zero"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "stop_gradients": stop_gradients, "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ys = [tf.constant([[1.0, 2.0], [3.0, 4.0]])]
    xs = [tf.constant([[5.0, 6.0], [7.0, 8.0]])]
    grad_ys = [tf.constant([[1.0, 1.0], [1.0, 1.0]])]
    name = "gradients3"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = [tf.constant([[5.0, 6.0], [7.0, 8.0]])]
    unconnected_gradients = "none"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "stop_gradients": stop_gradients, "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ys = [tf.constant(1.0), tf.constant(2.0)]
    xs = [tf.constant(3.0), tf.constant(4.0)]
    grad_ys = [tf.constant(1.0), tf.constant(1.0)]
    name = "gradients4"
    gate_gradients = True
    aggregation_method = None
    stop_gradients = [tf.constant(3.0), tf.constant(4.0)]
    unconnected_gradients = "zero"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "stop_gradients": stop_gradients, "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    ys = [tf.constant([1.0, 2.0]), tf.constant([3.0, 4.0])]
    xs = [tf.constant([5.0, 6.0]), tf.constant([7.0, 8.0])]
    grad_ys = [tf.constant([1.0, 1.0]), tf.constant([1.0, 1.0])]
    name = "gradients5"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = [tf.constant([5.0, 6.0]), tf.constant([7.0, 8.0])]
    unconnected_gradients = "none"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "stop_gradients": stop_gradients, "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ys = [tf.constant([[1.0, 2.0], [3.0, 4.0]]), tf.constant([[5.0, 6.0], [7.0, 8.0]])]
    xs = [tf.constant([[9.0, 10.0], [11.0, 12.0]]), tf.constant([[13.0, 14.0], [15.0, 16.0]])]
    grad_ys = [tf.constant([[1.0, 1.0], [1.0, 1.0]]), tf.constant([[1.0, 1.0], [1.0, 1.0]])]
    name = "gradients6"
    gate_gradients = True
    aggregation_method = None
    stop_gradients = [tf.constant([[9.0, 10.0], [11.0, 12.0]]), tf.constant([[13.0, 14.0], [15.0, 16.0]])]
    unconnected_gradients = "zero"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "stop_gradients": stop_gradients, "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    ys = [tf.constant(1.0, dtype=tf.float64)]
    xs = [tf.constant(2.0, dtype=tf.float64)]
    grad_ys = [tf.constant(1.0, dtype=tf.float64)]
    name = "gradients7"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = [tf.constant(2.0, dtype=tf.float64)]
    unconnected_gradients = "none"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "stop_gradients": stop_gradients, "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    ys = [tf.constant(1.0), tf.constant(2.0)]
    xs = [tf.constant(3.0)]
    grad_ys = [tf.constant(1.0), tf.constant(1.0)]
    name = "gradients8"
    gate_gradients = True
    aggregation_method = None
    stop_gradients = [tf.constant(3.0)]
    unconnected_gradients = "zero"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "stop_gradients": stop_gradients, "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    ys = [tf.constant(np.array([1.0, 2.0]))]
    xs = [tf.constant(np.array([5.0, 6.0]))]
    grad_ys = [tf.constant(np.array([1.0, 1.0]))]
    name = "gradients9"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = [tf.constant(np.array([5.0, 6.0]))]
    unconnected_gradients = "none"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "stop_gradients": stop_gradients, "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    ys = [tf.constant(1.0), tf.constant(2.0)]
    xs = [tf.constant(3.0), tf.constant(4.0)]
    grad_ys = [tf.constant(1.0), tf.constant(1.0)]
    name = "gradients10"
    gate_gradients = True
    aggregation_method = None
    stop_gradients = [tf.constant(3.0), tf.constant(4.0)]
    unconnected_gradients = "none"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "stop_gradients": stop_gradients, "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.gradients"] = tf_gradients_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.gradients' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.gradients'.")

check_valid('tf.gradients', generated_inputs['tf.gradients'], lib="tf", suffix=0)
