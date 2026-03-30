
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_gradients_inputs():
    list_of_inputs = []

    # Input 1
    ys = [tf.constant(1.0)]
    xs = [tf.constant(2.0)]
    grad_ys = [tf.constant(1.0)]
    name = "gradients_1"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = None
    unconnected_gradients = "none"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients,
                  "aggregation_method": aggregation_method, "stop_gradients": stop_gradients,
                  "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ys = [tf.constant([1.0, 2.0])]
    xs = [tf.constant([5.0, 6.0])]
    grad_ys = [tf.constant([1.0, 1.0])]
    name = "gradients_2"
    gate_gradients = True
    aggregation_method = None
    stop_gradients = None
    unconnected_gradients = "zero"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients,
                  "aggregation_method": aggregation_method, "stop_gradients": stop_gradients,
                  "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ys = [tf.constant([[1.0, 2.0], [3.0, 4.0]])]
    xs = [tf.constant([[5.0, 6.0], [7.0, 8.0]])]
    grad_ys = [tf.constant([[1.0, 1.0], [1.0, 1.0]])]
    name = "gradients_3"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = None
    unconnected_gradients = "none"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients,
                  "aggregation_method": aggregation_method, "stop_gradients": stop_gradients,
                  "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ys = [tf.constant(1.0, dtype=tf.float64)]
    xs = [tf.constant(2.0, dtype=tf.float64)]
    grad_ys = [tf.constant(1.0, dtype=tf.float64)]
    name = "gradients_4"
    gate_gradients = True
    aggregation_method = None
    stop_gradients = None
    unconnected_gradients = "zero"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients,
                  "aggregation_method": aggregation_method, "stop_gradients": stop_gradients,
                  "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    ys = [tf.constant([1.0, 2.0, 3.0])]
    xs = [tf.constant([4.0, 5.0, 6.0])]
    grad_ys = [tf.constant([0.5, 0.5, 0.5])]
    name = "gradients_5"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = None
    unconnected_gradients = "none"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients,
                  "aggregation_method": aggregation_method, "stop_gradients": stop_gradients,
                  "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ys = [tf.constant([1.0])]
    xs = [tf.constant([2.0]), tf.constant([3.0])]
    grad_ys = [tf.constant([1.0])]
    name = "gradients_6"
    gate_gradients = True
    aggregation_method = None
    stop_gradients = None
    unconnected_gradients = "zero"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients,
                  "aggregation_method": aggregation_method, "stop_gradients": stop_gradients,
                  "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
   
    ys = [tf.constant([[1.0, 2.0], [3.0, 4.0]])]
    xs = [tf.constant([[5.0, 6.0], [7.0, 8.0]])]
    grad_ys = [tf.constant([[1.0, 1.0], [1.0, 1.0]])]
    name = "gradients_7"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = None
    unconnected_gradients = "none"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients,
                  "aggregation_method": aggregation_method, "stop_gradients": stop_gradients,
                  "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    ys = [tf.constant([1.0, 2.0], dtype=tf.float32)]
    xs = [tf.constant([3.0, 4.0], dtype=tf.float32), tf.constant([5.0, 6.0], dtype=tf.float32)]
    grad_ys = [tf.constant([0.5, 0.5], dtype=tf.float32)]
    name = "gradients_8"
    gate_gradients = True
    aggregation_method = None
    stop_gradients = None
    unconnected_gradients = "zero"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients,
                  "aggregation_method": aggregation_method, "stop_gradients": stop_gradients,
                  "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    ys = [tf.constant(1.0, dtype=tf.float64)]
    xs = [tf.constant(3.0, dtype=tf.float64)]
    grad_ys = [tf.constant(0.5, dtype=tf.float64)]
    name = "gradients_9"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = None
    unconnected_gradients = "none"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients,
                  "aggregation_method": aggregation_method, "stop_gradients": stop_gradients,
                  "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    ys = [tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)]
    xs = [tf.constant([5.0, 6.0], dtype=tf.float32), tf.constant([[7.0, 8.0], [9.0, 10.0]], dtype=tf.float32)]
    grad_ys = [tf.constant([[0.1, 0.2], [0.3, 0.4]], dtype=tf.float32)]
    name = "gradients_10"
    gate_gradients = True
    aggregation_method = None
    stop_gradients = None
    unconnected_gradients = "zero"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients,
                  "aggregation_method": aggregation_method, "stop_gradients": stop_gradients,
                  "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    ys = [tf.constant(1.0)]
    xs = [tf.constant(2.0)]
    grad_ys = None
    name = "gradients_11"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = None
    unconnected_gradients = "none"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients,
                  "aggregation_method": aggregation_method, "stop_gradients": stop_gradients,
                  "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    ys = [tf.constant(1.0)]
    xs = [tf.constant(2.0)]
    grad_ys = None
    name = "gradients_12"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = [tf.constant(3.0)]
    unconnected_gradients = "none"
    input_dict = {"ys": ys, "xs": xs, "grad_ys": grad_ys, "name": name, "gate_gradients": gate_gradients,
                  "aggregation_method": aggregation_method, "stop_gradients": stop_gradients,
                  "unconnected_gradients": unconnected_gradients}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.gradients"] = tf_gradients_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.gradients' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.gradients'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.gradients', generated_inputs['tf.gradients'], lib="tf", suffix=0)
