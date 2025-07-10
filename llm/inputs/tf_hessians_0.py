
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_hessians_inputs():
    list_of_inputs = []

    # Input 1
    x = tf.constant(1.0)
    y = x**2
    ys = [y]
    xs = [x]
    gate_gradients = False
    aggregation_method = None
    name = "hessians_1"
    input_dict = {"ys": [np.float64(y.numpy())], "xs": [np.float64(x.numpy())], "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = tf.constant([1.0, 2.0])
    y = tf.reduce_sum(x**2)
    ys = [y]
    xs = [x]
    gate_gradients = True
    aggregation_method = "experimental_accumulate_n"
    name = "hessians_2"
    input_dict = {"ys": [np.float64(y.numpy())], "xs": [x.numpy()], "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y = tf.reduce_sum(x**2)
    ys = [y]
    xs = [x]
    gate_gradients = False
    aggregation_method = "tree"
    name = "hessians_3"
    input_dict = {"ys": [np.float64(y.numpy())], "xs": [x.numpy()], "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x1 = tf.constant(1.0)
    x2 = tf.constant(2.0)
    y = x1**2 + x2**3
    ys = [y]
    xs = [x1, x2]
    gate_gradients = True
    aggregation_method = "unsorted_segments_sum"
    name = "hessians_4"
    input_dict = {"ys": [np.float64(y.numpy())], "xs": [np.float64(x1.numpy()), np.float64(x2.numpy())], "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = tf.constant([1.0, 2.0])
    y1 = x[0]**2
    y2 = x[1]**3
    ys = [y1, y2]
    xs = [x]
    gate_gradients = False
    aggregation_method = "sum"
    name = "hessians_5"
    input_dict = {"ys": [np.float64(y1.numpy()), np.float64(y2.numpy())], "xs": [x.numpy()], "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y1 = tf.reduce_sum(x[0]**2)
    y2 = tf.reduce_sum(x[1]**3)
    ys = [y1, y2]
    xs = [x]
    gate_gradients = True
    aggregation_method = None
    name = "hessians_6"
    input_dict = {"ys": [np.float64(y1.numpy()), np.float64(y2.numpy())], "xs": [x.numpy()], "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x1 = tf.constant(1.0)
    x2 = tf.constant(2.0)
    y1 = x1**2
    y2 = x2**3
    ys = [y1, y2]
    xs = [x1, x2]
    gate_gradients = False
    aggregation_method = "mean"
    name = "hessians_7"
    input_dict = {"ys": [np.float64(y1.numpy()), np.float64(y2.numpy())], "xs": [np.float64(x1.numpy()), np.float64(x2.numpy())], "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = tf.constant([[-1.0, -2.0], [-3.0, -4.0]])
    y = tf.reduce_sum(x**2)
    ys = [y]
    xs = [x]
    gate_gradients = True
    aggregation_method = "median"
    name = "hessians_8"
    input_dict = {"ys": [np.float64(y.numpy())], "xs": [x.numpy()], "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x1 = tf.constant([-1.0, -2.0])
    x2 = tf.constant([-3.0, -4.0])
    y = tf.reduce_sum(x1**2 + x2**3)
    ys = [y]
    xs = [x1, x2]
    gate_gradients = False
    aggregation_method = None
    name = "hessians_9"
    input_dict = {"ys": [np.float64(y.numpy())], "xs": [x1.numpy(), x2.numpy()], "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = tf.constant([[-1.0, -2.0], [-3.0, -4.0]])
    y1 = tf.reduce_sum(x[0]**2)
    y2 = tf.reduce_sum(x[1]**3)
    ys = [y1, y2]
    xs = [x]
    gate_gradients = True
    aggregation_method = "experimental_tree"
    name = "hessians_10"
    input_dict = {"ys": [np.float64(y1.numpy()), np.float64(y2.numpy())], "xs": [x.numpy()], "gate_gradients": gate_gradients, "aggregation_method": aggregation_method, "name": name}
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
