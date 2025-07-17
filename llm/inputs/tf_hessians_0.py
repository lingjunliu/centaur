
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_hessians_inputs():
    list_of_inputs = []

    # Input 1
    y1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    x1 = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    gate_gradients1 = False
    aggregation_method1 = None
    name1 = "hessian_test1"
    input_dict1 = {"ys": [y1], "xs": [x1], "gate_gradients": gate_gradients1, "aggregation_method": aggregation_method1, "name": name1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    y2 = tf.constant([1.0, 2.0, 3.0])
    x2 = tf.constant([4.0, 5.0, 6.0])
    gate_gradients2 = True
    aggregation_method2 = None
    name2 = "hessian_test2"
    input_dict2 = {"ys": [y2], "xs": [x2], "gate_gradients": gate_gradients2, "aggregation_method": aggregation_method2, "name": name2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    y3 = tf.constant([[1.0]])
    x3 = tf.constant([[2.0]])
    gate_gradients3 = False
    aggregation_method3 = None
    name3 = "hessian_test3"
    input_dict3 = {"ys": [y3], "xs": [x3], "gate_gradients": gate_gradients3, "aggregation_method": aggregation_method3, "name": name3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4
    y4 = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    x4 = tf.constant([4.0, 5.0, 6.0], dtype=tf.float32)
    gate_gradients4 = True
    aggregation_method4 = None
    name4 = "hessian_test4"
    input_dict4 = {"ys": [y4], "xs": [x4], "gate_gradients": gate_gradients4, "aggregation_method": aggregation_method4, "name": name4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    y5 = tf.constant([[-1.0, 2.0], [3.0, -4.0]], dtype=tf.float32)
    x5 = tf.constant([[5.0, -6.0], [-7.0, 8.0]], dtype=tf.float32)
    gate_gradients5 = False
    aggregation_method5 = None
    name5 = "hessian_test5"
    input_dict5 = {"ys": [y5], "xs": [x5], "gate_gradients": gate_gradients5, "aggregation_method": aggregation_method5, "name": name5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    y6 = tf.constant([1.0, 2.0], dtype=tf.float64)
    x6 = tf.constant([3.0, 4.0], dtype=tf.float64)
    gate_gradients6 = True
    aggregation_method6 = None
    name6 = "hessian_test6"
    input_dict6 = {"ys": [y6], "xs": [x6], "gate_gradients": gate_gradients6, "aggregation_method": aggregation_method6, "name": name6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Multiple ys
    y7_1 = tf.constant([1.0, 2.0], dtype=tf.float32)
    y7_2 = tf.constant([3.0, 4.0], dtype=tf.float32)
    x7 = tf.constant([5.0, 6.0], dtype=tf.float32)
    gate_gradients7 = False
    aggregation_method7 = None
    name7 = "hessian_test7"
    input_dict7 = {"ys": [y7_1, y7_2], "xs": [x7], "gate_gradients": gate_gradients7, "aggregation_method": aggregation_method7, "name": name7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Multiple xs
    y8 = tf.constant([1.0, 2.0], dtype=tf.float32)
    x8_1 = tf.constant([3.0, 4.0], dtype=tf.float32)
    x8_2 = tf.constant([5.0, 6.0], dtype=tf.float32)
    gate_gradients8 = True
    aggregation_method8 = None
    name8 = "hessian_test8"
    input_dict8 = {"ys": [y8], "xs": [x8_1, x8_2], "gate_gradients": gate_gradients8, "aggregation_method": aggregation_method8, "name": name8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: ys and xs both lists
    y9_1 = tf.constant([1.0, 2.0], dtype=tf.float32)
    y9_2 = tf.constant([3.0, 4.0], dtype=tf.float32)
    x9_1 = tf.constant([5.0, 6.0], dtype=tf.float32)
    x9_2 = tf.constant([7.0, 8.0], dtype=tf.float32)
    gate_gradients9 = False
    aggregation_method9 = None
    name9 = "hessian_test9"
    input_dict9 = {"ys": [y9_1, y9_2], "xs": [x9_1, x9_2], "gate_gradients": gate_gradients9, "aggregation_method": aggregation_method9, "name": name9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10: ys and xs both lists, different shapes
    y10_1 = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    y10_2 = tf.constant([5.0, 6.0], dtype=tf.float32)
    x10_1 = tf.constant([7.0, 8.0], dtype=tf.float32)
    x10_2 = tf.constant([[9.0, 10.0], [11.0, 12.0]], dtype=tf.float32)
    gate_gradients10 = True
    aggregation_method10 = None
    name10 = "hessian_test10"
    input_dict10 = {"ys": [y10_1, y10_2], "xs": [x10_1, x10_2], "gate_gradients": gate_gradients10, "aggregation_method": aggregation_method10, "name": name10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

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
