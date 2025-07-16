
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_gradients_inputs():
    list_of_inputs = []

    # Input 1
    ys = [tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))]
    xs = [tf.constant(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))]
    grad_ys = [tf.constant(np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32))]
    name = "gradients_1"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = []
    unconnected_gradients = 'none'

    input_dict = {
        "ys": ys,
        "xs": xs,
        "grad_ys": grad_ys,
        "name": name,
        "gate_gradients": gate_gradients,
        "aggregation_method": aggregation_method,
        "stop_gradients": stop_gradients,
        "unconnected_gradients": unconnected_gradients
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ys = [tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32), shape=(1,3))]
    xs = [tf.constant(np.array([4.0, 5.0, 6.0], dtype=np.float32), shape=(1,3))]
    grad_ys = [tf.constant(np.array([7.0, 8.0, 9.0], dtype=np.float32), shape=(1,3))]
    name = "gradients_2"
    gate_gradients = True
    aggregation_method = None
    stop_gradients = []
    unconnected_gradients = 'zero'

    input_dict = {
        "ys": ys,
        "xs": xs,
        "grad_ys": grad_ys,
        "name": name,
        "gate_gradients": gate_gradients,
        "aggregation_method": aggregation_method,
        "stop_gradients": stop_gradients,
        "unconnected_gradients": unconnected_gradients
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ys = [tf.constant(np.array(1.0, dtype=np.float32))]
    xs = [tf.constant(np.array(2.0, dtype=np.float32))]
    grad_ys = [tf.constant(np.array(3.0, dtype=np.float32))]
    name = "gradients_3"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = []
    unconnected_gradients = 'none'

    input_dict = {
        "ys": ys,
        "xs": xs,
        "grad_ys": grad_ys,
        "name": name,
        "gate_gradients": gate_gradients,
        "aggregation_method": aggregation_method,
        "stop_gradients": stop_gradients,
        "unconnected_gradients": unconnected_gradients
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ys = [tf.constant(np.array([-1.0, -2.0], dtype=np.float32))]
    xs = [tf.constant(np.array([-3.0, -4.0], dtype=np.float32))]
    grad_ys = [tf.constant(np.array([-5.0, -6.0], dtype=np.float32))]
    name = "gradients_4"
    gate_gradients = True
    aggregation_method = None
    stop_gradients = []
    unconnected_gradients = 'zero'

    input_dict = {
        "ys": ys,
        "xs": xs,
        "grad_ys": grad_ys,
        "name": name,
        "gate_gradients": gate_gradients,
        "aggregation_method": aggregation_method,
        "stop_gradients": stop_gradients,
        "unconnected_gradients": unconnected_gradients
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    ys = [tf.constant(np.array([[[1.0]]], dtype=np.float32))]
    xs = [tf.constant(np.array([[[2.0]]], dtype=np.float32))]
    grad_ys = [tf.constant(np.array([[[3.0]]], dtype=np.float32))]
    name = "gradients_5"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = []
    unconnected_gradients = 'none'

    input_dict = {
        "ys": ys,
        "xs": xs,
        "grad_ys": grad_ys,
        "name": name,
        "gate_gradients": gate_gradients,
        "aggregation_method": aggregation_method,
        "stop_gradients": stop_gradients,
        "unconnected_gradients": unconnected_gradients
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ys = [tf.constant(np.array([1.0, 2.0], dtype=np.float32)), tf.constant(np.array([3.0, 4.0], dtype=np.float32))]
    xs = [tf.constant(np.array([5.0, 6.0], dtype=np.float32)), tf.constant(np.array([7.0, 8.0], dtype=np.float32))]
    grad_ys = [tf.constant(np.array([9.0, 10.0], dtype=np.float32)), tf.constant(np.array([11.0, 12.0], dtype=np.float32))]
    name = "gradients_6"
    gate_gradients = True
    aggregation_method = None
    stop_gradients = []
    unconnected_gradients = 'zero'

    input_dict = {
        "ys": ys,
        "xs": xs,
        "grad_ys": grad_ys,
        "name": name,
        "gate_gradients": gate_gradients,
        "aggregation_method": aggregation_method,
        "stop_gradients": stop_gradients,
        "unconnected_gradients": unconnected_gradients
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    ys = [tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))]
    xs = [tf.constant(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))]
    grad_ys = [tf.constant(np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32))]
    name = "gradients_7"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = [tf.constant(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))]
    unconnected_gradients = 'none'

    input_dict = {
        "ys": ys,
        "xs": xs,
        "grad_ys": grad_ys,
        "name": name,
        "gate_gradients": gate_gradients,
        "aggregation_method": aggregation_method,
        "stop_gradients": [tf.constant(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))],
        "unconnected_gradients": unconnected_gradients
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    ys = [tf.constant(np.array([1.0], dtype=np.float32), shape=(1,1))]
    xs = [tf.constant(np.array([2.0], dtype=np.float32), shape=(1,1))]
    grad_ys = [tf.constant(np.array([3.0], dtype=np.float32), shape=(1,1))]
    name = "gradients_8"
    gate_gradients = True
    aggregation_method = None
    stop_gradients = [tf.constant(np.array([2.0], dtype=np.float32), shape=(1,1))]
    unconnected_gradients = 'zero'

    input_dict = {
        "ys": ys,
        "xs": xs,
        "grad_ys": grad_ys,
        "name": name,
        "gate_gradients": gate_gradients,
        "aggregation_method": aggregation_method,
        "stop_gradients": [tf.constant(np.array([2.0], dtype=np.float32), shape=(1,1))],
        "unconnected_gradients": unconnected_gradients
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9
    ys = [tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),tf.constant(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))]
    xs = [tf.constant(np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32)),tf.constant(np.array([[13.0, 14.0], [15.0, 16.0]], dtype=np.float32))]
    grad_ys = [tf.constant(np.array([[17.0, 18.0], [19.0, 20.0]], dtype=np.float32)),tf.constant(np.array([[21.0, 22.0], [23.0, 24.0]], dtype=np.float32))]
    name = "gradients_9"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = [tf.constant(np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32))]
    unconnected_gradients = 'none'

    input_dict = {
        "ys": ys,
        "xs": xs,
        "grad_ys": grad_ys,
        "name": name,
        "gate_gradients": gate_gradients,
        "aggregation_method": aggregation_method,
        "stop_gradients": [tf.constant(np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32))],
        "unconnected_gradients": unconnected_gradients
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    ys = [tf.constant(np.array(1.0, dtype=np.float32))]
    xs = [tf.constant(np.array(2.0, dtype=np.float32))]
    grad_ys = [tf.constant(np.array(1.0, dtype=np.float32))]
    name = "gradients_10"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = []
    unconnected_gradients = 'none'

    input_dict = {
        "ys": ys,
        "xs": xs,
        "grad_ys": grad_ys,
        "name": name,
        "gate_gradients": gate_gradients,
        "aggregation_method": aggregation_method,
        "stop_gradients": stop_gradients,
        "unconnected_gradients": unconnected_gradients
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    ys = [tf.constant(np.array([1.0, 2.0], dtype=np.float32))]
    xs = [tf.constant(np.array([3.0, 4.0], dtype=np.float32)), tf.constant(np.array([5.0, 6.0], dtype=np.float32))]
    grad_ys = [tf.constant(np.array([7.0, 8.0], dtype=np.float32))]
    name = "gradients_11"
    gate_gradients = False
    aggregation_method = None
    stop_gradients = []
    unconnected_gradients = 'none'

    input_dict = {
        "ys": ys,
        "xs": xs,
        "grad_ys": grad_ys,
        "name": name,
        "gate_gradients": gate_gradients,
        "aggregation_method": aggregation_method,
        "stop_gradients": stop_gradients,
        "unconnected_gradients": unconnected_gradients
    }
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
