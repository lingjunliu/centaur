
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_hessians_inputs():
    list_of_inputs = []

    # Input 1
    ys = [tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))]
    xs = [tf.constant(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))]
    gate_gradients = False
    aggregation_method = None
    name = 'hessians_1'
    input_dict = {'ys': ys, 'xs': xs, 'gate_gradients': gate_gradients, 'aggregation_method': aggregation_method, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ys = [tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))]
    xs = [tf.constant(np.array([4.0, 5.0, 6.0], dtype=np.float32))]
    gate_gradients = True
    aggregation_method = "sum"
    name = 'hessians_2'
    input_dict = {'ys': ys, 'xs': xs, 'gate_gradients': gate_gradients, 'aggregation_method': aggregation_method, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ys = [tf.constant(np.array(1.0, dtype=np.float32))]
    xs = [tf.constant(np.array(2.0, dtype=np.float32))]
    gate_gradients = False
    aggregation_method = "tree"
    name = 'hessians_3'
    input_dict = {'ys': ys, 'xs': xs, 'gate_gradients': gate_gradients, 'aggregation_method': aggregation_method, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ys = [tf.constant(np.array([[1.0]], dtype=np.float32))]
    xs = [tf.constant(np.array([[2.0]], dtype=np.float32))]
    gate_gradients = True
    aggregation_method = "experimental_tree"
    name = 'hessians_4'
    input_dict = {'ys': ys, 'xs': xs, 'gate_gradients': gate_gradients, 'aggregation_method': aggregation_method, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    ys = [tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32))]
    xs = [tf.constant(np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]], dtype=np.float32))]
    gate_gradients = False
    aggregation_method = None
    name = 'hessians_5'
    input_dict = {'ys': ys, 'xs': xs, 'gate_gradients': gate_gradients, 'aggregation_method': aggregation_method, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ys = [tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)), tf.constant(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))]
    xs = [tf.constant(np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32))]
    gate_gradients = True
    aggregation_method = "sum"
    name = 'hessians_6'
    input_dict = {'ys': ys, 'xs': xs, 'gate_gradients': gate_gradients, 'aggregation_method': aggregation_method, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    ys = [tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))]
    xs = [tf.constant(np.array([4.0, 5.0, 6.0], dtype=np.float32)), tf.constant(np.array([7.0, 8.0, 9.0], dtype=np.float32))]
    gate_gradients = False
    aggregation_method = "tree"
    name = 'hessians_7'
    input_dict = {'ys': ys, 'xs': xs, 'gate_gradients': gate_gradients, 'aggregation_method': aggregation_method, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    ys = [tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))]
    xs = [tf.constant(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))]
    gate_gradients = True
    aggregation_method = "experimental_tree"
    name = 'hessians_8'
    input_dict = {'ys': ys, 'xs': xs, 'gate_gradients': gate_gradients, 'aggregation_method': aggregation_method, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    ys = [tf.constant(np.array(1.0, dtype=np.float32)), tf.constant(np.array(2.0, dtype=np.float32))]
    xs = [tf.constant(np.array(3.0, dtype=np.float32)), tf.constant(np.array(4.0, dtype=np.float32))]
    gate_gradients = False
    aggregation_method = None
    name = 'hessians_9'
    input_dict = {'ys': ys, 'xs': xs, 'gate_gradients': gate_gradients, 'aggregation_method': aggregation_method, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - Modified inputs to potentially avoid the error
    ys = [tf.constant(np.array([1.0, 2.0], dtype=np.float32))]
    xs = [tf.constant(np.array([3.0, 4.0], dtype=np.float32))]
    gate_gradients = True
    aggregation_method = "sum"
    name = 'hessians_10'
    input_dict = {'ys': ys, 'xs': xs, 'gate_gradients': gate_gradients, 'aggregation_method': aggregation_method, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.hessians"] = tf_hessians_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.hessians' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.hessians'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.hessians', generated_inputs['tf.hessians'], lib="tf", suffix=0)
