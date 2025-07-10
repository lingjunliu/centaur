
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_sigmoid_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 tensor
    x = tf.constant(np.array([0.0, 1.0, -1.0, 10.0, -10.0], dtype=np.float32))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 tensor
    x = tf.constant(np.array([0.0, 0.5, -0.5, 100.0, -100.0], dtype=np.float64))
    name = "sigmoid_example_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16 tensor
    x = tf.constant(np.array([0.0, 0.25, -0.25, 10.0, -10.0], dtype=np.float16))
    name = "sigmoid_example_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64 tensor
    x = tf.constant(np.array([1+1j, 2-2j, -1+0j, 0-1j], dtype=np.complex64))
    name = "sigmoid_example_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex128 tensor
    x = tf.constant(np.array([1.0+1.0j, 2.0-2.0j, -1.0+0.0j, 0.0-1.0j], dtype=np.complex128))
    name = "sigmoid_example_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional float32 tensor
    x = tf.constant(np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32))
    name = "sigmoid_example_6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large values for float32
    x = tf.constant(np.array([1e6, -1e6], dtype=np.float32))
    name = "sigmoid_example_7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small values for float32
    x = tf.constant(np.array([1e-6, -1e-6], dtype=np.float32))
    name = "sigmoid_example_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D tensor float32
    x = tf.constant(np.random.rand(3, 3, 3).astype(np.float32))
    name = "sigmoid_example_9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mix of positive and negative, float64
    x = tf.constant(np.array([-5.0, -2.0, 0.0, 2.0, 5.0], dtype=np.float64))
    name = "sigmoid_example_10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.sigmoid"] = tf_math_sigmoid_inputs()

for i in range(len(generated_inputs["tf.math.sigmoid"])):
    generated_inputs["tf.math.sigmoid"][i]["x"] = generated_inputs["tf.math.sigmoid"][i]["x"].numpy()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.sigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.sigmoid'.")

check_valid('tf.math.sigmoid', generated_inputs['tf.math.sigmoid'], lib="tf", suffix=0)
