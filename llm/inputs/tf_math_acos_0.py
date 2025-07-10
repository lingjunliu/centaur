
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_acos_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    x = np.array([0.0, 0.5, 1.0, -0.5, -1.0], dtype=np.float32)
    name = "acos_example_1"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data type
    x = np.array([0.2, 0.8, -0.2, -0.8], dtype=np.float64)
    name = "acos_example_2"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional array
    x = np.array([[0.1, 0.9], [-0.3, -0.7]], dtype=np.float32)
    name = "acos_example_3"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Input close to the boundary
    x = np.array([0.99, -0.99], dtype=np.float32)
    name = "acos_example_4"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Input with zero
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = "acos_example_5"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger array
    x = np.linspace(-1, 1, 10, dtype=np.float32)
    name = "acos_example_6"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16
    x = np.array([0.25, -0.75, 0.5], dtype=np.float16)
    name = "acos_example_7"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger multi-dimensional array
    x = np.random.uniform(-1, 1, size=(3, 4, 5)).astype(np.float32)
    name = "acos_example_8"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half type
    x = np.array([0.6, -0.1], dtype=np.float16)
    name = "acos_example_9"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 with shape ()
    x = np.float32(0.5)
    name = "acos_example_10"
    input_dict = {"x": tf.constant(x).numpy(), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.acos"] = tf_math_acos_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.acos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.acos'.")

check_valid('tf.math.acos', generated_inputs['tf.math.acos'], lib="tf", suffix=0)
