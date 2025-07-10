
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_softmax_inputs():
    list_of_inputs = []

    # Input 1: Basic case, axis=-1
    logits = np.array([-1, 0., 1.], dtype=np.float32)
    axis = -1
    name = "softmax_1"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, axis=1
    logits = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis = 1
    name = "softmax_2"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor, axis=0
    logits = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis = 0
    name = "softmax_3"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor, axis=2
    logits = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axis = 2
    name = "softmax_4"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor, axis=0
    logits = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axis = 0
    name = "softmax_5"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values and axis=-1
    logits = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.float32)
    axis = -1
    name = "softmax_6"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: different float type
    logits = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.float64)
    axis = -1
    name = "softmax_7"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: axis beyond first dimension
    logits = np.array([1.,2.], dtype=np.float32)
    axis = 0
    name = "softmax_8"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D tensor, axis=3
    logits = np.random.rand(2, 3, 4, 5).astype(np.float32)
    axis = 3
    name = "softmax_9"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: half type
    logits = np.array([1.,2.], dtype=np.float16)
    axis = 0
    name = "softmax_10"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.softmax"] = tf_nn_softmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.softmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.softmax'.")

check_valid('tf.nn.softmax', generated_inputs['tf.nn.softmax'], lib="tf", suffix=0)
