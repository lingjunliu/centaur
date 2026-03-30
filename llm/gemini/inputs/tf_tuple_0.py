
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_tuple_inputs():
    list_of_inputs = []

    # Input 1: Basic list of tensors
    tensors = [tf.constant([1, 2, 3]), tf.constant([4, 5, 6])]
    control_inputs = []
    name = "basic_tuple"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List with different shapes
    tensors = [tf.constant(1), tf.constant([2, 3]), tf.constant([[4, 5], [6, 7]])]
    control_inputs = []
    name = "diff_shapes"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With control inputs (empty op)
    tensors = [tf.constant([1.0, 2.0])]
    control_inputs = [tf.no_op()]
    name = "with_control"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  Multiple control inputs
    a = tf.constant(1.0)
    b = a + 2.0
    tensors = [a,b]
    control_inputs = [tf.no_op(), tf.no_op()]
    name = "multiple_control"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Empty name
    tensors = [tf.constant([4,5,6])]
    control_inputs = []
    name = ""
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different data types
    tensors = [tf.constant(1, dtype=tf.int32), tf.constant(2.0, dtype=tf.float32), tf.constant(True, dtype=tf.bool)]
    control_inputs = []
    name = "diff_dtypes"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Higher rank tensor
    tensors = [tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])]
    control_inputs = []
    name = "high_rank"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Tensor with negative values
    tensors = [tf.constant([-1, -2, -3])]
    control_inputs = []
    name = "negative_values"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.tuple"] = tf_tuple_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.tuple' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.tuple'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.tuple', generated_inputs['tf.tuple'], lib="tf", suffix=0)
