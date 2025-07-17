
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_tuple_inputs():
    list_of_inputs = []

    # Input 1: Simple list of tensors, no control inputs, no name.
    tensors = [tf.constant([1, 2, 3]), tf.constant([4, 5, 6])]
    control_inputs = []
    name = None
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List with a single tensor, control input, name.
    tensors = [tf.constant(10.0)]
    op = tf.no_op()
    control_inputs = [op]
    name = "my_tuple"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List with None and a tensor. Removing None, as it causes issues further down the line
    tensors = [tf.constant([[1, 2], [3, 4]])]
    control_inputs = []
    name = None
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List of tensors with different shapes, control input list, name.
    tensors = [tf.constant(1), tf.constant([1, 2]), tf.constant([[1, 2], [3, 4]])]
    op1 = tf.no_op()
    op2 = tf.no_op()
    control_inputs = [op1, op2]
    name = "complex_tuple"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List of tensors with different dtypes.
    tensors = [tf.constant(1, dtype=tf.int32), tf.constant(2.0, dtype=tf.float32), tf.constant(True, dtype=tf.bool)]
    control_inputs = []
    name = None
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: List of tensors with negative values
    tensors = [tf.constant([-1, -2, -3]), tf.constant([-4.0, -5.0])]
    control_inputs = []
    name = None
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty control inputs, different name
    tensors = [tf.constant([1, 2, 3])]
    control_inputs = []
    name = "another_tuple"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: More complex tensors
    tensors = [tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), tf.constant([[9, 10], [11, 12]])]
    control_inputs = []
    name = None
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Control inputs as ops, not tensors - Changing this to comply with signature
    tensors = [tf.constant([1, 2])]
    op1 = tf.no_op()
    op2 = tf.no_op()
    control_inputs = [op1, op2]
    name = None
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
     # Input 10: More complex tensors, different control_inputs and name
    tensors = [tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), tf.constant([[9, 10], [11, 12]])]
    op1 = tf.no_op()
    control_inputs = [op1]
    name = "tuple_complex_control"
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Control inputs that are tensors - Removing this as the signature asks for Operations
    # tensors = [tf.constant([1, 2])]
    # control_inputs = [tf.constant(1), tf.constant(2)]
    # name = None
    # input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Indexed Slices tensor
    indices = tf.constant([0, 2, 4])
    values = tf.constant([[1, 2], [3, 4], [5, 6]])
    dense_shape = tf.constant([7, 2])
    indexed_slices = tf.IndexedSlices(values, indices, dense_shape)

    tensors = [indexed_slices, tf.constant([1, 2, 3])]
    control_inputs = []
    name = None
    input_dict = {"tensors": tensors, "control_inputs": control_inputs, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.tuple"] = tf_tuple_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.tuple' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.tuple'.")

check_valid('tf.tuple', generated_inputs['tf.tuple'], lib="tf", suffix=0)
