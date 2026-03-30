
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_compat_dimension_at_index_inputs():
    list_of_inputs = []

    # Input 1: Simple case with a defined shape
    shape = tf.TensorShape([2, 3, 4])
    index = 1
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Shape with unknown dimension
    shape = tf.TensorShape([None, 5, 6])
    index = 0
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Shape with only one dimension
    shape = tf.TensorShape([7])
    index = 0
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Shape with all dimensions defined
    shape = tf.TensorShape([1, 2, 3, 4, 5])
    index = 3
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Shape with some dimensions unknown
    shape = tf.TensorShape([8, None, 10, None])
    index = 1
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Shape with a large index
    shape = tf.TensorShape([11, 12, 13, 14])
    index = 2
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Shape with a dimension of size 1
    shape = tf.TensorShape([1, 15, 1, 16])
    index = 1
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Shape with repeated dimensions
    shape = tf.TensorShape([17, 17, 17])
    index = 0
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Shape with a very small dimension size
    shape = tf.TensorShape([1, 1, 1, 1])
    index = 2
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Shape with None dimension as the last entry
    shape = tf.TensorShape([1,2,None])
    index = 2
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: index as numpy integer
    shape = tf.TensorShape([1,2,3])
    index = np.int32(1)
    input_dict = {"shape": shape, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.compat.dimension_at_index"] = tf_compat_dimension_at_index_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.compat.dimension_at_index' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.compat.dimension_at_index'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.compat.dimension_at_index', generated_inputs['tf.compat.dimension_at_index'], lib="tf", suffix=0)
