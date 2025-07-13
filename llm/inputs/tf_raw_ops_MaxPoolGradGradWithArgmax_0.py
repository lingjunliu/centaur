
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPoolGradGradWithArgmax_inputs():
    list_of_inputs = []

    # Input 1
    input_val = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    grad_val = np.array([[[[0.1, 0.2]]]], dtype=np.float32)
    argmax_val = np.array([[[[0, 1]]]], dtype=np.int64)
    ksize_val = [1, 1, 1, 1]
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    include_batch_in_index_val = False
    name_val = "test1"

    input_dict = {
        "input": input_val,
        "grad": grad_val,
        "argmax": argmax_val,
        "ksize": ksize_val,
        "strides": strides_val,
        "padding": padding_val,
        "include_batch_in_index": include_batch_in_index_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_val = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    grad_val = np.array([[[[0.1, 0.2]]]], dtype=np.float32)
    argmax_val = np.array([[[[0, 1]]]], dtype=np.int32)
    ksize_val = [1, 2, 2, 1]
    strides_val = [1, 1, 1, 1]
    padding_val = "SAME"
    include_batch_in_index_val = True
    name_val = "test2"

    input_dict = {
        "input": input_val,
        "grad": grad_val,
        "argmax": argmax_val,
        "ksize": ksize_val,
        "strides": strides_val,
        "padding": padding_val,
        "include_batch_in_index": include_batch_in_index_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_val = np.array([[[[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]]], dtype=np.float64)
    grad_val = np.array([[[[0.1, 0.2, 0.3]]]], dtype=np.float64)
    argmax_val = np.array([[[[0, 1, 2]]]], dtype=np.int64)
    ksize_val = [1, 1, 3, 1]
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    include_batch_in_index_val = False
    name_val = "test3"

    input_dict = {
        "input": input_val,
        "grad": grad_val,
        "argmax": argmax_val,
        "ksize": ksize_val,
        "strides": strides_val,
        "padding": padding_val,
        "include_batch_in_index": include_batch_in_index_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_val = np.array([[[[1, 2], [3, 4]]]], dtype=np.int32)
    grad_val = np.array([[[[0, 1]]]], dtype=np.int32)
    argmax_val = np.array([[[[0, 1]]]], dtype=np.int32)
    ksize_val = [1, 2, 2, 1]
    strides_val = [1, 1, 1, 1]
    padding_val = "SAME"
    include_batch_in_index_val = True
    name_val = "test4"

    input_dict = {
        "input": input_val,
        "grad": grad_val,
        "argmax": argmax_val,
        "ksize": ksize_val,
        "strides": strides_val,
        "padding": padding_val,
        "include_batch_in_index": include_batch_in_index_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    input_val = np.array([[[[1, 2], [3, 4]]]], dtype=np.int64)
    grad_val = np.array([[[[0, 1]]]], dtype=np.int64)
    argmax_val = np.array([[[[0, 1]]]], dtype=np.int64)
    ksize_val = [1, 2, 2, 1]
    strides_val = [1, 1, 1, 1]
    padding_val = "SAME"
    include_batch_in_index_val = True
    name_val = "test5"

    input_dict = {
        "input": input_val,
        "grad": grad_val,
        "argmax": argmax_val,
        "ksize": ksize_val,
        "strides": strides_val,
        "padding": padding_val,
        "include_batch_in_index": include_batch_in_index_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_val = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float16)
    grad_val = np.array([[[[0.1, 0.2]]]], dtype=np.float16)
    argmax_val = np.array([[[[0, 1]]]], dtype=np.int32)
    ksize_val = [1, 1, 1, 1]
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    include_batch_in_index_val = False
    name_val = "test6"

    input_dict = {
        "input": input_val,
        "grad": grad_val,
        "argmax": argmax_val,
        "ksize": ksize_val,
        "strides": strides_val,
        "padding": padding_val,
        "include_batch_in_index": include_batch_in_index_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_val = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float16)
    grad_val = np.array([[[[0.1, 0.2]]]], dtype=np.float16)
    argmax_val = np.array([[[[0, 1]]]], dtype=np.int32)
    ksize_val = [1, 1, 1, 1]
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    include_batch_in_index_val = False
    name_val = "test7"

    input_dict = {
        "input": input_val,
        "grad": grad_val,
        "argmax": argmax_val,
        "ksize": ksize_val,
        "strides": strides_val,
        "padding": padding_val,
        "include_batch_in_index": include_batch_in_index_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_val = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    grad_val = np.array([[[[0.1, 0.2]]]], dtype=np.float32)
    argmax_val = np.array([[[[0, 1]]]], dtype=np.int64)
    ksize_val = [1, 1, 2, 1]
    strides_val = [1, 1, 1, 1]
    padding_val = "SAME"
    include_batch_in_index_val = True
    name_val = "test8"

    input_dict = {
        "input": input_val,
        "grad": grad_val,
        "argmax": argmax_val,
        "ksize": ksize_val,
        "strides": strides_val,
        "padding": padding_val,
        "include_batch_in_index": include_batch_in_index_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_val = np.array([[[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
    grad_val = np.array([[[[0.1, 0.2]]], [[[0.3, 0.4]]]] , dtype=np.float32)
    argmax_val = np.array([[[[0, 1]]], [[[0, 1]]]] , dtype=np.int64)
    ksize_val = [1, 1, 1, 1]
    strides_val = [1, 1, 1, 1]
    padding_val = "VALID"
    include_batch_in_index_val = False
    name_val = "test9"

    input_dict = {
        "input": input_val,
        "grad": grad_val,
        "argmax": argmax_val,
        "ksize": ksize_val,
        "strides": strides_val,
        "padding": padding_val,
        "include_batch_in_index": include_batch_in_index_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_val = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32)
    grad_val = np.array([[[[0.1, 0.2, 0.3]]]], dtype=np.float32)
    argmax_val = np.array([[[[0, 1, 2]]]], dtype=np.int32)
    ksize_val = [1, 2, 2, 1]
    strides_val = [1, 2, 2, 1]
    padding_val = "SAME"
    include_batch_in_index_val = True
    name_val = "test10"

    input_dict = {
        "input": input_val,
        "grad": grad_val,
        "argmax": argmax_val,
        "ksize": ksize_val,
        "strides": strides_val,
        "padding": padding_val,
        "include_batch_in_index": include_batch_in_index_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MaxPoolGradGradWithArgmax"] = tf_raw_ops_MaxPoolGradGradWithArgmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MaxPoolGradGradWithArgmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolGradGradWithArgmax'.")

check_valid('tf.raw_ops.MaxPoolGradGradWithArgmax', generated_inputs['tf.raw_ops.MaxPoolGradGradWithArgmax'], lib="tf", suffix=0)
