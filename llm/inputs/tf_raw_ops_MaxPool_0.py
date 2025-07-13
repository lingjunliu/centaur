
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_maxpool_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    name = None

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(1, 7, 7, 1).astype(np.float32)
    ksize = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
    padding = "SAME"
    explicit_paddings = []
    data_format = "NHWC"
    name = "max_pool_2"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(2, 10, 10, 5).astype(np.float32)
    ksize = [1, 4, 4, 1]
    strides = [1, 3, 3, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    name = "max_pool_3"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: different data_format
    input_tensor = np.random.rand(2, 5, 10, 10).astype(np.float32)
    ksize = [1, 1, 2, 2]
    strides = [1, 1, 1, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NCHW"
    name = "max_pool_4"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5: int32 type
    input_tensor = np.random.randint(0, 10, size=(1, 8, 8, 3), dtype=np.int32)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    name = "max_pool_5"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: explicit paddings
    input_tensor = np.random.rand(1, 3, 3, 1).astype(np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "EXPLICIT"
    explicit_paddings = [0, 0, 1, 1, 0, 0, 0, 0]
    data_format = "NHWC"
    name = "max_pool_6"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: float64 type
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float64)
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    name = None

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint8 type
    input_tensor = np.random.randint(0, 255, size=(1, 8, 8, 3), dtype=np.uint8)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    name = "max_pool_8"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: qint8 type
    input_tensor = np.random.randint(-128, 127, size=(1, 8, 8, 3), dtype=np.int8)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    name = "max_pool_9"

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bfloat16 type
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float16)
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    name = None

    input_dict = {
        "input": input_tensor,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MaxPool"] = tf_raw_ops_maxpool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MaxPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPool'.")

check_valid('tf.raw_ops.MaxPool', generated_inputs['tf.raw_ops.MaxPool'], lib="tf", suffix=0)
