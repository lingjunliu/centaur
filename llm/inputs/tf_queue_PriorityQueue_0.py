
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_queue_priority_queue_inputs():
    list_of_inputs = []

    # Input 1
    capacity = 5
    types = [np.int32]
    shapes = [()]
    names = [None]
    shared_name = None
    name = None
    input_dict = {'capacity': capacity, 'types': types.copy(), 'shapes': shapes.copy(), 'names': names.copy(), 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    capacity = 10
    types = [np.float32, np.int64]
    shapes = [(), (2,)]
    names = [None, None]
    shared_name = None
    name = None
    input_dict = {'capacity': capacity, 'types': types.copy(), 'shapes': shapes.copy(), 'names': names.copy(), 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    capacity = 2
    types = [np.string_]
    shapes = [(3, 4)]
    names = [None]
    shared_name = None
    name = None
    input_dict = {'capacity': capacity, 'types': types.copy(), 'shapes': shapes.copy(), 'names': names.copy(), 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    capacity = 7
    types = [np.bool_]
    shapes = [()]
    names = [None]
    shared_name = None
    name = None
    input_dict = {'capacity': capacity, 'types': types.copy(), 'shapes': shapes.copy(), 'names': names.copy(), 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    capacity = 3
    types = [np.int32, np.float64]
    shapes = [(), (2,2)]
    names = [None, None]
    shared_name = None
    name = None
    input_dict = {'capacity': capacity, 'types': types.copy(), 'shapes': shapes.copy(), 'names': names.copy(), 'shared_name': shared_name, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.queue.PriorityQueue"] = tf_queue_priority_queue_inputs()

import sys

def check_valid(api, input_list, lib="tf", suffix=0):
    from fuzzingbook.GrammarFuzzer import GrammarFuzzer
    import inspect
    from api_extractor import get_signature
    from input_generators import get_abstract_input, generate_values, adapt_type
    # from models import get_model
    import numpy as np
    # from config import CHECK_ALL, API_CONFIG
    from tensorflow.python.framework import dtypes
    from inspect import signature

    def get_arg_type(arg):
      if arg in dtypes.DType.__members__:
        return dtypes.DType.__members__[arg]
      else:
        return arg

    def get_tf_dtype(dtype):
      if dtype == "int":
        return tf.int32
      elif dtype == "float":
        return tf.float32
      elif dtype == "string":
        return tf.string
      elif dtype == "double":
        return tf.float64
      elif dtype == "bool":
        return tf.bool
      elif dtype == 'int8':
        return tf.int8
      elif dtype == 'int16':
        return tf.int16
      elif dtype == 'int32':
        return tf.int32
      elif dtype == 'int64':
        return tf.int64
      elif dtype == 'uint8':
        return tf.uint8
      elif dtype == 'uint16':
        return tf.uint16
      elif dtype == 'uint32':
        return tf.uint32
      elif dtype == 'uint64':
        return tf.uint64
      elif dtype == 'float16':
        return tf.float16
      elif dtype == 'float32':
        return tf.float32
      elif dtype == 'float64':
        return tf.float64
      elif dtype == 'complex64':
        return tf.complex64
      elif dtype == 'complex128':
        return tf.complex128
      else:
        return dtype


    for input_dict in input_list:
        # print("current api {}".format(api))
        # print("current input dict {}".format(input_dict))
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.queue.PriorityQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.PriorityQueue'.")

check_valid('tf.queue.PriorityQueue', generated_inputs['tf.queue.PriorityQueue'], lib="tf", suffix=0)
