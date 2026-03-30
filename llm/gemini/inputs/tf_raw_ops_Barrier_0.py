
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_barrier_inputs():
    list_of_inputs = []

    # Input 1
    component_types = [tf.float32.as_numpy_dtype]
    shapes = [tf.TensorShape([1, 2]) if tf.TensorShape([1,2]).rank else []]
    capacity = 10
    container = "container1"
    shared_name = "shared1"
    name = "barrier1"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    component_types = [tf.int32.as_numpy_dtype, tf.float64.as_numpy_dtype]
    shapes = [tf.TensorShape([1]) if tf.TensorShape([1]).rank else [], tf.TensorShape([1, 3]) if tf.TensorShape([1,3]).rank else []]
    capacity = -1
    container = ""
    shared_name = ""
    name = "barrier2"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    component_types = [tf.string.as_numpy_dtype]
    shapes = [tf.TensorShape([1, 1, 1]) if tf.TensorShape([1,1,1]).rank else []]
    capacity = 1
    container = "container3"
    shared_name = "shared3"
    name = None

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    component_types = [tf.bool.as_numpy_dtype, tf.complex64.as_numpy_dtype, tf.int8.as_numpy_dtype]
    shapes = [tf.TensorShape([1, 4]) if tf.TensorShape([1,4]).rank else [], tf.TensorShape([1]) if tf.TensorShape([1]).rank else [], tf.TensorShape([1, 2, 3]) if tf.TensorShape([1,2,3]).rank else []]
    capacity = 100
    container = ""
    shared_name = "shared4"
    name = "barrier4"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    component_types = [tf.uint16.as_numpy_dtype]
    shapes = []
    capacity = -1
    container = "container5"
    shared_name = ""
    name = "barrier5"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    component_types = [tf.qint8.as_numpy_dtype]
    shapes = [tf.TensorShape([1,5,5]) if tf.TensorShape([1,5,5]).rank else []]
    capacity = 2
    container = "container6"
    shared_name = "shared6"
    name = "barrier6"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    component_types = [tf.int64.as_numpy_dtype]
    shapes = [tf.TensorShape([1]) if tf.TensorShape([1]).rank else []]
    capacity = 1
    container = ""
    shared_name = ""
    name = "barrier7"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    component_types = [tf.float64.as_numpy_dtype]
    shapes = [tf.TensorShape([1,1,1,1]) if tf.TensorShape([1,1,1,1]).rank else []]
    capacity = 200
    container = "container8"
    shared_name = "shared8"
    name = None

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    component_types = [tf.float16.as_numpy_dtype]
    shapes = [tf.TensorShape([1,2,3,4,5]) if tf.TensorShape([1,2,3,4,5]).rank else []]
    capacity = 5
    container = ""
    shared_name = "shared9"
    name = "barrier9"

    input_dict = {
        "component_types": component_types,
        "shapes": shapes,
        "capacity": capacity,
        "container": container,
        "shared_name": shared_name,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Barrier"] = tf_raw_ops_barrier_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Barrier' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Barrier'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Barrier', generated_inputs['tf.raw_ops.Barrier'], lib="tf", suffix=0)
