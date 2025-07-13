
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_priority_queue_inputs():
    list_of_inputs = []

    # Input 1
    shapes = []
    component_types = []
    capacity = -1
    container = ""
    shared_name = ""
    name = "priority_queue_1"
    input_dict = {"shapes": shapes, "component_types": component_types, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shapes = [tf.TensorShape([10])]
    component_types = [tf.float32.as_datatype_enum]
    capacity = 100
    container = "test_container"
    shared_name = "test_shared_name"
    name = "priority_queue_2"
    input_dict = {"shapes": shapes, "component_types": component_types, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shapes = [tf.TensorShape([None])]
    component_types = [tf.int32.as_datatype_enum]
    capacity = 50
    container = ""
    shared_name = "shared_queue"
    name = "priority_queue_3"
    input_dict = {"shapes": shapes, "component_types": component_types, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shapes = [tf.TensorShape([5, 5])]
    component_types = [tf.double.as_datatype_enum]
    capacity = -1
    container = "container_name"
    shared_name = ""
    name = "priority_queue_4"
    input_dict = {"shapes": shapes, "component_types": component_types, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shapes = [tf.TensorShape([2, 3, 4])]
    component_types = [tf.string.as_datatype_enum]
    capacity = 200
    container = ""
    shared_name = ""
    name = "priority_queue_5"
    input_dict = {"shapes": shapes, "component_types": component_types, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shapes = [tf.TensorShape([])]
    component_types = [tf.bool.as_datatype_enum]
    capacity = 10
    container = "abc"
    shared_name = "def"
    name = "priority_queue_6"
    input_dict = {"shapes": shapes, "component_types": component_types, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shapes = [tf.TensorShape([1, 1])]
    component_types = [tf.complex64.as_datatype_enum]
    capacity = 5
    container = ""
    shared_name = "ghi"
    name = "priority_queue_7"
    input_dict = {"shapes": shapes, "component_types": component_types, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shapes = [tf.TensorShape([1]), tf.TensorShape([10])]
    component_types = [tf.int64.as_datatype_enum, tf.float64.as_datatype_enum]
    capacity = 1
    container = "container1"
    shared_name = "shared1"
    name = "priority_queue_8"
    input_dict = {"shapes": shapes, "component_types": component_types, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shapes = [tf.TensorShape([None]), tf.TensorShape([5,5])]
    component_types = [tf.uint8.as_datatype_enum, tf.int16.as_datatype_enum]
    capacity = 1024
    container = "container2"
    shared_name = "shared2"
    name = "priority_queue_9"
    input_dict = {"shapes": shapes, "component_types": component_types, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shapes = [tf.TensorShape([2, 3, 4]), tf.TensorShape([])]
    component_types = [tf.qint8.as_datatype_enum, tf.quint8.as_datatype_enum]
    capacity = -1
    container = ""
    shared_name = ""
    name = "priority_queue_10"
    input_dict = {"shapes": shapes, "component_types": component_types, "capacity": capacity, "container": container, "shared_name": shared_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    generated_inputs["tf.raw_ops.PriorityQueue"] = list_of_inputs
    return list_of_inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.PriorityQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PriorityQueue'.")

check_valid('tf.raw_ops.PriorityQueue', generated_inputs['tf.raw_ops.PriorityQueue'], lib="tf", suffix=0)
