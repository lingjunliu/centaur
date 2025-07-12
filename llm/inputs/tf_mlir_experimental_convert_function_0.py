
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_mlir_experimental_convert_function_inputs():
    list_of_inputs = []

    # Helper function to serialize concrete functions
    def serialize_concrete_function(concrete_function):
        return concrete_function.name

    # Input 1
    concrete_function = tf.function(lambda x: x + 1).get_concrete_function(tf.TensorSpec(None, tf.float32))
    input_dict = {
        'concrete_function': serialize_concrete_function(concrete_function),
        'pass_pipeline': 'tf-standard-pipeline',
        'show_debug_info': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    concrete_function = tf.function(lambda x: x * 2).get_concrete_function(tf.TensorSpec(None, tf.int32))
    input_dict = {
        'concrete_function': serialize_concrete_function(concrete_function),
        'pass_pipeline': '',
        'show_debug_info': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    concrete_function = tf.function(lambda x, y: tf.matmul(x, y)).get_concrete_function(tf.TensorSpec([None, 2], tf.float32), tf.TensorSpec([2, None], tf.float32))
    input_dict = {
        'concrete_function': serialize_concrete_function(concrete_function),
        'pass_pipeline': 'func.func(tf-lower-graph)',
        'show_debug_info': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    concrete_function = tf.function(lambda x: tf.nn.relu(x)).get_concrete_function(tf.TensorSpec(None, tf.float32))
    input_dict = {
        'concrete_function': serialize_concrete_function(concrete_function),
        'pass_pipeline': 'tf-opt',
        'show_debug_info': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    concrete_function = tf.function(lambda x: tf.sin(x)).get_concrete_function(tf.TensorSpec(None, tf.float32))
    input_dict = {
        'concrete_function': serialize_concrete_function(concrete_function),
        'pass_pipeline': 'cse',
        'show_debug_info': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    concrete_function = tf.function(lambda x: tf.cos(x)).get_concrete_function(tf.TensorSpec(None, tf.float32))
    input_dict = {
        'concrete_function': serialize_concrete_function(concrete_function),
        'pass_pipeline': 'canonicalize',
        'show_debug_info': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    concrete_function = tf.function(lambda x: tf.sqrt(x)).get_concrete_function(tf.TensorSpec(None, tf.float32))
    input_dict = {
        'concrete_function': serialize_concrete_function(concrete_function),
        'pass_pipeline': 'tf-shape-inference',
        'show_debug_info': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    concrete_function = tf.function(lambda x: tf.math.exp(x)).get_concrete_function(tf.TensorSpec(None, tf.float32))
    input_dict = {
        'concrete_function': serialize_concrete_function(concrete_function),
        'pass_pipeline': 'eliminate-dead-code',
        'show_debug_info': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    concrete_function = tf.function(lambda x: tf.reshape(x, [1, -1])).get_concrete_function(tf.TensorSpec([None], tf.float32))
    input_dict = {
        'concrete_function': serialize_concrete_function(concrete_function),
        'pass_pipeline': 'inliner',
        'show_debug_info': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    concrete_function = tf.function(lambda x: tf.cast(x, tf.int32)).get_concrete_function(tf.TensorSpec(None, tf.float32))
    input_dict = {
        'concrete_function': serialize_concrete_function(concrete_function),
        'pass_pipeline': 'loop-fusion',
        'show_debug_info': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.mlir.experimental.convert_function"] = tf_mlir_experimental_convert_function_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.mlir.experimental.convert_function' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.mlir.experimental.convert_function'.")

check_valid('tf.mlir.experimental.convert_function', generated_inputs['tf.mlir.experimental.convert_function'], lib="tf", suffix=0)
