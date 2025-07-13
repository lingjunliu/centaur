
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_mlir_experimental_convert_function_inputs():
    list_of_inputs = []

    # Helper function to create a tf.function
    def create_tf_function():
        @tf.function
        def my_func(a, b):
            return a + b
        return my_func.get_concrete_function(
            tf.TensorSpec(None, tf.float32),
            tf.TensorSpec(None, tf.float32)
        )

    # Input 1
    concrete_function = create_tf_function()
    input_dict = {
        "concrete_function": concrete_function,
        "pass_pipeline": "tf-standard-pipeline",
        "show_debug_info": False
    }
    list_of_inputs.append(input_dict)

    # Input 2
    concrete_function = create_tf_function()
    input_dict = {
        "concrete_function": concrete_function,
        "pass_pipeline": "tf-standard-pipeline",
        "show_debug_info": True
    }
    list_of_inputs.append(input_dict)

    # Input 3
    concrete_function = create_tf_function()
    input_dict = {
        "concrete_function": concrete_function,
        "pass_pipeline": "",
        "show_debug_info": False
    }
    list_of_inputs.append(input_dict)

    # Input 4
    concrete_function = create_tf_function()
    input_dict = {
        "concrete_function": concrete_function,
        "pass_pipeline": "",
        "show_debug_info": True
    }
    list_of_inputs.append(input_dict)

    # Input 5
    concrete_function = create_tf_function()
    input_dict = {
        "concrete_function": concrete_function,
        "pass_pipeline": "tf-standard-pipeline",
        "show_debug_info": False
    }
    list_of_inputs.append(input_dict)

    # Input 6
    concrete_function = create_tf_function()
    input_dict = {
        "concrete_function": concrete_function,
        "pass_pipeline": "tf-standard-pipeline",
        "show_debug_info": True
    }
    list_of_inputs.append(input_dict)

    # Input 7
    concrete_function = create_tf_function()
    input_dict = {
        "concrete_function": concrete_function,
        "pass_pipeline": "",
        "show_debug_info": False
    }
    list_of_inputs.append(input_dict)

    # Input 8
    concrete_function = create_tf_function()
    input_dict = {
        "concrete_function": concrete_function,
        "pass_pipeline": "",
        "show_debug_info": True
    }
    list_of_inputs.append(input_dict)

    # Input 9
    concrete_function = create_tf_function()
    input_dict = {
        "concrete_function": concrete_function,
        "pass_pipeline": "tf-standard-pipeline",
        "show_debug_info": False
    }
    list_of_inputs.append(input_dict)

    # Input 10
    concrete_function = create_tf_function()
    input_dict = {
        "concrete_function": concrete_function,
        "pass_pipeline": "tf-standard-pipeline",
        "show_debug_info": True
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.mlir.experimental.convert_function"] = tf_mlir_experimental_convert_function_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.mlir.experimental.convert_function' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.mlir.experimental.convert_function'.")

check_valid('tf.mlir.experimental.convert_function', generated_inputs['tf.mlir.experimental.convert_function'], lib="tf", suffix=0)
