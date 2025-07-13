
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_register_tensor_conversion_function_inputs():
    list_of_inputs = []

    # Input 1: Custom class conversion
    class MyClass:
        def __init__(self, value):
            self.value = value

    def conversion_func_1(value, dtype=None, name=None, as_ref=False):
        return tf.convert_to_tensor(value.value, dtype=dtype, name=name)

    input_dict = {
        "base_type": (MyClass,),
        "conversion_func": [conversion_func_1],
        "priority": 50
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Another custom class
    class AnotherClass:
        def __init__(self, data):
            self.data = data

    def conversion_func_2(value, dtype=None, name=None, as_ref=False):
        return tf.convert_to_tensor(value.data, dtype=dtype, name=name)

    input_dict = {
        "base_type": (AnotherClass,),
        "conversion_func": [conversion_func_2],
        "priority": 75
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Custom class with specific dtype
    class YetAnotherClass:
        def __init__(self, value):
            self.value = value

    def conversion_func_3(value, dtype=None, name=None, as_ref=False):
        return tf.convert_to_tensor(value.value, dtype=tf.float32, name=name)

    input_dict = {
        "base_type": (YetAnotherClass,),
        "conversion_func": [conversion_func_3],
        "priority": 100
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Using as_ref
    class RefClass:
        def __init__(self, value):
            self.value = tf.Variable(value)

    def conversion_func_4(value, dtype=None, name=None, as_ref=True):
        return value.value

    input_dict = {
        "base_type": (RefClass,),
        "conversion_func": [conversion_func_4],
        "priority": 25
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Returning NotImplemented
    class NotImplClass:
        def __init__(self, value):
            self.value = value

    def conversion_func_5(value, dtype=None, name=None, as_ref=False):
        return NotImplemented

    input_dict = {
        "base_type": (NotImplClass,),
        "conversion_func": [conversion_func_5],
        "priority": 125
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Numpy array
    def conversion_func_6(value, dtype=None, name=None, as_ref=False):
        return tf.convert_to_tensor(value, dtype=dtype, name=name)

    input_dict = {
        "base_type": (np.ndarray,),
        "conversion_func": [conversion_func_6],
        "priority": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Integer
    def conversion_func_7(value, dtype=None, name=None, as_ref=False):
        return tf.convert_to_tensor(value, dtype=dtype, name=name)

    input_dict = {
        "base_type": (np.int32,),
        "conversion_func": [conversion_func_7],
        "priority": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.register_tensor_conversion_function"] = tf_register_tensor_conversion_function_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.register_tensor_conversion_function' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.register_tensor_conversion_function'.")

check_valid('tf.register_tensor_conversion_function', generated_inputs['tf.register_tensor_conversion_function'], lib="tf", suffix=0)
