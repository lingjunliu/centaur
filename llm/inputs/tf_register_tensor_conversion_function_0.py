
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_register_tensor_conversion_function_inputs():
    list_of_inputs = []

    # Input 1: Simple conversion
    def conversion_func1(value, dtype=None, name=None, as_ref=False):
        return tf.constant(value.value, dtype=dtype, name=name)

    class MyType1:
        def __init__(self, value):
            self.value = value

    input_dict = {
        "base_type": (MyType1,),
        "conversion_func": conversion_func1,
        "priority": 50
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Conversion with dtype specification
    def conversion_func2(value, dtype=None, name=None, as_ref=False):
        return tf.constant(value.value, dtype=dtype, name=name)

    class MyType2:
        def __init__(self, value):
            self.value = value

    input_dict = {
        "base_type": (MyType2,),
        "conversion_func": conversion_func2,
        "priority": 75
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Conversion returning NotImplemented
    def conversion_func3(value, dtype=None, name=None, as_ref=False):
        if isinstance(value.value, str):
            return tf.constant(value.value, dtype=dtype, name=name)
        return NotImplemented

    class MyType3:
        def __init__(self, value):
            self.value = value

    input_dict = {
        "base_type": (MyType3,),
        "conversion_func": conversion_func3,
        "priority": 25
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Conversion with name specification
    def conversion_func4(value, dtype=None, name=None, as_ref=False):
        return tf.constant(value.value, dtype=dtype, name=name)

    class MyType4:
        def __init__(self, value):
            self.value = value

    input_dict = {
        "base_type": (MyType4,),
        "conversion_func": conversion_func4,
        "priority": 100
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Conversion with as_ref=True (returning a Variable)
    def conversion_func5(value, dtype=None, name=None, as_ref=False):
        if as_ref:
            return tf.Variable(value.value, dtype=dtype, name=name)
        else:
            return tf.constant(value.value, dtype=dtype, name=name)

    class MyType5:
        def __init__(self, value):
            self.value = value

    input_dict = {
        "base_type": (MyType5,),
        "conversion_func": conversion_func5,
        "priority": 125
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiple base types

    def conversion_func6(value, dtype=None, name=None, as_ref=False):
        return tf.constant(value.value, dtype=dtype, name=name)

    class MyType6_1:
        def __init__(self, value):
            self.value = value

    class MyType6_2:
        def __init__(self, value):
            self.value = value

    def conversion_func6_wrapper(value, dtype=None, name=None, as_ref=False):
        return conversion_func6(value, dtype, name, as_ref)

    input_dict = {
        "base_type": (MyType6_1, MyType6_2),
        "conversion_func": conversion_func6_wrapper,
        "priority": 80
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7: Different priority
    def conversion_func7(value, dtype=None, name=None, as_ref=False):
        return tf.constant(value.value, dtype=dtype, name=name)

    class MyType7:
        def __init__(self, value):
            self.value = value

    input_dict = {
        "base_type": (MyType7,),
        "conversion_func": conversion_func7,
        "priority": 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Using different datatypes in conversion function
    def conversion_func8(value, dtype=None, name=None, as_ref=False):
        if dtype is None:
            dtype = tf.float32
        return tf.constant(value.value, dtype=dtype, name=name)
    class MyType8:
        def __init__(self, value):
            self.value = value

    input_dict = {
        "base_type": (MyType8,),
        "conversion_func": conversion_func8,
        "priority": 90
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
