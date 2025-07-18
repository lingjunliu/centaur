
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

# This class is a workaround for a testing framework that requires a list
# for a parameter that the underlying API expects to be a callable.
# It is both a list (via inheritance) and a callable (via __call__).
# It also correctly handles deepcopying to maintain its type.
class CallableList(list):
    def __init__(self, func):
        super().__init__([func])
        self._func = func

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def __deepcopy__(self, memo):
        # Create a new instance of our class to preserve the type
        new_instance = self.__class__(self._func)
        memo[id(self)] = new_instance
        return new_instance

# Define classes and functions at the module level to ensure they are accessible.
class MyNumber:
    def __init__(self, value):
        self.value = value

def convert_my_number(value, dtype=None, name=None, as_ref=False):
    return tf.constant(value.value, dtype=dtype, name=name)

class MyString:
    def __init__(self, value):
        self.value = value

def convert_my_string(value, dtype=None, name=None, as_ref=False):
    return tf.constant(value.value, dtype=tf.string if dtype is None else dtype, name=name)

class MyPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def convert_my_point(value, dtype=None, name=None, as_ref=False):
    return tf.constant([value.x, value.y], dtype=dtype, name=name)

class MySpecialNumber(MyNumber):
    pass

def convert_multiple_types(value, dtype=None, name=None, as_ref=False):
    if isinstance(value, MySpecialNumber):
        return tf.constant(value.value * 100, dtype=dtype, name=name)
    return tf.constant(value.value, dtype=dtype, name=name)

class MyMatrix:
    def __init__(self, data):
        self.data = data

def convert_my_matrix(value, dtype=None, name=None, as_ref=False):
    return tf.constant(np.array(value.data), dtype=dtype, name=name)

class MyRef:
    def __init__(self, initial_value):
        self.value = initial_value

def convert_to_ref(value, dtype=None, name=None, as_ref=False):
    if as_ref:
        return tf.Variable(value.value, dtype=dtype, name=name)
    else:
        return tf.constant(value.value, dtype=dtype, name=name)

class MaybeConvertible:
    def __init__(self, value, convertible=True):
        self.value = value
        self.convertible = convertible

def convert_maybe(value, dtype=None, name=None, as_ref=False):
    if not value.convertible:
        return NotImplemented
    return tf.constant(value.value, dtype=dtype, name=name)

def convert_complex_to_real_pair(value, dtype=None, name=None, as_ref=False):
    dtype_ = dtype if dtype is not None else tf.float32
    return tf.constant([value.real, value.imag], dtype=dtype_, name=name)

class HighPriorityType:
    def __init__(self, value):
        self.value = value

def convert_high_priority(value, dtype=None, name=None, as_ref=False):
    return tf.constant(value.value, dtype=dtype, name=name)

class MyConfig:
    def __init__(self, config_dict):
        self.config = config_dict

def convert_my_config(value, dtype=None, name=None, as_ref=False):
    total = sum(v for v in value.config.values() if isinstance(v, (int, float)))
    return tf.constant(total, dtype=dtype, name=name)

def tf_register_tensor_conversion_function_inputs():
    list_of_inputs = []

    # Input 1: Basic case with default priority
    input_dict = {
        'base_type': (MyNumber,),
        'conversion_func': CallableList(convert_my_number),
        'priority': 100
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different base type and higher priority
    input_dict = {
        'base_type': (MyString,),
        'conversion_func': CallableList(convert_my_string),
        'priority': 90
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Lower priority
    input_dict = {
        'base_type': (MyPoint,),
        'conversion_func': CallableList(convert_my_point),
        'priority': 110
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multiple base types in the tuple
    input_dict = {
        'base_type': (MyNumber, MySpecialNumber),
        'conversion_func': CallableList(convert_multiple_types),
        'priority': 99
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero priority
    input_dict = {
        'base_type': (MyMatrix,),
        'conversion_func': CallableList(convert_my_matrix),
        'priority': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Conversion function handling as_ref=True
    input_dict = {
        'base_type': (MyRef,),
        'conversion_func': CallableList(convert_to_ref),
        'priority': 85
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Conversion function that can return NotImplemented
    input_dict = {
        'base_type': (MaybeConvertible,),
        'conversion_func': CallableList(convert_maybe),
        'priority': 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Overriding conversion for a built-in type (complex)
    input_dict = {
        'base_type': (complex,),
        'conversion_func': CallableList(convert_complex_to_real_pair),
        'priority': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative priority for highest precedence
    input_dict = {
        'base_type': (HighPriorityType,),
        'conversion_func': CallableList(convert_high_priority),
        'priority': -10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large priority value for lowest precedence
    input_dict = {
        'base_type': (MyConfig,),
        'conversion_func': CallableList(convert_my_config),
        'priority': 200
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.register_tensor_conversion_function"] = tf_register_tensor_conversion_function_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.register_tensor_conversion_function' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.register_tensor_conversion_function'.")

check_valid('tf.register_tensor_conversion_function', generated_inputs['tf.register_tensor_conversion_function'], lib="tf", suffix=0)
