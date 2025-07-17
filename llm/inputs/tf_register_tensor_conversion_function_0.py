
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_register_tensor_conversion_function_inputs():
    list_of_inputs = []

    def conversion_func_custom(value, dtype=None, name=None, as_ref=False):
        if isinstance(value, CustomClass):
            return tf.constant(value.value, dtype=dtype, name=name)
        else:
            return NotImplemented

    class CustomClass:
        def __init__(self, value):
            self.value = value

    def conversion_func_list(value, dtype=None, name=None, as_ref=False):
        if isinstance(value, list):
            return tf.constant(np.array(value), dtype=dtype, name=name)
        else:
            return NotImplemented

    def conversion_func_dict(value, dtype=None, name=None, as_ref=False):
        if isinstance(value, dict):
            return tf.constant(list(value.values()), dtype=dtype, name=name)
        else:
            return NotImplemented

    def conversion_func_variable(value, dtype=None, name=None, as_ref=False):
        if as_ref:
            return tf.Variable(value, dtype=dtype, name=name)
        else:
            return NotImplemented

    class AnotherCustomClass:
        def __init__(self, value):
            self.value = value

    def conversion_func_another_custom(value, dtype=None, name=None, as_ref=False):
        if isinstance(value, AnotherCustomClass):
            return tf.constant(value.value, dtype=dtype, name=name)
        else:
            return NotImplemented

    class YetAnotherClass:
        def __init__(self, value):
            self.value = value

    def conversion_func_yet_another(value, dtype=None, name=None, as_ref=False):
        if isinstance(value, YetAnotherClass):
            return tf.constant(value.value, dtype=dtype, name=name)
        else:
            return NotImplemented

    input_dict_1 = {"base_type": (CustomClass,), "conversion_func": [conversion_func_custom], "priority": 50}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_dict_2 = {"base_type": (list,), "conversion_func": [conversion_func_list], "priority": 75}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_dict_3 = {"base_type": (dict,), "conversion_func": [conversion_func_dict], "priority": 100}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_dict_4 = {"base_type": (AnotherCustomClass,), "conversion_func": [conversion_func_another_custom], "priority": 25}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_dict_5 = {"base_type": (YetAnotherClass,), "conversion_func": [conversion_func_yet_another], "priority": 125}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    class RefClass:
      def __init__(self, value):
        self.value = value

    def conversion_func_ref(value, dtype=None, name=None, as_ref=False):
        if isinstance(value, RefClass) and as_ref:
          return tf.Variable(value.value, dtype=dtype, name=name)
        else:
          return NotImplemented

    input_dict_6 = {"base_type": (RefClass,), "conversion_func": [conversion_func_ref], "priority": 60}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    def conversion_func_tuple(value, dtype=None, name=None, as_ref=False):
        if isinstance(value, tuple):
            return tf.constant(np.array(value), dtype=dtype, name=name)
        else:
            return NotImplemented

    input_dict_7 = {"base_type": (tuple,), "conversion_func": [conversion_func_tuple], "priority": 80}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    class StringClass:
        def __init__(self, value):
            self.value = value

    def conversion_func_string_class(value, dtype=None, name=None, as_ref=False):
        if isinstance(value, StringClass):
            return tf.constant(value.value, dtype=dtype, name=name)
        else:
            return NotImplemented

    input_dict_8 = {"base_type": (StringClass,), "conversion_func": [conversion_func_string_class], "priority": 90}
    list_of_inputs.append(copy.deepcopy(input_dict_8))


    class BytesClass:
        def __init__(self, value):
            self.value = value

    def conversion_func_bytes_class(value, dtype=None, name=None, as_ref=False):
        if isinstance(value, BytesClass):
            return tf.constant(value.value, dtype=dtype, name=name)
        else:
            return NotImplemented

    input_dict_9 = {"base_type": (BytesClass,), "conversion_func": [conversion_func_bytes_class], "priority": 105}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

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
