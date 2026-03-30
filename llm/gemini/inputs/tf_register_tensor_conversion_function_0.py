
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_register_tensor_conversion_function_inputs():
    list_of_inputs = []

    def conversion_func_1(value, dtype=None, name=None, as_ref=False):
        return tf.convert_to_tensor(np.array(value.value), dtype=dtype, name=name)

    class CustomType1:
        def __init__(self, value):
            self.value = value

    input_dict_1 = {
        "base_type": (CustomType1,),
        "conversion_func": [conversion_func_1],
        "priority": 100
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    def conversion_func_2(value, dtype=None, name=None, as_ref=False):
        return tf.convert_to_tensor(np.array(value.value), dtype=dtype, name=name)

    class CustomType2:
        def __init__(self, value):
            self.value = value

    input_dict_2 = {
        "base_type": (CustomType2,),
        "conversion_func": [conversion_func_2],
        "priority": 50
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    def conversion_func_3(value, dtype=None, name=None, as_ref=False):
        return tf.convert_to_tensor(np.array(value.value), dtype=dtype, name=name)

    class CustomType3:
        def __init__(self, value):
            self.value = value

    input_dict_3 = {
        "base_type": (CustomType3,),
        "conversion_func": [conversion_func_3],
        "priority": 75
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    def conversion_func_4(value, dtype=None, name=None, as_ref=False):
        return tf.convert_to_tensor(np.array(value.value), dtype=dtype, name=name)

    class CustomType4:
        def __init__(self, value):
            self.value = value

    input_dict_4 = {
        "base_type": (CustomType4,),
        "conversion_func": [conversion_func_4],
        "priority": 25
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    def conversion_func_5(value, dtype=None, name=None, as_ref=False):
        return tf.convert_to_tensor(np.array(value.value), dtype=dtype, name=name)

    class CustomType5:
        def __init__(self, value):
            self.value = value

    input_dict_5 = {
        "base_type": (CustomType5,),
        "conversion_func": [conversion_func_5],
        "priority": 125
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    def conversion_func_6(value, dtype=None, name=None, as_ref=False):
        return tf.convert_to_tensor(np.array(value.value), dtype=dtype, name=name)

    class CustomType6:
        def __init__(self, value):
            self.value = value

    input_dict_6 = {
        "base_type": (CustomType6,),
        "conversion_func": [conversion_func_6],
        "priority": 175
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    def conversion_func_7(value, dtype=None, name=None, as_ref=False):
        return tf.convert_to_tensor(np.array(value.value), dtype=dtype, name=name)

    class CustomType7:
        def __init__(self, value):
            self.value = value

    input_dict_7 = {
        "base_type": (CustomType7,),
        "conversion_func": [conversion_func_7],
        "priority": 250
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    def conversion_func_8(value, dtype=None, name=None, as_ref=False):
        return tf.convert_to_tensor(np.array(value.value), dtype=dtype, name=name)

    class CustomType8:
        def __init__(self, value):
            self.value = value

    input_dict_8 = {
        "base_type": (CustomType8,),
        "conversion_func": [conversion_func_8],
        "priority": 300
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    def conversion_func_9(value, dtype=None, name=None, as_ref=False):
        return tf.convert_to_tensor(np.array(value.value), dtype=dtype, name=name)

    class CustomType9:
        def __init__(self, value):
            self.value = value

    input_dict_9 = {
        "base_type": (CustomType9,),
        "conversion_func": [conversion_func_9],
        "priority": 350
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    def conversion_func_10(value, dtype=None, name=None, as_ref=False):
        return tf.convert_to_tensor(np.array(value.value), dtype=dtype, name=name)

    class CustomType10:
        def __init__(self, value):
            self.value = value

    input_dict_10 = {
        "base_type": (CustomType10,),
        "conversion_func": [conversion_func_10],
        "priority": 400
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs = {}
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


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.register_tensor_conversion_function', generated_inputs['tf.register_tensor_conversion_function'], lib="tf", suffix=0)
