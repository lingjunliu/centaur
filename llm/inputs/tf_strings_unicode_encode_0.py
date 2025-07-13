
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_unicode_encode_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    input_tensor = np.array([[71, 246, 246, 100, 110, 105, 103, 104, 116]], dtype=np.int32)
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = "example1"
    input_dict = {"input": input_tensor, "output_encoding": output_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: UTF-16-BE encoding
    input_tensor = np.array([[71, 0, 246, 0, 246, 0, 100, 0, 110, 0, 105, 0, 103, 0, 104, 0, 116, 0]], dtype=np.int32)
    output_encoding = "UTF-16-BE"
    errors = "replace"
    replacement_char = 65533
    name = "example2"
    input_dict = {"input": input_tensor, "output_encoding": output_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: UTF-32-BE encoding
    input_tensor = np.array([[71, 0, 0, 0, 246, 0, 0, 0, 246, 0, 0, 0, 100, 0, 0, 0, 110, 0, 0, 0, 105, 0, 0, 0, 103, 0, 0, 0, 104, 0, 0, 0, 116, 0, 0, 0]], dtype=np.int32)
    output_encoding = "UTF-32-BE"
    errors = "replace"
    replacement_char = 65533
    name = "example3"
    input_dict = {"input": input_tensor, "output_encoding": output_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different replacement character
    input_tensor = np.array([[65, 66, 67]], dtype=np.int32)
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 42
    name = "example4"
    input_dict = {"input": input_tensor, "output_encoding": output_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: errors = "ignore"
    input_tensor = np.array([[71, 246, 246, 100, 110, 105, 103, 104, 116]], dtype=np.int32)
    output_encoding = "UTF-8"
    errors = "ignore"
    replacement_char = 65533
    name = "example5"
    input_dict = {"input": input_tensor, "output_encoding": output_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty input tensor
    input_tensor = np.array([[]], dtype=np.int32)
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = "example6"
    input_dict = {"input": input_tensor, "output_encoding": output_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multidimensional input tensor
    input_tensor = np.array([[[71, 246], [246, 100]], [[110, 105], [103, 104]]], dtype=np.int32)
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = "example7"
    input_dict = {"input": input_tensor, "output_encoding": output_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: different shape
    input_tensor = np.array([[71, 246], [246, 100], [110, 105]], dtype=np.int32)
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = "example8"
    input_dict = {"input": input_tensor, "output_encoding": output_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger replacement char
    input_tensor = np.array([[65, 66, 67]], dtype=np.int32)
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 128522
    name = "example9"
    input_dict = {"input": input_tensor, "output_encoding": output_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single element array
    input_tensor = np.array([[65]], dtype=np.int32)
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = "example10"
    input_dict = {"input": input_tensor, "output_encoding": output_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Name is None
    input_tensor = np.array([[71, 246, 246, 100, 110, 105, 103, 104, 116]], dtype=np.int32)
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = None
    input_dict = {"input": input_tensor, "output_encoding": output_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Valid unicode number
    input_tensor = np.array([[169]], dtype=np.int32)
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = "example12"
    input_dict = {"input": input_tensor, "output_encoding": output_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: More rows
    input_tensor = np.array([[71, 246, 246], [100, 110, 105], [103, 104, 116]], dtype=np.int32)
    output_encoding = "UTF-8"
    errors = "replace"
    replacement_char = 65533
    name = "example13"
    input_dict = {"input": input_tensor, "output_encoding": output_encoding, "errors": errors, "replacement_char": replacement_char, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.unicode_encode"] = tf_strings_unicode_encode_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.unicode_encode' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.unicode_encode'.")

check_valid('tf.strings.unicode_encode', generated_inputs['tf.strings.unicode_encode'], lib="tf", suffix=0)
