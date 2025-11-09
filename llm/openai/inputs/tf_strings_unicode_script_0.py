
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_unicode_script_inputs():
    list_of_inputs = []

    arr = np.array([1, 31, 38], dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[65, 97, 48], [90, 122, 57]], dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array(
        [
            [[233, 1046, 937], [1569, 1488, 2309]],
            [[4352, 12354, 20013], [12452, 65313, 0x1F600]],
        ],
        dtype=np.int32,
    )
    input_dict = {"input": arr, "name": "unicode_script_case_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([-1, -100, 1114112, 2000000], dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_4_invalids"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([0, 127, 128512, 1114111], dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_5_boundaries"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array(128169, dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_6_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([], dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_7_empty_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.empty((0, 3), dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_8_empty_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([55296, 56320, 57343, 57344], dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_9_surrogates_pua"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[[[9, 945, 1040, 44032]], [[19968, 20108, 20225, 20320]]]], dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_10_4d_mixed"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[769, 8205, 1632], [8212, 8226, 8364]], dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_11_combining_symbols"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array(
        [
            [[-5, 65], [0, 55297], [1114111, 128175]],
            [[173, 180], [156, 711], [701, 7424]],
        ],
        dtype=np.int32,
    )
    input_dict = {"input": arr, "name": "unicode_script_case_12_3d_varied"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.strings.unicode_script"] = tf_strings_unicode_script_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strings.unicode_script' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.unicode_script'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.strings.unicode_script', generated_inputs['tf.strings.unicode_script'], lib="tf", suffix=0)
