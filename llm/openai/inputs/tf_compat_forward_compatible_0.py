
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_compat_forward_compatible_inputs():
    list_of_inputs = []

    year = np.int32(2020); month = np.int32(1); day = np.int32(15)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int64(2024); month = np.int8(2); day = np.int8(29)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int16(2023); month = np.int8(12); day = np.int8(31)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int32(1999); month = np.int16(11); day = np.int16(30)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int64(2000); month = np.int16(2); day = np.int16(29)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int16(1904); month = np.int8(2); day = np.int8(29)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int16(1); month = np.int8(1); day = np.int8(1)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int32(9999); month = np.int8(12); day = np.int8(31)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int32(2022); month = np.int8(4); day = np.int8(30)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int16(2021); month = np.int8(2); day = np.int8(28)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int32(2018); month = np.int8(3); day = np.int8(1)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int32(2019); month = np.int8(6); day = np.int8(1)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    return list_of_inputs

generated_inputs["tf.compat.forward_compatible"] = tf_compat_forward_compatible_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.compat.forward_compatible' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.compat.forward_compatible'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.compat.forward_compatible', generated_inputs['tf.compat.forward_compatible'], lib="tf", suffix=0)
