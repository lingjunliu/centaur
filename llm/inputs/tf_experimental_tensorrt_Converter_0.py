
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_experimental_tensorrt_converter_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "input_saved_model_dir": "path/to/model",
        "input_saved_model_tags": ["serve"],
        "input_saved_model_signature_key": "serving_default",
        "use_dynamic_shape": True,
        "dynamic_shape_profile_strategy": "Range",
        "max_workspace_size_bytes": 2147483648,
        "precision_mode": "FP16",
        "minimum_segment_size": 5,
        "maximum_cached_engines": 4,
        "use_calibration": False,
        "allow_build_at_runtime": True,
        "conversion_params": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "input_saved_model_dir": "another/path",
        "input_saved_model_tags": ["gpu"],
        "input_saved_model_signature_key": "predict",
        "use_dynamic_shape": False,
        "dynamic_shape_profile_strategy": "Optimal",
        "max_workspace_size_bytes": 536870912,
        "precision_mode": "FP32",
        "minimum_segment_size": 10,
        "maximum_cached_engines": 1,
        "use_calibration": True,
        "allow_build_at_runtime": False,
        "conversion_params": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "input_saved_model_dir": "/tmp/model",
        "input_saved_model_tags": [],
        "input_saved_model_signature_key": "",
        "use_dynamic_shape": True,
        "dynamic_shape_profile_strategy": "Range+Optimal",
        "max_workspace_size_bytes": 1073741824,
        "precision_mode": "INT8",
        "minimum_segment_size": 3,
        "maximum_cached_engines": 2,
        "use_calibration": False,
        "allow_build_at_runtime": True,
        "conversion_params": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "input_saved_model_dir": "model_dir",
        "input_saved_model_tags": ["latest"],
        "input_saved_model_signature_key": "infer",
        "use_dynamic_shape": False,
        "dynamic_shape_profile_strategy": "Range",
        "max_workspace_size_bytes": 805306368,
        "precision_mode": "FP16",
        "minimum_segment_size": 4,
        "maximum_cached_engines": 8,
        "use_calibration": True,
        "allow_build_at_runtime": False,
        "conversion_params": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "input_saved_model_dir": "saved_model",
        "input_saved_model_tags": ["tag1", "tag2"],
        "input_saved_model_signature_key": "serving",
        "use_dynamic_shape": True,
        "dynamic_shape_profile_strategy": "Optimal",
        "max_workspace_size_bytes": 4294967296,
        "precision_mode": "FP32",
        "minimum_segment_size": 7,
        "maximum_cached_engines": 16,
        "use_calibration": True,
        "allow_build_at_runtime": True,
        "conversion_params": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    input_dict = {
        "input_saved_model_dir": "path/to/a/different/model",
        "input_saved_model_tags": ["new_tag"],
        "input_saved_model_signature_key": "another_signature",
        "use_dynamic_shape": False,
        "dynamic_shape_profile_strategy": "Range+Optimal",
        "max_workspace_size_bytes": 67108864,
        "precision_mode": "INT8",
        "minimum_segment_size": 1,
        "maximum_cached_engines": 32,
        "use_calibration": True,
        "allow_build_at_runtime": False,
        "conversion_params": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "input_saved_model_dir": "a_model_location",
        "input_saved_model_tags": ["serving_tag"],
        "input_saved_model_signature_key": "a_key",
        "use_dynamic_shape": True,
        "dynamic_shape_profile_strategy": "Range",
        "max_workspace_size_bytes": 1610612736,
        "precision_mode": "FP16",
        "minimum_segment_size": 2,
        "maximum_cached_engines": 64,
        "use_calibration": True,
        "allow_build_at_runtime": True,
        "conversion_params": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "input_saved_model_dir": "some_other_dir",
        "input_saved_model_tags": ["production"],
        "input_saved_model_signature_key": "output",
        "use_dynamic_shape": False,
        "dynamic_shape_profile_strategy": "Optimal",
        "max_workspace_size_bytes": 3221225472,
        "precision_mode": "FP32",
        "minimum_segment_size": 6,
        "maximum_cached_engines": 128,
        "use_calibration": True,
        "allow_build_at_runtime": False,
        "conversion_params": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "input_saved_model_dir": "another/model/path",
        "input_saved_model_tags": ["testing"],
        "input_saved_model_signature_key": "results",
        "use_dynamic_shape": True,
        "dynamic_shape_profile_strategy": "Range+Optimal",
        "max_workspace_size_bytes": 134217728,
        "precision_mode": "INT8",
        "minimum_segment_size": 9,
        "maximum_cached_engines": 256,
        "use_calibration": True,
        "allow_build_at_runtime": False,
        "conversion_params": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "input_saved_model_dir": "final/model/location",
        "input_saved_model_tags": ["deployment"],
        "input_saved_model_signature_key": "predictions",
        "use_dynamic_shape": False,
        "dynamic_shape_profile_strategy": "Range",
        "max_workspace_size_bytes": 268435456,
        "precision_mode": "FP16",
        "minimum_segment_size": 8,
        "maximum_cached_engines": 512,
        "use_calibration": True,
        "allow_build_at_runtime": True,
        "conversion_params": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.tensorrt.Converter"] = tf_experimental_tensorrt_converter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.tensorrt.Converter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.tensorrt.Converter'.")

check_valid('tf.experimental.tensorrt.Converter', generated_inputs['tf.experimental.tensorrt.Converter'], lib="tf", suffix=0)
