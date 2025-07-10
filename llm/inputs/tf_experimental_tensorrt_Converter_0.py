
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
        "input_saved_model_dir": "path/to/model1",
        "input_saved_model_tags": ["serve"],
        "input_saved_model_signature_key": "serving_default",
        "use_dynamic_shape": False,
        "dynamic_shape_profile_strategy": "Range",
        "max_workspace_size_bytes": 2048000000,
        "precision_mode": "FP32",
        "minimum_segment_size": 5,
        "maximum_cached_engines": 2,
        "use_calibration": True,
        "allow_build_at_runtime": True,
        "conversion_params": "a"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "input_saved_model_dir": "path/to/model2",
        "input_saved_model_tags": ["train", "serve"],
        "input_saved_model_signature_key": "another_signature",
        "use_dynamic_shape": True,
        "dynamic_shape_profile_strategy": "Optimal",
        "max_workspace_size_bytes": 512000000,
        "precision_mode": "FP16",
        "minimum_segment_size": 1,
        "maximum_cached_engines": 1,
        "use_calibration": False,
        "allow_build_at_runtime": False,
        "conversion_params": "b"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "input_saved_model_dir": "path/to/model3",
        "input_saved_model_tags": [],
        "input_saved_model_signature_key": "default",
        "use_dynamic_shape": False,
        "dynamic_shape_profile_strategy": "Range+Optimal",
        "max_workspace_size_bytes": 1073741824,
        "precision_mode": "INT8",
        "minimum_segment_size": 10,
        "maximum_cached_engines": 4,
        "use_calibration": True,
        "allow_build_at_runtime": True,
        "conversion_params": "c"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "input_saved_model_dir": "path/to/model4",
        "input_saved_model_tags": ["gpu"],
        "input_saved_model_signature_key": "predict",
        "use_dynamic_shape": True,
        "dynamic_shape_profile_strategy": "Range",
        "max_workspace_size_bytes": 4294967296,
        "precision_mode": "FP32",
        "minimum_segment_size": 3,
        "maximum_cached_engines": 8,
        "use_calibration": False,
        "allow_build_at_runtime": False,
        "conversion_params": "d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "input_saved_model_dir": "path/to/model5",
        "input_saved_model_tags": ["tag1", "tag2"],
        "input_saved_model_signature_key": "infer",
        "use_dynamic_shape": False,
        "dynamic_shape_profile_strategy": "Optimal",
        "max_workspace_size_bytes": 268435456,
        "precision_mode": "FP16",
        "minimum_segment_size": 2,
        "maximum_cached_engines": 16,
        "use_calibration": True,
        "allow_build_at_runtime": True,
        "conversion_params": "e"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    input_dict = {
        "input_saved_model_dir": "path/to/model6",
        "input_saved_model_tags": ["test"],
        "input_saved_model_signature_key": "classifier",
        "use_dynamic_shape": True,
        "dynamic_shape_profile_strategy": "Range+Optimal",
        "max_workspace_size_bytes": 8589934592,
        "precision_mode": "INT8",
        "minimum_segment_size": 4,
        "maximum_cached_engines": 32,
        "use_calibration": False,
        "allow_build_at_runtime": False,
        "conversion_params": "f"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "input_saved_model_dir": "path/to/model7",
        "input_saved_model_tags": [],
        "input_saved_model_signature_key": "regressor",
        "use_dynamic_shape": False,
        "dynamic_shape_profile_strategy": "Range",
        "max_workspace_size_bytes": 67108864,
        "precision_mode": "FP32",
        "minimum_segment_size": 6,
        "maximum_cached_engines": 1,
        "use_calibration": True,
        "allow_build_at_runtime": True,
        "conversion_params": "g"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "input_saved_model_dir": "path/to/model8",
        "input_saved_model_tags": ["validation"],
        "input_saved_model_signature_key": "validation_func",
        "use_dynamic_shape": True,
        "dynamic_shape_profile_strategy": "Optimal",
        "max_workspace_size_bytes": 17179869184,
        "precision_mode": "FP16",
        "minimum_segment_size": 7,
        "maximum_cached_engines": 2,
        "use_calibration": False,
        "allow_build_at_runtime": False,
        "conversion_params": "h"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "input_saved_model_dir": "path/to/model9",
        "input_saved_model_tags": [],
        "input_saved_model_signature_key": "",
        "use_dynamic_shape": False,
        "dynamic_shape_profile_strategy": "Range+Optimal",
        "max_workspace_size_bytes": 34359738368,
        "precision_mode": "INT8",
        "minimum_segment_size": 8,
        "maximum_cached_engines": 4,
        "use_calibration": True,
        "allow_build_at_runtime": True,
        "conversion_params": "i"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "input_saved_model_dir": "path/to/model10",
        "input_saved_model_tags": ["example"],
        "input_saved_model_signature_key": "example_func",
        "use_dynamic_shape": True,
        "dynamic_shape_profile_strategy": "Range",
        "max_workspace_size_bytes": 134217728,
        "precision_mode": "FP32",
        "minimum_segment_size": 9,
        "maximum_cached_engines": 8,
        "use_calibration": False,
        "allow_build_at_runtime": False,
        "conversion_params": "j"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_dict = {
        "input_saved_model_dir": "",
        "input_saved_model_tags": [],
        "input_saved_model_signature_key": "",
        "use_dynamic_shape": False,
        "dynamic_shape_profile_strategy": "Range",
        "max_workspace_size_bytes": 1024,
        "precision_mode": "FP32",
        "minimum_segment_size": 1,
        "maximum_cached_engines": 1,
        "use_calibration": True,
        "allow_build_at_runtime": True,
        "conversion_params": "k"
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
