
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_feature_column_make_parse_example_spec_inputs():
    list_of_inputs = []

    # Input 1: Empty list
    feature_columns = []
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single numeric column
    feature_columns = [tf.feature_column.numeric_column("feature_a")]
    input_dict = {"feature_columns": feature_columns}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.feature_column.make_parse_example_spec"] = tf_feature_column_make_parse_example_spec_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.feature_column.make_parse_example_spec' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.make_parse_example_spec'.")

check_valid('tf.feature_column.make_parse_example_spec', generated_inputs['tf.feature_column.make_parse_example_spec'], lib="tf", suffix=0)
