
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_io_decode_csv_inputs():
    list_of_inputs = []

    # Input 1: Basic usage with mixed numeric types
    input_dict = {
        'records': np.array(['1,2.0,100', '4,5.1,200'], dtype=object),
        'record_defaults': np.array([
            np.array(0, dtype=np.int32),
            np.array(0.0, dtype=np.float32),
            np.array(0, dtype=np.int64)
        ], dtype=object),
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different delimiter and all int64
    input_dict = {
        'records': np.array(['-10;20;30', '40;50;60'], dtype=object),
        'record_defaults': np.array([
            np.array(0, dtype=np.int64),
            np.array(0, dtype=np.int64),
            np.array(0, dtype=np.int64)
        ], dtype=object),
        'field_delim': ';',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'test_delim'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Handling missing values with mixed numeric types
    input_dict = {
        'records': np.array(['1,,10', ',5.1,', '7,8.2,30'], dtype=object),
        'record_defaults': np.array([
            np.array(-1, dtype=np.int32),
            np.array(-1.0, dtype=np.float64),
            np.array(-1, dtype=np.int64)
        ], dtype=object),
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Using `na_value`, all numeric
    input_dict = {
        'records': np.array(['1,NA,3', 'NA,5.1,4'], dtype=object),
        'record_defaults': np.array([
            np.array(0, dtype=np.int32),
            np.array(0.0, dtype=np.float32),
            np.array(0, dtype=np.int32)
        ], dtype=object),
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': 'NA',
        'select_cols': None,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Using `select_cols` with all numeric types
    input_dict = {
        'records': np.array(['1,2.0,3,4.0', '5,6.0,7,8.0'], dtype=object),
        'record_defaults': np.array([
            np.array(0, dtype=np.int32),
            np.array(0, dtype=np.int64)
        ], dtype=object),
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': np.array([0, 2], dtype=np.int32),
        'name': 'select_cols_test'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All string types
    input_dict = {
        'records': np.array(['"a,b",c,d', 'e,"f,g",h'], dtype=object),
        'record_defaults': np.array([
            np.array('', dtype=object),
            np.array('', dtype=object),
            np.array('', dtype=object)
        ], dtype=object),
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'all_string'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: `use_quote_delim` is False with all strings
    input_dict = {
        'records': np.array(['a,"b",c'], dtype=object),
        'record_defaults': np.array([
            np.array('', dtype=object),
            np.array('', dtype=object),
            np.array('', dtype=object)
        ], dtype=object),
        'field_delim': ',',
        'use_quote_delim': False,
        'na_value': '',
        'select_cols': None,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Required fields (empty default tensors), all numeric
    input_dict = {
        'records': np.array(['1.1,2,3.3', '4.4,5,6.6'], dtype=object),
        'record_defaults': np.array([
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64)
        ], dtype=object),
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Mix of required and default value fields, all numeric
    input_dict = {
        'records': np.array(['1,2.0,', '4,,6'], dtype=object),
        'record_defaults': np.array([
            np.array([], dtype=np.int32),
            np.array(-1.0, dtype=np.float64),
            np.array(99, dtype=np.int64)
        ], dtype=object),
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'mixed_defaults_numeric'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: All float types with special values and a different delimiter
    input_dict = {
        'records': np.array(['1.2e3| -4.5e-2', 'INF| -INF', 'NaN| 0.0'], dtype=object),
        'record_defaults': np.array([
            np.array(0.0, dtype=np.float32),
            np.array(0.0, dtype=np.float64)
        ], dtype=object),
        'field_delim': '|',
        'use_quote_delim': True,
        'na_value': 'NaN',
        'select_cols': None,
        'name': 'float_test'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.io.decode_csv"] = tf_io_decode_csv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.decode_csv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_csv'.")

check_valid('tf.io.decode_csv', generated_inputs['tf.io.decode_csv'], lib="tf", suffix=0)
