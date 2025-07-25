
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_io_decode_csv_inputs():
    list_of_inputs = []

    # Input 1: Basic case with required columns (int, float, string)
    input_dict_1 = {
        'records': np.array(['1,2.0,hello', '4,5.0,world'], dtype=object),
        'record_defaults': [np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=object)],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'basic_required'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Using default values for missing fields
    input_dict_2 = {
        'records': np.array(['1,2.0,', '4,,world'], dtype=object),
        'record_defaults': [np.array(0, dtype=np.int32), np.array(-1.0, dtype=np.float32), np.array('default', dtype=object)],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'with_defaults'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Using a different field delimiter (semicolon)
    input_dict_3 = {
        'records': np.array(['-10;3.14;tensor', '100;-9.99;flow'], dtype=object),
        'record_defaults': [np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=object)],
        'field_delim': ';',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'semicolon_delim'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: `use_quote_delim` set to False
    input_dict_4 = {
        'records': np.array(['"a",b', 'c,"d"'], dtype=object),
        'record_defaults': [np.array([], dtype=object), np.array([], dtype=object)],
        'field_delim': ',',
        'use_quote_delim': False,
        'na_value': '',
        'select_cols': None,
        'name': 'no_quote_delim'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Using a custom `na_value`
    input_dict_5 = {
        'records': np.array(['1,NA,hello', '99,5.0,NA'], dtype=object),
        'record_defaults': [np.array(0, dtype=np.int32), np.array(0.0, dtype=np.float32), np.array('was_na', dtype=object)],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': 'NA',
        'select_cols': None,
        'name': 'custom_na'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: RFC 4180 compliant quoted fields
    input_dict_6 = {
        'records': np.array(['1,"a,b,c",3.0', '4,"d,e",6.0'], dtype=object),
        'record_defaults': [np.array([], dtype=np.int32), np.array([], dtype=object), np.array([], dtype=np.float32)],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'rfc4180_quotes'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty `records` tensor
    input_dict_7 = {
        'records': np.array([], dtype=object),
        'record_defaults': [np.array(0, dtype=np.int32), np.array(0.0, dtype=np.float32), np.array('a', dtype=object)],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'empty_records'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Numeric fields with leading/trailing spaces
    input_dict_8 = {
        'records': np.array(['  1  ,  2.5  ', ' -3 , -4.5 '], dtype=object),
        'record_defaults': [np.array([], dtype=np.int32), np.array([], dtype=np.float32)],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'numeric_spaces'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Using int64 and float64 dtypes for large numbers
    input_dict_9 = {
        'records': np.array(['9223372036854775807,1.79e+308', '-9223372036854775808,-1.79e+308'], dtype=object),
        'record_defaults': [np.array(0, dtype=np.int64), np.array(0.0, dtype=np.float64)],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'large_numbers'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Single record in the input tensor
    input_dict_10 = {
        'records': np.array(['-1,-2.2,a string'], dtype=object),
        'record_defaults': [np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=object)],
        'field_delim': ',',
        'use_quote_delim': True,
        'na_value': '',
        'select_cols': None,
        'name': 'single_record'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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
