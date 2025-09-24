
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_raw_ops_stringsplitv2_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.StringSplitV2 operation.
    """
    list_of_inputs = []

    # Helper function to create input dictionaries
    def create_input_dict(input_tensor, sep_tensor, maxsplit=-1, name=""):
        return {
            'input': np.array(input_tensor, dtype=np.object_),
            'sep': np.array(sep_tensor, dtype=np.object_),
            'maxsplit': int(maxsplit),
            'name': str(name)
        }

    # Input 1: Basic case with space delimiter
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(
            input_tensor=['hello world', 'a b c'],
            sep_tensor=' '
        )
    ))

    # Input 2: Multi-character separator
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(
            input_tensor=['apple<>banana<>cherry', '1<>2<><>3'],
            sep_tensor='<>'
        )
    ))

    # Input 3: Empty string separator (splits on consecutive whitespace)
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(
            input_tensor=['  leading and  trailing  ', 'one \t two\nthree'],
            sep_tensor='',
            name='empty_sep'
        )
    ))

    # Input 4: maxsplit = 1
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(
            input_tensor=['a,b,c', 'd,e,f,g'],
            sep_tensor=',',
            maxsplit=1
        )
    ))

    # Input 5: maxsplit = 0 (no splits)
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(
            input_tensor=['a,b,c', 'd,e,f'],
            sep_tensor=',',
            maxsplit=0
        )
    ))

    # Input 6: maxsplit > 1
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(
            input_tensor=['a-b-c-d', 'e-f-g'],
            sep_tensor='-',
            maxsplit=2,
            name='maxsplit_2'
        )
    ))

    # Input 7: Consecutive delimiters, producing empty strings
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(
            input_tensor=['1..2...3', '4..5'],
            sep_tensor='.'
        )
    ))

    # Input 8: Input tensor with an empty string
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(
            input_tensor=['first item', '', 'third item'],
            sep_tensor=' '
        )
    ))

    # Input 9: Leading and trailing delimiters
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(
            input_tensor=['|a|b|', '|c|d'],
            sep_tensor='|',
            name='leading_trailing_delimiters'
        )
    ))

    # Input 10: Separator not found in some strings
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(
            input_tensor=['no_delimiter_here', 'a;b;c'],
            sep_tensor=';'
        )
    ))

    # Input 11: Single element in the input tensor
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(
            input_tensor=['this is a single sentence to split'],
            sep_tensor=' '
        )
    ))

    # Input 12: Unicode characters in input and separator
    list_of_inputs.append(copy.deepcopy(
        create_input_dict(
            input_tensor=['你好 世界', 'こんにちは 世界'],
            sep_tensor=' ',
            name='unicode_split'
        )
    ))

    return list_of_inputs

generated_inputs["tf.raw_ops.StringSplitV2"] = get_tf_raw_ops_stringsplitv2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.StringSplitV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringSplitV2'.")

check_valid('tf.raw_ops.StringSplitV2', generated_inputs['tf.raw_ops.StringSplitV2'], lib="tf", suffix=0)
