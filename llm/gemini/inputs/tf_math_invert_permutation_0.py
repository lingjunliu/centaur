
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_invert_permutation_inputs():
    list_of_inputs = []
    
    # 1. 1-element permutation (int32)
    list_of_inputs.append({
        'x': np.array([0], dtype=np.int32),
        'name': 'perm_1'
    })
    
    # 2. 2-element permutation (int32)
    list_of_inputs.append({
        'x': np.array([1, 0], dtype=np.int32),
        'name': 'perm_2'
    })
    
    # 3. 3-element permutation (int32)
    list_of_inputs.append({
        'x': np.array([2, 0, 1], dtype=np.int32),
        'name': 'perm_3'
    })
    
    # 4. 5-element permutation (int32)
    list_of_inputs.append({
        'x': np.array([3, 4, 0, 2, 1], dtype=np.int32),
        'name': 'perm_4'
    })
    
    # 5. Identity permutation (int32)
    list_of_inputs.append({
        'x': np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
        'name': 'perm_5'
    })
    
    # 6. Reversed permutation (int64)
    list_of_inputs.append({
        'x': np.array([5, 4, 3, 2, 1, 0], dtype=np.int64),
        'name': 'perm_6'
    })
    
    # 7. Shift permutation (int64)
    list_of_inputs.append({
        'x': np.array([1, 2, 3, 0], dtype=np.int64),
        'name': 'perm_7'
    })
    
    # 8. Reversed 3-element permutation (int64)
    list_of_inputs.append({
        'x': np.array([2, 1, 0], dtype=np.int64),
        'name': 'perm_8'
    })
    
    # 9. Large elements (int32)
    list_of_inputs.append({
        'x': np.array([4, 0, 1, 2, 3], dtype=np.int32),
        'name': 'perm_9'
    })
    
    # 10. Multi-swap permutation (int32)
    list_of_inputs.append({
        'x': np.array([1, 0, 3, 2], dtype=np.int32),
        'name': 'perm_10'
    })
    
    return list_of_inputs

generated_inputs["tf.math.invert_permutation"] = tf_math_invert_permutation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.invert_permutation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.invert_permutation'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.invert_permutation', generated_inputs['tf.math.invert_permutation'], lib="tf", suffix=0)
