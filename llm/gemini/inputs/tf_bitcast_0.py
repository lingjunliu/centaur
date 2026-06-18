
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_bitcast_inputs():
    list_of_inputs = []
    
    # 1. float32 to uint8 (larger to smaller)
    list_of_inputs.append({
        'input': np.array([1.0, -2.0, 3.5], dtype=np.float32),
        'type': np.dtype('uint8'),
        'name': 'bitcast_1'
    })
    
    # 2. uint8 to float32 (smaller to larger)
    list_of_inputs.append({
        'input': np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8),
        'type': np.dtype('float32'),
        'name': 'bitcast_2'
    })
    
    # 3. int32 to float32 (equal size)
    list_of_inputs.append({
        'input': np.array([-1, 0, 5], dtype=np.int32),
        'type': np.dtype('float32'),
        'name': 'bitcast_3'
    })
    
    # 4. float64 to int64 (equal size)
    list_of_inputs.append({
        'input': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        'type': np.dtype('int64'),
        'name': 'bitcast_4'
    })
    
    # 5. int16 to uint8 (larger to smaller)
    list_of_inputs.append({
        'input': np.array([10, -10], dtype=np.int16),
        'type': np.dtype('uint8'),
        'name': 'bitcast_5'
    })
    
    # 6. uint8 to int16 (smaller to larger)
    list_of_inputs.append({
        'input': np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8),
        'type': np.dtype('int16'),
        'name': 'bitcast_6'
    })
    
    # 7. float32 to int32 (equal size, 3D)
    list_of_inputs.append({
        'input': np.array([[[1.5]]], dtype=np.float32),
        'type': np.dtype('int32'),
        'name': 'bitcast_7'
    })
    
    # 8. complex64 to float32 (larger to smaller)
    list_of_inputs.append({
        'input': np.array([1.0 + 2.0j], dtype=np.complex64),
        'type': np.dtype('float32'),
        'name': 'bitcast_8'
    })
    
    # 9. float32 to complex64 (smaller to larger)
    list_of_inputs.append({
        'input': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'type': np.dtype('complex64'),
        'name': 'bitcast_9'
    })
    
    # 10. int64 to uint8 (larger to smaller)
    list_of_inputs.append({
        'input': np.array([123456789], dtype=np.int64),
        'type': np.dtype('uint8'),
        'name': 'bitcast_10'
    })
    
    return list_of_inputs

generated_inputs["tf.bitcast"] = tf_bitcast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.bitcast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.bitcast'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.bitcast', generated_inputs['tf.bitcast'], lib="tf", suffix=0)
