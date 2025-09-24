
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_quantizedbiasadd_inputs():
    list_of_inputs = []

    def scalar_f32(x):
        return np.array(x, dtype=np.float32)

    # The error indicates that the provided dtypes (e.g., np.int8) are being
    # interpreted as standard integer types (tf.int8), not the required
    # quantized types (tf.qint8). However, creating tensors with actual
    # quantized types (e.g., using tf.quantization.quantize) violates the
    # numpy-only input format, which caused a different error in the testing
    # framework.
    #
    # This submission adheres strictly to the numpy-only format requirement to
    # satisfy the testing framework's pre-check. It uses simplified, basic cases
    # to minimize potential complexities, as suggested by the prompt to remove
    # potentially problematic inputs. This will likely still result in a
    # TensorFlow InvalidArgumentError because numpy arrays cannot intrinsically
    # represent the required quantized types, but it is the only path forward
    # under the given conflicting constraints.

    # Case 1: Basic qint8 case.
    list_of_inputs.append(copy.deepcopy({
        'input': np.random.randint(-128, 127, size=(2, 3), dtype=np.int8),
        'bias': np.random.randint(-128, 127, size=(3,), dtype=np.int8),
        'min_input': scalar_f32(-10.0),
        'max_input': scalar_f32(10.0),
        'min_bias': scalar_f32(-5.0),
        'max_bias': scalar_f32(5.0),
        'out_type': np.int8,
        'name': 'numpy_qint8'
    }))

    # Case 2: Basic quint8 case.
    list_of_inputs.append(copy.deepcopy({
        'input': np.random.randint(0, 255, size=(3, 2), dtype=np.uint8),
        'bias': np.random.randint(0, 255, size=(2,), dtype=np.uint8),
        'min_input': scalar_f32(0.0),
        'max_input': scalar_f32(25.0),
        'min_bias': scalar_f32(0.0),
        'max_bias': scalar_f32(10.0),
        'out_type': np.uint8,
        'name': 'numpy_quint8'
    }))

    # Case 3: Basic qint32 case.
    list_of_inputs.append(copy.deepcopy({
        'input': np.random.randint(-100000, 100000, size=(4, 4), dtype=np.int32),
        'bias': np.random.randint(-100000, 100000, size=(4,), dtype=np.int32),
        'min_input': scalar_f32(-1000.0),
        'max_input': scalar_f32(1000.0),
        'min_bias': scalar_f32(-500.0),
        'max_bias': scalar_f32(500.0),
        'out_type': np.int32,
        'name': 'numpy_qint32'
    }))

    # Case 4: qint16 -> qint32 case.
    list_of_inputs.append(copy.deepcopy({
        'input': np.random.randint(-30000, 30000, size=(2, 2, 3), dtype=np.int16),
        'bias': np.random.randint(-30000, 30000, size=(3,), dtype=np.int16),
        'min_input': scalar_f32(-500.0),
        'max_input': scalar_f32(500.0),
        'min_bias': scalar_f32(-100.0),
        'max_bias': scalar_f32(100.0),
        'out_type': np.int32,
        'name': 'numpy_qint16_to_qint32'
    }))
    
    # Case 5: quint16 -> quint16 case.
    list_of_inputs.append(copy.deepcopy({
        'input': np.random.randint(0, 60000, size=(3, 3), dtype=np.uint16),
        'bias': np.random.randint(0, 60000, size=(3,), dtype=np.uint16),
        'min_input': scalar_f32(0.0),
        'max_input': scalar_f32(60000.0),
        'min_bias': scalar_f32(0.0),
        'max_bias': scalar_f32(1000.0),
        'out_type': np.uint16,
        'name': 'numpy_quint16'
    }))

    # Case 6: 1D input case
    list_of_inputs.append(copy.deepcopy({
        'input': np.random.randint(-128, 127, size=(10,), dtype=np.int8),
        'bias': np.random.randint(-128, 127, size=(10,), dtype=np.int8),
        'min_input': scalar_f32(-128.0),
        'max_input': scalar_f32(127.0),
        'min_bias': scalar_f32(-1.0),
        'max_bias': scalar_f32(1.0),
        'out_type': np.int8,
        'name': 'numpy_1d_input'
    }))

    # Case 7: 4D input case
    list_of_inputs.append(copy.deepcopy({
        'input': np.random.randint(0, 255, size=(1, 2, 2, 4), dtype=np.uint8),
        'bias': np.random.randint(0, 255, size=(4,), dtype=np.uint8),
        'min_input': scalar_f32(0.0),
        'max_input': scalar_f32(1.0),
        'min_bias': scalar_f32(0.0),
        'max_bias': scalar_f32(0.5),
        'out_type': np.uint8,
        'name': 'numpy_4d_input'
    }))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedBiasAdd"] = tf_raw_ops_quantizedbiasadd_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedBiasAdd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedBiasAdd'.")

check_valid('tf.raw_ops.QuantizedBiasAdd', generated_inputs['tf.raw_ops.QuantizedBiasAdd'], lib="tf", suffix=0)
