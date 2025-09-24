
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_decode_raw_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.DecodeRaw.
    """
    list_of_inputs = []

    # Input 1: Decode to uint8
    input_dict = {
        'bytes': np.array([b'\x01\x02\x03\x04', b'\x05\x06\x07\x08'], dtype=object),
        'out_type': np.uint8,
        'little_endian': True,
        'name': 'decode_uint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Decode to int16, little-endian
    b1_i16 = np.array([10, -20], dtype=np.int16).tobytes()
    b2_i16 = np.array([300, -400], dtype=np.int16).tobytes()
    input_dict = {
        'bytes': np.array([b1_i16, b2_i16], dtype=object),
        'out_type': np.int16,
        'little_endian': True,
        'name': 'decode_int16_little'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Decode to int16, big-endian
    input_dict = {
        'bytes': np.array([b1_i16, b2_i16], dtype=object),
        'out_type': np.int16,
        'little_endian': False,
        'name': 'decode_int16_big'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Decode to float32
    b_f32 = np.array([1.0, 3.14, -2.71], dtype=np.float32).tobytes()
    input_dict = {
        'bytes': np.array([b_f32], dtype=object),
        'out_type': np.float32,
        'little_endian': True,
        'name': 'decode_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D bytes tensor, decode to int32
    b1_i32 = np.array([1, -1], dtype=np.int32).tobytes()
    b2_i32 = np.array([1000, -1000], dtype=np.int32).tobytes()
    b3_i32 = np.array([2**30, -(2**30)], dtype=np.int32).tobytes()
    b4_i32 = np.array([0, 0], dtype=np.int32).tobytes()
    input_dict = {
        'bytes': np.array([[b1_i32, b2_i32], [b3_i32, b4_i32]], dtype=object),
        'out_type': np.int32,
        'little_endian': True,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Decode to bool
    input_dict = {
        'bytes': np.array([b'\x01\x00\x01\x01\x00', b'\x00\x00\x01\x00\x01'], dtype=object),
        'out_type': np.bool_,
        'little_endian': True,
        'name': 'decode_bools'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Scalar bytes tensor (0-D), decode to float64
    b_f64 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64).tobytes()
    input_dict = {
        'bytes': np.array(b_f64, dtype=object),
        'out_type': np.float64,
        'little_endian': True,
        'name': 'scalar_bytes_to_vector'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Decode to complex64
    b_c64 = np.array([1+2j, 3-4j], dtype=np.complex64).tobytes()
    input_dict = {
        'bytes': np.array([b_c64], dtype=object),
        'out_type': np.complex64,
        'little_endian': True,
        'name': 'decode_complex64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Decode to complex128, big-endian
    b_c128 = np.array([1.23+4.56j, -9.87-6.54j], dtype=np.complex128).tobytes()
    input_dict = {
        'bytes': np.array([b_c128, b_c128], dtype=object),
        'out_type': np.complex128,
        'little_endian': False,
        'name': 'decode_complex128_big_endian'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Decode to uint16
    b1_u16 = np.array([65535, 1024, 0], dtype=np.uint16).tobytes()
    b2_u16 = b'\x00\x01\x02\x03\x04\x05'
    input_dict = {
        'bytes': np.array([b1_u16, b2_u16], dtype=object),
        'out_type': np.uint16,
        'little_endian': True,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Decode to int64
    b_i64 = np.array([2**60, -2**60], dtype=np.int64).tobytes()
    input_dict = {
        'bytes': np.array([b_i64], dtype=object),
        'out_type': np.int64,
        'little_endian': True,
        'name': 'decode_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Decode to half (float16)
    b_f16 = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float16).tobytes()
    input_dict = {
        'bytes': np.array([b_f16], dtype=object),
        'out_type': np.float16,
        'little_endian': True,
        'name': 'decode_float16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DecodeRaw"] = tf_raw_ops_decode_raw_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DecodeRaw' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeRaw'.")

check_valid('tf.raw_ops.DecodeRaw', generated_inputs['tf.raw_ops.DecodeRaw'], lib="tf", suffix=0)
