
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_QuantizedConv2D_inputs():
    list_of_inputs = []

    # Helper to create a single valid input dictionary
    def _create_input_dict(input_shape, filter_shape, input_tf_dtype, filter_tf_dtype,
                           strides, padding, out_tf_dtype, dilations, name=""):
        
        # Generate random float data to be quantized.
        input_float = (np.random.rand(*input_shape) * 255.0 - 128.0).astype(np.float32)
        filter_float = (np.random.rand(*filter_shape) * 10.0 - 5.0).astype(np.float32)

        min_range_input = np.min(input_float)
        max_range_input = np.max(input_float)
        min_range_filter = np.min(filter_float)
        max_range_filter = np.max(filter_float)
        
        # tf.quantization.quantize produces Tensors with the special qint/quint dtypes.
        # These TensorFlow Tensor objects must be passed directly to the op to avoid
        # the InvalidArgumentError, as converting them to numpy arrays strips the
        # necessary quantized type information.
        quantized_input, min_input_val, max_input_val = tf.quantization.quantize(
            input_float, min_range_input, max_range_input, T=input_tf_dtype, narrow_range=input_tf_dtype.is_signed)
        
        quantized_filter, min_filter_val, max_filter_val = tf.quantization.quantize(
            filter_float, min_range_filter, max_range_filter, T=filter_tf_dtype, narrow_range=filter_tf_dtype.is_signed)

        # The min/max values are returned as rank-0 tensors (scalars).
        # We convert them to numpy arrays to conform to the 'tensor' type in the signature.
        return {
            'input': quantized_input,
            'filter': quantized_filter,
            'min_input': np.array(min_input_val.numpy(), dtype=np.float32),
            'max_input': np.array(max_input_val.numpy(), dtype=np.float32),
            'min_filter': np.array(min_filter_val.numpy(), dtype=np.float32),
            'max_filter': np.array(max_filter_val.numpy(), dtype=np.float32),
            'strides': strides,
            'padding': padding,
            'out_type': out_tf_dtype,
            'dilations': dilations,
            'name': name
        }

    # Input 1: Basic qint8, VALID padding
    list_of_inputs.append(copy.deepcopy(
        _create_input_dict(
            input_shape=(1, 3, 3, 1), filter_shape=(2, 2, 1, 1),
            input_tf_dtype=tf.qint8, filter_tf_dtype=tf.qint8,
            strides=[1, 1, 1, 1], padding="VALID",
            out_tf_dtype=tf.qint32, dilations=[1, 1, 1, 1], name="basic_qint8"
        )
    ))

    # Input 2: quint8 with "SAME" padding
    list_of_inputs.append(copy.deepcopy(
        _create_input_dict(
            input_shape=(1, 4, 4, 2), filter_shape=(3, 3, 2, 4),
            input_tf_dtype=tf.quint8, filter_tf_dtype=tf.quint8,
            strides=[1, 1, 1, 1], padding="SAME",
            out_tf_dtype=tf.qint32, dilations=[1, 1, 1, 1], name="basic_quint8"
        )
    ))

    # Input 3: Strides > 1
    list_of_inputs.append(copy.deepcopy(
        _create_input_dict(
            input_shape=(1, 5, 5, 3), filter_shape=(3, 3, 3, 6),
            input_tf_dtype=tf.qint8, filter_tf_dtype=tf.qint8,
            strides=[1, 2, 2, 1], padding="VALID",
            out_tf_dtype=tf.qint32, dilations=[1, 1, 1, 1], name="strided"
        )
    ))

    # Input 4: Dilations > 1
    list_of_inputs.append(copy.deepcopy(
        _create_input_dict(
            input_shape=(1, 7, 7, 1), filter_shape=(2, 2, 1, 2),
            input_tf_dtype=tf.quint8, filter_tf_dtype=tf.quint8,
            strides=[1, 1, 1, 1], padding="VALID",
            out_tf_dtype=tf.qint32, dilations=[1, 2, 2, 1], name="dilated"
        )
    ))

    # Input 5: Different out_type (quint8)
    list_of_inputs.append(copy.deepcopy(
        _create_input_dict(
            input_shape=(1, 2, 2, 4), filter_shape=(1, 1, 4, 2),
            input_tf_dtype=tf.quint8, filter_tf_dtype=tf.quint8,
            strides=[1, 1, 1, 1], padding="SAME",
            out_tf_dtype=tf.quint8, dilations=[1, 1, 1, 1], name="out_type_quint8"
        )
    ))
    
    # Input 6: qint16 type
    list_of_inputs.append(copy.deepcopy(
        _create_input_dict(
            input_shape=(1, 6, 6, 2), filter_shape=(2, 2, 2, 4),
            input_tf_dtype=tf.qint16, filter_tf_dtype=tf.qint16,
            strides=[1, 2, 2, 1], padding="VALID",
            out_tf_dtype=tf.qint32, dilations=[1, 1, 1, 1], name="qint16_type"
        )
    ))

    # Input 7: quint16 type
    list_of_inputs.append(copy.deepcopy(
        _create_input_dict(
            input_shape=(1, 5, 5, 3), filter_shape=(3, 3, 3, 5),
            input_tf_dtype=tf.quint16, filter_tf_dtype=tf.quint16,
            strides=[1, 1, 1, 1], padding="SAME",
            out_tf_dtype=tf.qint32, dilations=[1, 1, 1, 1], name="quint16_type"
        )
    ))
    
    # Input 8: Mixed precision (qint32 input, qint8 filter)
    list_of_inputs.append(copy.deepcopy(
        _create_input_dict(
            input_shape=(1, 3, 3, 1), filter_shape=(2, 2, 1, 2),
            input_tf_dtype=tf.qint32, filter_tf_dtype=tf.qint8,
            strides=[1, 1, 1, 1], padding="VALID",
            out_tf_dtype=tf.qint32, dilations=[1, 1, 1, 1], name="mixed_precision"
        )
    ))

    # Input 9: Combination of Strides and Dilations
    list_of_inputs.append(copy.deepcopy(
        _create_input_dict(
            input_shape=(1, 10, 10, 1), filter_shape=(3, 3, 1, 2),
            input_tf_dtype=tf.qint8, filter_tf_dtype=tf.qint8,
            strides=[1, 2, 2, 1], padding="SAME",
            out_tf_dtype=tf.qint32, dilations=[1, 2, 2, 1], name="strided_dilated"
        )
    ))

    # Input 10: Non-square filters and images
    list_of_inputs.append(copy.deepcopy(
        _create_input_dict(
            input_shape=(1, 5, 7, 2), filter_shape=(2, 3, 2, 4),
            input_tf_dtype=tf.qint8, filter_tf_dtype=tf.qint8,
            strides=[1, 1, 1, 1], padding="VALID",
            out_tf_dtype=tf.qint32, dilations=[1, 1, 1, 1], name="non_square"
        )
    ))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedConv2D"] = tf_raw_ops_QuantizedConv2D_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedConv2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedConv2D'.")

check_valid('tf.raw_ops.QuantizedConv2D', generated_inputs['tf.raw_ops.QuantizedConv2D'], lib="tf", suffix=0)
