
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_quantizedownandshrinkrange_inputs():
    list_of_inputs = []

    def _generate_quantized_input(target_type, shape=(4, 5)):
        """
        Generates a quantized tensor of a specific type and its float range.
        This helper now uses QuantizeDownAndShrinkRange itself to create
        intermediate quantized types like qint16/quint16 which are hard
        to generate due to missing CPU kernels in other ops.
        """
        with tf.device('/CPU:0'):
            # Step 1: Create a base qint32 tensor using QuantizedMatMul, which is reliable.
            original_shape = shape
            num_elements = np.prod(original_shape)
            shape_2d = (1, num_elements)
            K = 5
            
            a_shape = (shape_2d[0], K)
            b_shape = (K, shape_2d[1])
            
            a_float = np.random.uniform(low=0.0, high=1.0, size=a_shape).astype(np.float32)
            b_float = np.random.uniform(low=0.0, high=1.0, size=b_shape).astype(np.float32)

            a_quant, a_min, a_max = tf.quantization.quantize(a_float, 0.0, 1.0, T=tf.quint8)
            b_quant, b_min, b_max = tf.quantization.quantize(b_float, 0.0, 1.0, T=tf.quint8)

            qint32_out, qint32_min, qint32_max = tf.raw_ops.QuantizedMatMul(
                a=a_quant, b=b_quant, min_a=a_min, max_a=a_max, min_b=b_min, max_b=b_max, Toutput=tf.qint32
            )
            
            # Step 2: If the target type is not qint32, down-quantize to it.
            if target_type == tf.qint32:
                final_out = qint32_out
                final_min = qint32_min
                final_max = qint32_max
            else:
                # Use QuantizeDownAndShrinkRange to get the desired input type.
                final_out, final_min, final_max = tf.raw_ops.QuantizeDownAndShrinkRange(
                    input=qint32_out,
                    input_min=qint32_min,
                    input_max=qint32_max,
                    out_type=target_type
                )
            
            reshaped_out = tf.reshape(final_out, original_shape)
            
            return reshaped_out.numpy(), final_min.numpy(), final_max.numpy()

    # Valid combinations of (input_type, output_type) where output has lower or equal bit-depth.
    test_combinations = [
        (tf.qint32, tf.qint16),
        (tf.qint32, tf.quint16),
        (tf.qint32, tf.qint8),
        (tf.qint32, tf.quint8),
        (tf.qint16, tf.qint8),
        (tf.qint16, tf.quint8),
        (tf.quint16, tf.qint8),
        (tf.quint16, tf.quint8),
        (tf.qint8, tf.qint8), 
    ]

    for i, (in_type, out_type) in enumerate(test_combinations):
        # Generate the input tensor of type `in_type`
        input_val, input_min_val, input_max_val = _generate_quantized_input(in_type, shape=(4, 5))
        
        # Prepare the dictionary for the API call
        input_dict = {
            'input': input_val,
            'input_min': np.array(input_min_val, dtype=np.float32),
            'input_max': np.array(input_max_val, dtype=np.float32),
            'out_type': out_type,
            'name': f'test_{in_type.name}_to_{out_type.name}_{i}'
        }
        list_of_inputs.append(copy.deepcopy(input_dict))
        
    # Additional test cases with different shapes
    q_input, q_min, q_max = _generate_quantized_input(tf.qint32, shape=(1, 2, 3, 4))
    input_dict = {
        'input': q_input,
        'input_min': np.array(q_min, dtype=np.float32),
        'input_max': np.array(q_max, dtype=np.float32),
        'out_type': tf.qint8,
        'name': '4d_qint32_to_qint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    q_input, q_min, q_max = _generate_quantized_input(tf.quint16, shape=(1000,))
    input_dict = {
        'input': q_input,
        'input_min': np.array(q_min, dtype=np.float32),
        'input_max': np.array(q_max, dtype=np.float32),
        'out_type': tf.quint8,
        'name': 'large_1d_quint16_to_quint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizeDownAndShrinkRange"] = tf_raw_ops_quantizedownandshrinkrange_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizeDownAndShrinkRange' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizeDownAndShrinkRange'.")

check_valid('tf.raw_ops.QuantizeDownAndShrinkRange', generated_inputs['tf.raw_ops.QuantizeDownAndShrinkRange'], lib="tf", suffix=0)
