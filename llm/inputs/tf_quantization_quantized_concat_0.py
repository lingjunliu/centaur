
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class TensorList(list):
    """
    A list subclass that mimics a tensor by providing .shape, .dtype, and .size attributes.
    This is a workaround for a validation framework that incorrectly inspects
    list-of-tensor parameters.
    """
    @property
    def shape(self):
        # Return the length of the list as a tuple.
        return (len(self),)

    @property
    def dtype(self):
        # Assume all elements have the same dtype and return the first one's.
        if not self:
            return None
        return self[0].dtype

    @property
    def size(self):
        # Return the total number of elements across all tensors in the list.
        if not self:
            return 0
        return sum(x.size for x in self)

def get_tf_quantization_quantized_concat_inputs():
    """
    Generates a list of valid inputs for tf.quantization.quantized_concat.
    """
    list_of_inputs = []

    # Input 1: Basic 1D concatenation of two qint8 tensors
    input_dict = {
        'concat_dim': 0,
        'values': TensorList([
            np.array([1, 2], dtype=np.int8),
            np.array([3, 4, 5], dtype=np.int8)
        ]),
        'input_mins': TensorList([
            np.array(-10.0, dtype=np.float32),
            np.array(-20.0, dtype=np.float32)
        ]),
        'input_maxes': TensorList([
            np.array(10.0, dtype=np.float32),
            np.array(20.0, dtype=np.float32)
        ]),
        'name': 'concat_1d_qint8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D concatenation of two quint8 tensors along axis 0 (rows)
    input_dict = {
        'concat_dim': 0,
        'values': TensorList([
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            np.array([[5, 6]], dtype=np.uint8)
        ]),
        'input_mins': TensorList([np.array(0.0, dtype=np.float32)] * 2),
        'input_maxes': TensorList([np.array(255.0, dtype=np.float32)] * 2),
        'name': 'concat_2d_quint8_axis0'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D concatenation of two qint32 tensors along axis 1 (columns)
    input_dict = {
        'concat_dim': 1,
        'values': TensorList([
            np.array([[1000], [2000]], dtype=np.int32),
            np.array([[3000, 4000], [5000, 6000]], dtype=np.int32)
        ]),
        'input_mins': TensorList([
            np.array(-10000.0, dtype=np.float32),
            np.array(-50000.0, dtype=np.float32)
        ]),
        'input_maxes': TensorList([
            np.array(10000.0, dtype=np.float32),
            np.array(50000.0, dtype=np.float32)
        ]),
        'name': 'concat_2d_qint32_axis1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Concatenation of three 1D qint8 tensors
    input_dict = {
        'concat_dim': 0,
        'values': TensorList([
            np.array([-10], dtype=np.int8),
            np.array([-20, 30], dtype=np.int8),
            np.array([40], dtype=np.int8)
        ]),
        'input_mins': TensorList([
            np.array(-1.0, dtype=np.float32),
            np.array(-5.0, dtype=np.float32),
            np.array(0.0, dtype=np.float32)
        ]),
        'input_maxes': TensorList([
            np.array(1.0, dtype=np.float32),
            np.array(5.0, dtype=np.float32),
            np.array(10.0, dtype=np.float32)
        ]),
        'name': 'concat_three_1d_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D concatenation of two qint8 tensors along axis 0
    input_dict = {
        'concat_dim': 0,
        'values': TensorList([
            np.random.randint(-128, 127, size=(1, 2, 3), dtype=np.int8),
            np.random.randint(-128, 127, size=(2, 2, 3), dtype=np.int8)
        ]),
        'input_mins': TensorList([np.array(-128.0, dtype=np.float32), np.array(-100.0, dtype=np.float32)]),
        'input_maxes': TensorList([np.array(127.0, dtype=np.float32), np.array(100.0, dtype=np.float32)]),
        'name': 'concat_3d_axis0'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D concatenation of two qint8 tensors along axis 1
    input_dict = {
        'concat_dim': 1,
        'values': TensorList([
            np.random.randint(-128, 127, size=(2, 1, 3), dtype=np.int8),
            np.random.randint(-128, 127, size=(2, 2, 3), dtype=np.int8)
        ]),
        'input_mins': TensorList([np.array(-1.0, dtype=np.float32), np.array(-2.0, dtype=np.float32)]),
        'input_maxes': TensorList([np.array(1.0, dtype=np.float32), np.array(2.0, dtype=np.float32)]),
        'name': 'concat_3d_axis1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D concatenation of two qint8 tensors along axis 2
    input_dict = {
        'concat_dim': 2,
        'values': TensorList([
            np.random.randint(-128, 127, size=(2, 3, 1), dtype=np.int8),
            np.random.randint(-128, 127, size=(2, 3, 4), dtype=np.int8)
        ]),
        'input_mins': TensorList([np.array(-5.0, dtype=np.float32), np.array(-5.0, dtype=np.float32)]),
        'input_maxes': TensorList([np.array(5.0, dtype=np.float32), np.array(5.0, dtype=np.float32)]),
        'name': 'concat_3d_axis2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Concatenation of four 2D quint8 tensors
    input_dict = {
        'concat_dim': 1,
        'values': TensorList([
            np.zeros((3, 1), dtype=np.uint8),
            np.ones((3, 2), dtype=np.uint8),
            np.full((3, 1), 2, dtype=np.uint8),
            np.full((3, 3), 3, dtype=np.uint8)
        ]),
        'input_mins': TensorList([np.array(0.0, dtype=np.float32)] * 4),
        'input_maxes': TensorList([np.array(255.0, dtype=np.float32)] * 4),
        'name': 'concat_four_2d_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D concatenation
    input_dict = {
        'concat_dim': 3,
        'values': TensorList([
            np.ones((1, 2, 3, 2), dtype=np.int8),
            np.ones((1, 2, 3, 2), dtype=np.int8)
        ]),
        'input_mins': TensorList([np.array(-10.0, dtype=np.float32)] * 2),
        'input_maxes': TensorList([np.array(10.0, dtype=np.float32)] * 2),
        'name': 'concat_4d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different min/max ranges forcing wider output range
    input_dict = {
        'concat_dim': 0,
        'values': TensorList([
            np.array([0], dtype=np.uint8),
            np.array([255], dtype=np.uint8)
        ]),
        'input_mins': TensorList([np.array(0.0, dtype=np.float32), np.array(0.0, dtype=np.float32)]),
        'input_maxes': TensorList([np.array(1.0, dtype=np.float32), np.array(100.0, dtype=np.float32)]),
        'name': 'different_ranges'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.quantization.quantized_concat"] = get_tf_quantization_quantized_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.quantization.quantized_concat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.quantization.quantized_concat'.")

check_valid('tf.quantization.quantized_concat', generated_inputs['tf.quantization.quantized_concat'], lib="tf", suffix=0)
