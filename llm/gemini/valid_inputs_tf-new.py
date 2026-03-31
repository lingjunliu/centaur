generated_inputs = {}

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_unsorted_segment_prod_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    name = "segment_prod_1"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], dtype=np.int32)
    segment_ids = np.array([0, 1, 0], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    name = "segment_prod_2"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    name = "segment_prod_3"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    segment_ids = np.array([0, 1, 0, 1, 2], dtype=np.int32)
    num_segments = np.array(3, dtype=np.int32)
    name = "segment_prod_4"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    name = "segment_prod_5"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 1, 0, 1], dtype=np.int64)
    num_segments = np.array(2, dtype=np.int64)
    name = "segment_prod_6"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    segment_ids = np.array([0, 1, 0], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    name = "segment_prod_7"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    num_segments = np.array(3, dtype=np.int32)
    name = "segment_prod_8"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    name = "segment_prod_9"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    data = np.array([1, 2, 3, 4], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 2], dtype=np.int32)
    num_segments = np.array(3, dtype=np.int32)
    name = "segment_prod_10"

    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.math.unsorted_segment_prod"] = tf_math_unsorted_segment_prod_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_quantization_fake_quant_with_min_max_vars_inputs():
    list_of_inputs = []

    # Input 1
    inputs = np.array([-1.0, -0.5, 0.0, 0.3, 0.8, 1.0], dtype=np.float32)
    min_val = np.array(-1.0, dtype=np.float32)
    max_val = np.array(1.0, dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test_quant1"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = np.array([[1.2, -0.3, 0.7], [2.1, 0.5, -1.0]], dtype=np.float32)
    min_val = np.array(-0.5, dtype=np.float32)
    max_val = np.array(0.8, dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test_quant2"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(10.0, dtype=np.float32)
    num_bits = 4
    narrow_range = True
    name = "test_quant3"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float32)
    min_val = np.array(-5.0, dtype=np.float32)
    max_val = np.array(5.0, dtype=np.float32)
    num_bits = 16
    narrow_range = False
    name = "test_quant4"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = np.array([1.0], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(1.0, dtype=np.float32)
    num_bits = 2
    narrow_range = True
    name = "test_quant5"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = np.array([[-1.0, 0.5], [0.2, -0.8]], dtype=np.float32)
    min_val = np.array(-1.0, dtype=np.float32)
    max_val = np.array(0.5, dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test_quant6"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    min_val = np.array(1.0, dtype=np.float32)
    max_val = np.array(5.0, dtype=np.float32)
    num_bits = 8
    narrow_range = True
    name = "test_quant7"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    inputs = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    min_val = np.array(-4.0, dtype=np.float32)
    max_val = np.array(-1.0, dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test_quant8"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    min_val = np.array(1.0, dtype=np.float32)
    max_val = np.array(4.0, dtype=np.float32)
    num_bits = 5
    narrow_range = True
    name = "test_quant9"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(1.0, dtype=np.float32)
    num_bits = 10
    narrow_range = False
    name = "test_quant10"

    input_dict = {
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.quantization.fake_quant_with_min_max_vars"] = tf_quantization_fake_quant_with_min_max_vars_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_not_equal_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([1, 4, 3], dtype=np.int32)
    name = None

    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[1, 2], [5, 4]], dtype=np.int64)
    name = "not_equal_op"

    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.0, 2.1, 3.0], dtype=np.float32)
    name = None

    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([True, False, True], dtype=np.bool_)
    y = np.array([True, True, False], dtype=np.bool_)
    name = "bool_comparison"

    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([1, 5, 3], dtype=np.int32)
    name = None

    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    name = "broadcast_comparison"

    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([-1, -2, -3], dtype=np.int32)
    y = np.array([1, -2, 3], dtype=np.int32)
    name = None

    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    y = np.array([1.1, 2.2, 3.4], dtype=np.float64)
    name = "float64_comparison"

    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([1, 5, 3], dtype=np.uint8)
    name = None

    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[1, 2], [3, 4]], dtype=np.int16)
    y = np.array([[5, 6], [7, 8]], dtype=np.int16)
    name = "int16_comparison"

    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.math.not_equal"] = tf_math_not_equal_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_pow_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 0.5], dtype=np.float32)
    name = "power_example_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer values
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([2, 3, 4], dtype=np.int32)
    name = "power_example_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([[2.0, 0.5], [3.0, 1.0]], dtype=np.float64)
    name = "power_example_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative exponent
    x = np.array([2.0, 3.0], dtype=np.float32)
    y = np.array([-1.0, -2.0], dtype=np.float32)
    name = "power_example_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex numbers
    x = np.array([1 + 1j, 2 - 1j], dtype=np.complex64)
    y = np.array([2, 0.5], dtype=np.complex64)
    name = "power_example_5"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger integers
    x = np.array([1000, 2000], dtype=np.int64)
    y = np.array([2, 3], dtype=np.int64)
    name = "power_example_6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different shapes
    x = np.array([2.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    name = "power_example_7"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float16
    x = np.array([2.0, 3.0], dtype=np.float16)
    y = np.array([2.0, 0.5], dtype=np.float16)
    name = "power_example_8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D Tensor
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    y = np.array([[[2, 1], [0.5, 1]], [[1, 2], [1, 0.5]]], dtype=np.float32)
    name = "power_example_9"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex128 numbers
    x = np.array([1 + 1j, 2 - 1j], dtype=np.complex128)
    y = np.array([2, 0.5], dtype=np.complex128)
    name = "power_example_10"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.math.pow"] = tf_math_pow_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_reduce_prod_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1, 2, 3, 4], dtype=np.float32)
    axis = None
    keepdims = False
    name = "prod_1"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.float32)
    axis = 0
    keepdims = False
    name = "prod_2"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.float32)
    axis = 1
    keepdims = True
    name = "prod_3"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axis = 0
    keepdims = False
    name = "prod_4"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axis = 1
    keepdims = True
    name = "prod_5"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axis = 2
    keepdims = False
    name = "prod_6"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7 - negative axis
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.float32)
    axis = -1
    keepdims = False
    name = "prod_7"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - all negative values
    input_tensor = np.array([[-1, -2], [-3, -4]], dtype=np.float32)
    axis = None
    keepdims = False
    name = "prod_8"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - combination of positive and negative numbers
    input_tensor = np.array([[-1, 2], [3, -4]], dtype=np.float32)
    axis = 0
    keepdims = True
    name = "prod_9"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - 3D tensor with negative axis
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axis = -1
    keepdims = False
    name = "prod_10"
    input_dict = {"input_tensor": input_tensor, "axis": axis, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.math.reduce_prod"] = tf_math_reduce_prod_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_constant_inputs():
    list_of_inputs = []

    # Input 1
    value = np.array([1, 2, 3], dtype=np.int32)
    dtype = np.int32
    shape = None
    name = "const1"
    input_dict = {"value": value, "dtype": dtype, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dtype = np.float32
    shape = None
    name = "const2"
    input_dict = {"value": value, "dtype": dtype, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.array(5, dtype=np.int64)
    dtype = np.int64
    shape = (2, 2)
    name = "const3"
    input_dict = {"value": value, "dtype": dtype, "shape": np.array(shape), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    dtype = np.float64
    shape = (3, 2)
    name = "const4"
    input_dict = {"value": value, "dtype": dtype, "shape": np.array(shape), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    dtype = np.int32
    shape = None
    name = "const5"
    input_dict = {"value": value, "dtype": dtype, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    value = np.array([1.5, 2.5, 3.5], dtype=np.float16)
    dtype = np.float16
    shape = None
    name = "const6"
    input_dict = {"value": value, "dtype": dtype, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    value = np.array([True, False, True], dtype=np.bool_)
    dtype = np.bool_
    shape = None
    name = "const7"
    input_dict = {"value": value, "dtype": dtype, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    value = np.array([1, 2, 3, 4], dtype=np.int8)
    dtype = np.int8
    shape = (2, 2)
    name = "const8"
    input_dict = {"value": value, "dtype": dtype, "shape": np.array(shape), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    value = np.array([0], dtype=np.int32)
    dtype = np.int32
    shape = (5,5)
    name = "const9"
    input_dict = {"value": value, "dtype": dtype, "shape": np.array(shape), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    value = np.array([1+1j, 2+2j], dtype=np.complex128)
    dtype = np.complex128
    shape = None
    name = "const10"
    input_dict = {"value": value, "dtype": dtype, "shape": shape, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    for input_dict in list_of_inputs:
        if input_dict['dtype'] is not None:
            input_dict['dtype'] = tf.as_dtype(input_dict['dtype'])
    
    return list_of_inputs


generated_inputs["tf.constant"] = tf_constant_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_range_inputs():
    list_of_inputs = []

    # Input 1
    start = tf.constant(3, dtype=tf.int32)
    limit = tf.constant(18, dtype=tf.int32)
    delta = tf.constant(3, dtype=tf.int32)
    dtype = tf.int32
    name = "range1"
    input_dict = {"start": start, "limit": limit, "delta": delta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    start = tf.constant(3.0, dtype=tf.float32)
    limit = tf.constant(1.0, dtype=tf.float32)
    delta = tf.constant(-0.5, dtype=tf.float32)
    dtype = tf.float32
    name = "range2"
    input_dict = {"start": start, "limit": limit, "delta": delta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    start = tf.constant(0, dtype=tf.int32)
    limit = tf.constant(5, dtype=tf.int32)
    delta = tf.constant(1, dtype=tf.int32)
    dtype = tf.int32
    name = "range3"
    input_dict = {"start": start, "limit": limit, "delta": delta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    start = tf.constant(-5, dtype=tf.int32)
    limit = tf.constant(5, dtype=tf.int32)
    delta = tf.constant(2, dtype=tf.int32)
    dtype = tf.int32
    name = "range4"
    input_dict = {"start": start, "limit": limit, "delta": delta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    start = tf.constant(0, dtype=tf.float32)
    limit = tf.constant(10, dtype=tf.float32)
    delta = tf.constant(0.5, dtype=tf.float32)
    dtype = tf.float32
    name = "range5"
    input_dict = {"start": start, "limit": limit, "delta": delta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    start = tf.constant(10, dtype=tf.int64)
    limit = tf.constant(0, dtype=tf.int64)
    delta = tf.constant(-1, dtype=tf.int64)
    dtype = tf.int64
    name = "range6"
    input_dict = {"start": start, "limit": limit, "delta": delta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    start = tf.constant(-10, dtype=tf.float64)
    limit = tf.constant(-5, dtype=tf.float64)
    delta = tf.constant(0.2, dtype=tf.float64)
    dtype = tf.float64
    name = "range7"
    input_dict = {"start": start, "limit": limit, "delta": delta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    start = tf.constant(0, dtype=tf.uint8)
    limit = tf.constant(10, dtype=tf.uint8)
    delta = tf.constant(1, dtype=tf.uint8)
    dtype = tf.int32 #tf.uint8 gives error, changing to int32
    name = "range9"
    input_dict = {"start": start, "limit": limit, "delta": delta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    start = tf.constant(1, dtype=tf.int8)
    limit = tf.constant(5, dtype=tf.int8)
    delta = tf.constant(1, dtype=tf.int8)
    dtype = tf.int32 #tf.int8 gives error, changing to int32
    name = "range10"
    input_dict = {"start": start, "limit": limit, "delta": delta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    start = tf.constant(0, dtype=tf.int32)
    limit = tf.constant(5, dtype=tf.int32)
    delta = tf.constant(1, dtype=tf.int32)
    dtype = None
    name = "range11"
    input_dict = {"start": start, "limit": limit, "delta": delta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.range"] = tf_range_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_split_inputs():
    list_of_inputs = []

    # Input 1
    value = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    num_or_size_splits = 2
    axis = 1
    num = None
    name = "split_example_1"
    input_dict = {"value": value, "num_or_size_splits": num_or_size_splits, "axis": axis, "num": num, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    num_or_size_splits = 2
    axis = 0
    num = None
    name = "split_example_2"
    input_dict = {"value": value, "num_or_size_splits": num_or_size_splits, "axis": axis, "num": num, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = np.array([1, 2, 3, 4, 5, 6])
    num_or_size_splits = np.int32(3)
    axis = np.int32(0)
    num = None
    name = "split_example_3"
    input_dict = {"value": value, "num_or_size_splits": num_or_size_splits, "axis": axis, "num": num, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    num_or_size_splits = 3
    axis = 0
    num = None
    name = "split_example_4"
    input_dict = {"value": value, "num_or_size_splits": num_or_size_splits, "axis": axis, "num": num, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    num_or_size_splits = 2
    axis = 1
    num = None
    name = "split_example_5"
    input_dict = {"value": value, "num_or_size_splits": num_or_size_splits, "axis": axis, "num": num, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    value = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    num_or_size_splits = 5
    axis = 1
    num = None
    name = "split_example_6"
    input_dict = {"value": value, "num_or_size_splits": num_or_size_splits, "axis": axis, "num": num, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    value = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    num_or_size_splits = 4
    axis = 0
    num = None
    name = "split_example_7"
    input_dict = {"value": value, "num_or_size_splits": num_or_size_splits, "axis": axis, "num": num, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    value = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
    num_or_size_splits = 3
    axis = 1
    num = None
    name = "split_example_8"
    input_dict = {"value": value, "num_or_size_splits": num_or_size_splits, "axis": axis, "num": num, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    value = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    num_or_size_splits = 2
    axis = 1
    num = 2
    name = "split_example_9"
    input_dict = {"value": value, "num_or_size_splits": num_or_size_splits, "axis": axis, "num": num, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    value = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    num_or_size_splits = 5
    axis = 0
    num = None
    name = "split_example_10"
    input_dict = {"value": value, "num_or_size_splits": num_or_size_splits, "axis": axis, "num": num, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    num_or_size_splits = 3
    axis = 0
    num = None
    name = "split_example_11"
    input_dict = {"value": value, "num_or_size_splits": num_or_size_splits, "axis": axis, "num": num, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    value = np.array([[1, 2, 3], [4, 5, 6]])
    num_or_size_splits = 1
    axis = 0
    num = None
    name = "split_example_12"
    input_dict = {"value": value, "num_or_size_splits": num_or_size_splits, "axis": axis, "num": num, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs


generated_inputs["tf.split"] = tf_split_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_where_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array
    condition = np.array([True, False, True, False], dtype=np.bool_)
    input_dict = {"condition": condition, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D boolean array
    condition = np.array([[True, False], [False, True]], dtype=np.bool_)
    input_dict = {"condition": condition, "name": "bool_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D boolean array
    condition = np.array([[[True, False], [False, True]], [[False, True], [True, False]]], dtype=np.bool_)
    input_dict = {"condition": condition, "name": "bool_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D integer array (treated as boolean)
    condition = np.array([1, 0, 1, 0], dtype=np.int32)
    input_dict = {"condition": condition, "name": "int_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float array (treated as boolean)
    condition = np.array([[0.1, 0.0], [0.0, 0.5]], dtype=np.float32)
    input_dict = {"condition": condition, "name": "float_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array with negative integer
    condition = np.array([-1, 0, 1, 0], dtype=np.int32)
    input_dict = {"condition": condition, "name": "neg_int_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with mixed positive and negative floats
    condition = np.array([[1.5, -0.5], [-0.1, 2.0]], dtype=np.float64)
    input_dict = {"condition": condition, "name": "mixed_float_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D integer array (treated as boolean)
    condition = np.array([[[1, 0], [0, 2]], [[0, 3], [4, 0]]], dtype=np.int64)
    input_dict = {"condition": condition, "name": "int_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty boolean array
    condition = np.array([], dtype=np.bool_)
    input_dict = {"condition": condition, "name": "empty_bool"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-dimensional array with a mix of positive and negative numbers
    condition = np.array([[[1, -1], [0, 1]], [[-2, 1], [1, 0]]], dtype=np.int32)
    input_dict = {"condition": condition, "name": "multi_dimensional"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.where"] = tf_where_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_lrn_grad_inputs():
    list_of_inputs = []

    # Input 1
    input_grads = np.random.rand(1, 5, 5, 3).astype(np.float32)
    input_image = np.random.rand(1, 5, 5, 3).astype(np.float32)
    output_image = np.random.rand(1, 5, 5, 3).astype(np.float32)
    depth_radius = 5
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "test_lrn_grad_1"
    input_dict = {"input_grads": input_grads, "input_image": input_image, "output_image": output_image, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_grads = np.random.rand(2, 10, 10, 5).astype(np.float32)
    input_image = np.random.rand(2, 10, 10, 5).astype(np.float32)
    output_image = np.random.rand(2, 10, 10, 5).astype(np.float32)
    depth_radius = 3
    bias = 0.5
    alpha = 2.0
    beta = 0.25
    name = "test_lrn_grad_2"
    input_dict = {"input_grads": input_grads, "input_image": input_image, "output_image": output_image, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_grads = np.random.rand(1, 3, 3, 1).astype(np.float32)
    input_image = np.random.rand(1, 3, 3, 1).astype(np.float32)
    output_image = np.random.rand(1, 3, 3, 1).astype(np.float32)
    depth_radius = 1
    bias = 0.1
    alpha = 0.5
    beta = 1.0
    name = "test_lrn_grad_3"
    input_dict = {"input_grads": input_grads, "input_image": input_image, "output_image": output_image, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_grads = np.random.rand(4, 8, 8, 4).astype(np.float32)
    input_image = np.random.rand(4, 8, 8, 4).astype(np.float32)
    output_image = np.random.rand(4, 8, 8, 4).astype(np.float32)
    depth_radius = 7
    bias = 2.0
    alpha = 0.25
    beta = 0.75
    name = "test_lrn_grad_4"
    input_dict = {"input_grads": input_grads, "input_image": input_image, "output_image": output_image, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_grads = np.random.rand(1, 5, 5, 3).astype(np.float16)
    input_image = np.random.rand(1, 5, 5, 3).astype(np.float16)
    output_image = np.random.rand(1, 5, 5, 3).astype(np.float16)
    depth_radius = 5
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "test_lrn_grad_5"
    input_dict = {"input_grads": input_grads, "input_image": input_image, "output_image": output_image, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    input_grads = np.random.rand(1, 5, 5, 3).astype(np.float16)
    input_image = np.random.rand(1, 5, 5, 3).astype(np.float16)
    output_image = np.random.rand(1, 5, 5, 3).astype(np.float16)
    depth_radius = 2
    bias = 0.8
    alpha = 0.7
    beta = 0.6
    name = "test_lrn_grad_6"
    input_dict = {"input_grads": input_grads, "input_image": input_image, "output_image": output_image, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_grads = np.random.rand(2, 4, 4, 2).astype(np.float16)
    input_image = np.random.rand(2, 4, 4, 2).astype(np.float16)
    output_image = np.random.rand(2, 4, 4, 2).astype(np.float16)
    depth_radius = 4
    bias = 1.5
    alpha = 1.2
    beta = 0.9
    name = "test_lrn_grad_7"
    input_dict = {"input_grads": input_grads, "input_image": input_image, "output_image": output_image, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_grads = np.random.rand(1, 1, 1, 1).astype(np.float16)
    input_image = np.random.rand(1, 1, 1, 1).astype(np.float16)
    output_image = np.random.rand(1, 1, 1, 1).astype(np.float16)
    depth_radius = 5
    bias = 1.0
    alpha = 1.0
    beta = 0.5
    name = "test_lrn_grad_8"
    input_dict = {"input_grads": input_grads, "input_image": input_image, "output_image": output_image, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_grads = np.random.rand(3, 7, 7, 3).astype(np.float32)
    input_image = np.random.rand(3, 7, 7, 3).astype(np.float32)
    output_image = np.random.rand(3, 7, 7, 3).astype(np.float32)
    depth_radius = 2
    bias = 0.3
    alpha = 0.9
    beta = 0.1
    name = "test_lrn_grad_9"
    input_dict = {"input_grads": input_grads, "input_image": input_image, "output_image": output_image, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_grads = np.random.rand(1, 6, 6, 2).astype(np.float32)
    input_image = np.random.rand(1, 6, 6, 2).astype(np.float32)
    output_image = np.random.rand(1, 6, 6, 2).astype(np.float32)
    depth_radius = 6
    bias = 1.8
    alpha = 0.4
    beta = 0.8
    name = "test_lrn_grad_10"
    input_dict = {"input_grads": input_grads, "input_image": input_image, "output_image": output_image, "depth_radius": depth_radius, "bias": bias, "alpha": alpha, "beta": beta, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.raw_ops.LRNGrad"] = tf_raw_ops_lrn_grad_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_StridedSliceGrad_inputs():
    list_of_inputs = []

    # Input 1
    shape = np.array([5, 5], dtype=np.int32)
    begin = np.array([1, 1], dtype=np.int32)
    end = np.array([4, 4], dtype=np.int32)
    strides = np.array([1, 1], dtype=np.int32)
    dy = np.random.rand(3, 3).astype(np.float32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    name = "strided_slice_grad_1"

    input_dict = {
        "shape": shape,
        "begin": begin,
        "end": end,
        "strides": strides,
        "dy": dy,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = np.array([10, 10, 10], dtype=np.int32)
    begin = np.array([2, 2, 2], dtype=np.int32)
    end = np.array([8, 8, 8], dtype=np.int32)
    strides = np.array([1, 1, 1], dtype=np.int32)
    dy = np.random.rand(6, 6, 6).astype(np.float32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    name = "strided_slice_grad_2"

    input_dict = {
        "shape": shape,
        "begin": begin,
        "end": end,
        "strides": strides,
        "dy": dy,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = np.array([4, 4, 4, 4], dtype=np.int32)
    begin = np.array([0, 0, 0, 0], dtype=np.int32)
    end = np.array([4, 4, 4, 4], dtype=np.int32)
    strides = np.array([2, 2, 2, 2], dtype=np.int32)
    dy = np.random.rand(2, 2, 2, 2).astype(np.float32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    name = "strided_slice_grad_3"

    input_dict = {
        "shape": shape,
        "begin": begin,
        "end": end,
        "strides": strides,
        "dy": dy,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = np.array([5, 5], dtype=np.int64)
    begin = np.array([1, 1], dtype=np.int64)
    end = np.array([4, 4], dtype=np.int64)
    strides = np.array([1, 1], dtype=np.int64)
    dy = np.random.rand(3, 3).astype(np.float64)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    name = "strided_slice_grad_4"

    input_dict = {
        "shape": shape,
        "begin": begin,
        "end": end,
        "strides": strides,
        "dy": dy,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = np.array([5, 5], dtype=np.int32)
    begin = np.array([1, 1], dtype=np.int32)
    end = np.array([4, 4], dtype=np.int32)
    strides = np.array([1, 1], dtype=np.int32)
    dy = np.random.rand(3, 3).astype(np.float32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    name = "strided_slice_grad_5"

    input_dict = {
        "shape": shape,
        "begin": begin,
        "end": end,
        "strides": strides,
        "dy": dy,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = np.array([5, 5], dtype=np.int32)
    begin = np.array([1, 1], dtype=np.int32)
    end = np.array([4, 4], dtype=np.int32)
    strides = np.array([1, 1], dtype=np.int32)
    dy = np.random.rand(3, 3).astype(np.float32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    name = "strided_slice_grad_6"

    input_dict = {
        "shape": shape,
        "begin": begin,
        "end": end,
        "strides": strides,
        "dy": dy,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    shape = np.array([5, 5], dtype=np.int32)
    begin = np.array([1, 1], dtype=np.int32)
    end = np.array([4, 4], dtype=np.int32)
    strides = np.array([1, 1], dtype=np.int32)
    dy = np.random.rand(3, 3).astype(np.float32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    name = "strided_slice_grad_7"

    input_dict = {
        "shape": shape,
        "begin": begin,
        "end": end,
        "strides": strides,
        "dy": dy,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = np.array([5, 5], dtype=np.int32)
    begin = np.array([1, 1], dtype=np.int32)
    end = np.array([4, 4], dtype=np.int32)
    strides = np.array([1, 1], dtype=np.int32)
    dy = np.random.rand(3, 3).astype(np.float32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    name = "strided_slice_grad_8"

    input_dict = {
        "shape": shape,
        "begin": begin,
        "end": end,
        "strides": strides,
        "dy": dy,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = np.array([5, 5], dtype=np.int32)
    begin = np.array([1, 1], dtype=np.int32)
    end = np.array([4, 4], dtype=np.int32)
    strides = np.array([1, 1], dtype=np.int32)
    dy = np.random.rand(3, 3).astype(np.float32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    name = "strided_slice_grad_9"

    input_dict = {
        "shape": shape,
        "begin": begin,
        "end": end,
        "strides": strides,
        "dy": dy,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = np.array([2, 3, 4], dtype=np.int32)
    begin = np.array([0, 1, 2], dtype=np.int32)
    end = np.array([2, 3, 4], dtype=np.int32)
    strides = np.array([1, 1, 1], dtype=np.int32)
    dy = np.random.rand(2, 2, 2).astype(np.float32)
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    name = "strided_slice_grad_10"

    input_dict = {
        "shape": shape,
        "begin": begin,
        "end": end,
        "strides": strides,
        "dy": dy,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.raw_ops.StridedSliceGrad"] = tf_raw_ops_StridedSliceGrad_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_keras_layers_conv2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(4, 10, 10, 3).astype(np.float32)
    input_dict = {
        "filters": 32,
        "kernel_size": (3, 3),
        "strides": (1, 1),
        "padding": "valid",
        "data_format": "channels_last",
        "dilation_rate": (1, 1),
        "groups": 1,
        "activation": "relu",
        "use_bias": True,
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(2, 20, 20, 64).astype(np.float32)
    input_dict = {
        "filters": 16,
        "kernel_size": (5, 5),
        "strides": (2, 2),
        "padding": "same",
        "data_format": "channels_last",
        "dilation_rate": (1, 1),
        "groups": 1,
        "activation": "sigmoid",
        "use_bias": False,
        "kernel_initializer": "he_normal",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(1, 28, 28, 1).astype(np.float32)
    input_dict = {
        "filters": 8,
        "kernel_size": (3, 3),
        "strides": (1, 1),
        "padding": "same",
        "data_format": "channels_last",
        "dilation_rate": (1, 1),
        "groups": 1,
        "activation": "tanh",
        "use_bias": True,
        "kernel_initializer": "truncated_normal",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(8, 16, 16, 32).astype(np.float32)
    input_dict = {
        "filters": 64,
        "kernel_size": (4, 4),
        "strides": (2, 2),
        "padding": "valid",
        "data_format": "channels_last",
        "dilation_rate": (1, 1),
        "groups": 1,
        "activation": None,
        "use_bias": True,
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(4, 32, 32, 128).astype(np.float32)
    input_dict = {
        "filters": 32,
        "kernel_size": (3, 3),
        "strides": (1, 1),
        "padding": "same",
        "data_format": "channels_last",
        "dilation_rate": (2, 2),
        "groups": 1,
        "activation": "relu",
        "use_bias": True,
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(2, 64, 64, 3).astype(np.float32)
    input_dict = {
        "filters": 16,
        "kernel_size": (5, 5),
        "strides": (2, 2),
        "padding": "valid",
        "data_format": "channels_last",
        "dilation_rate": (1, 1),
        "groups": 1,
        "activation": "sigmoid",
        "use_bias": False,
        "kernel_initializer": "he_normal",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.rand(1, 128, 128, 1).astype(np.float32)
    input_dict = {
        "filters": 8,
        "kernel_size": (3, 3),
        "strides": (1, 1),
        "padding": "same",
        "data_format": "channels_last",
        "dilation_rate": (1, 1),
        "groups": 1,
        "activation": "tanh",
        "use_bias": True,
        "kernel_initializer": "truncated_normal",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(8, 256, 256, 32).astype(np.float32)
    input_dict = {
        "filters": 64,
        "kernel_size": (4, 4),
        "strides": (2, 2),
        "padding": "valid",
        "data_format": "channels_last",
        "dilation_rate": (1, 1),
        "groups": 1,
        "activation": None,
        "use_bias": True,
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(1, 10, 10, 3).astype(np.float32)
    input_dict = {
        "filters": 32,
        "kernel_size": (3, 3),
        "strides": (1, 1),
        "padding": "valid",
        "data_format": "channels_first",
        "dilation_rate": (1, 1),
        "groups": 1,
        "activation": "relu",
        "use_bias": True,
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    input_tensor = np.random.rand(4, 20, 20, 64).astype(np.float32)
    input_dict = {
        "filters": 16,
        "kernel_size": (5, 5),
        "strides": (2, 2),
        "padding": "same",
        "data_format": "channels_first",
        "dilation_rate": (1, 1),
        "groups": 1,
        "activation": "sigmoid",
        "use_bias": False,
        "kernel_initializer": "he_normal",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.keras.layers.Conv2D"] = tf_keras_layers_conv2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_keras_layers_dense_inputs():
    list_of_inputs = []

    # Input 1
    units = 32
    activation = "relu"
    use_bias = True
    kernel_initializer = "glorot_uniform"
    bias_initializer = "zeros"
    kernel_regularizer = None
    bias_regularizer = None
    activity_regularizer = None
    kernel_constraint = None
    bias_constraint = None
    lora_rank = None
    input_tensor = np.random.rand(10, 10).astype(np.float32)

    input_dict = {
        "units": units,
        "activation": activation,
        "use_bias": use_bias,
        "kernel_initializer": kernel_initializer,
        "bias_initializer": bias_initializer,
        "kernel_regularizer": kernel_regularizer,
        "bias_regularizer": bias_regularizer,
        "activity_regularizer": activity_regularizer,
        "kernel_constraint": kernel_constraint,
        "bias_constraint": bias_constraint,
        "lora_rank": lora_rank,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    units = 64
    activation = "sigmoid"
    use_bias = False
    kernel_initializer = "he_normal"
    bias_initializer = "zeros"
    kernel_regularizer = None
    bias_regularizer = None
    activity_regularizer = None
    kernel_constraint = None
    bias_constraint = None
    lora_rank = 8
    input_tensor = np.random.rand(5, 20).astype(np.float32)

    input_dict = {
        "units": units,
        "activation": activation,
        "use_bias": use_bias,
        "kernel_initializer": kernel_initializer,
        "bias_initializer": bias_initializer,
        "kernel_regularizer": kernel_regularizer,
        "bias_regularizer": bias_regularizer,
        "activity_regularizer": activity_regularizer,
        "kernel_constraint": kernel_constraint,
        "bias_constraint": bias_constraint,
        "lora_rank": lora_rank,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    units = 128
    activation = "tanh"
    use_bias = True
    kernel_initializer = "lecun_uniform"
    bias_initializer = "zeros"
    kernel_regularizer = None
    bias_regularizer = None
    activity_regularizer = None
    kernel_constraint = None
    bias_constraint = None
    lora_rank = None
    input_tensor = np.random.rand(2, 15, 30).astype(np.float32)

    input_dict = {
        "units": units,
        "activation": activation,
        "use_bias": use_bias,
        "kernel_initializer": kernel_initializer,
        "bias_initializer": bias_initializer,
        "kernel_regularizer": kernel_regularizer,
        "bias_regularizer": bias_regularizer,
        "activity_regularizer": activity_regularizer,
        "kernel_constraint": kernel_constraint,
        "bias_constraint": bias_constraint,
        "lora_rank": lora_rank,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.keras.layers.Dense"] = tf_keras_layers_dense_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_keras_layers_flatten_inputs():
    list_of_inputs = []

    # Input 1: channels_last, 2D input
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]])
    data_format = "channels_last"
    input_dict = {"data_format": data_format, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: channels_first, 3D input
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    data_format = "channels_first"
    input_dict = {"data_format": data_format, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: channels_last, 4D input
    input_tensor = np.random.rand(2, 3, 4, 5)
    data_format = "channels_last"
    input_dict = {"data_format": data_format, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: channels_first, 4D input
    input_tensor = np.random.rand(2, 5, 3, 4)
    data_format = "channels_first"
    input_dict = {"data_format": data_format, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: channels_last, 5D input
    input_tensor = np.random.rand(2, 3, 4, 5, 6)
    data_format = "channels_last"
    input_dict = {"data_format": data_format, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: channels_first, 5D input
    input_tensor = np.random.rand(2, 6, 3, 4, 5)
    data_format = "channels_first"
    input_dict = {"data_format": data_format, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: channels_last, 1D input
    input_tensor = np.array([1, 2, 3, 4, 5])
    data_format = "channels_last"
    input_dict = {"data_format": data_format, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: channels_first, 1D input (should behave the same as channels_last for 1D)
    input_tensor = np.array([1, 2, 3, 4, 5])
    data_format = "channels_first"
    input_dict = {"data_format": data_format, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: channels_last, 3D input, different shape
    input_tensor = np.random.rand(5, 2, 7)
    data_format = "channels_last"
    input_dict = {"data_format": data_format, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: channels_first, 3D input, different shape
    input_tensor = np.random.rand(5, 7, 2)
    data_format = "channels_first"
    input_dict = {"data_format": data_format, "input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.keras.layers.Flatten"] = tf_keras_layers_flatten_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_keras_layers_lstm_inputs():
    list_of_inputs = []

    # Input 1
    units = 32
    activation = 'tanh'
    recurrent_activation = 'sigmoid'
    use_bias = True
    kernel_initializer = 'glorot_uniform'
    recurrent_initializer = 'orthogonal'
    bias_initializer = 'zeros'
    unit_forget_bias = True
    kernel_regularizer = None
    recurrent_regularizer = None
    bias_regularizer = None
    activity_regularizer = None
    kernel_constraint = None
    recurrent_constraint = None
    bias_constraint = None
    dropout = 0.0
    recurrent_dropout = 0.0
    seed = None
    return_sequences = False
    return_state = False
    go_backwards = False
    stateful = False
    unroll = False
    use_cudnn = 'auto'
    input_tensor = np.random.random((32, 10, 8)).astype(np.float32)
    mask = None
    training = False
    initial_state = None

    input_dict = {
        "units": units,
        "activation": activation,
        "recurrent_activation": recurrent_activation,
        "use_bias": use_bias,
        "kernel_initializer": kernel_initializer,
        "recurrent_initializer": recurrent_initializer,
        "bias_initializer": bias_initializer,
        "unit_forget_bias": unit_forget_bias,
        "kernel_regularizer": kernel_regularizer,
        "recurrent_regularizer": recurrent_regularizer,
        "bias_regularizer": bias_regularizer,
        "activity_regularizer": activity_regularizer,
        "kernel_constraint": kernel_constraint,
        "recurrent_constraint": recurrent_constraint,
        "bias_constraint": bias_constraint,
        "dropout": dropout,
        "recurrent_dropout": recurrent_dropout,
        "seed": seed,
        "return_sequences": return_sequences,
        "return_state": return_state,
        "go_backwards": go_backwards,
        "stateful": stateful,
        "unroll": unroll,
        "use_cudnn": use_cudnn,
        "input": input_tensor,
        "mask": mask,
        "training": training,
        "initial_state": initial_state
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    units = 64
    activation = 'relu'
    recurrent_activation = 'hard_sigmoid'
    use_bias = False
    kernel_initializer = 'he_normal'
    recurrent_initializer = 'identity'
    bias_initializer = 'zeros'  # Changed from 'ones' to 'zeros'
    unit_forget_bias = False
    kernel_regularizer = None
    recurrent_regularizer = None
    bias_regularizer = None
    activity_regularizer = None
    kernel_constraint = None
    recurrent_constraint = None
    bias_constraint = None
    dropout = 0.2
    recurrent_dropout = 0.1
    seed = 42
    return_sequences = True
    return_state = True
    go_backwards = True
    stateful = True
    unroll = False
    use_cudnn = 'auto'
    input_tensor = np.random.random((16, 20, 16)).astype(np.float32)
    mask = np.random.randint(0, 2, size=(16, 20)).astype(np.bool_)
    training = True
    initial_state = None

    input_dict = {
        "units": units,
        "activation": activation,
        "recurrent_activation": recurrent_activation,
        "use_bias": use_bias,
        "kernel_initializer": kernel_initializer,
        "recurrent_initializer": recurrent_initializer,
        "bias_initializer": bias_initializer,
        "unit_forget_bias": unit_forget_bias,
        "kernel_regularizer": kernel_regularizer,
        "recurrent_regularizer": recurrent_regularizer,
        "bias_regularizer": bias_regularizer,
        "activity_regularizer": activity_regularizer,
        "kernel_constraint": kernel_constraint,
        "recurrent_constraint": recurrent_constraint,
        "bias_constraint": bias_constraint,
        "dropout": dropout,
        "recurrent_dropout": recurrent_dropout,
        "seed": seed,
        "return_sequences": return_sequences,
        "return_state": return_state,
        "go_backwards": go_backwards,
        "stateful": stateful,
        "unroll": unroll,
        "use_cudnn": use_cudnn,
        "input": input_tensor,
        "mask": mask,
        "training": training,
        "initial_state": initial_state
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    units = 128
    activation = None
    recurrent_activation = None
    use_bias = True
    kernel_initializer = 'truncated_normal'
    recurrent_initializer = 'glorot_normal'
    bias_initializer = 'zeros' # Changed from constant to zeros
    unit_forget_bias = True
    kernel_regularizer = None
    recurrent_regularizer = None
    bias_regularizer = None
    activity_regularizer = None
    kernel_constraint = None
    recurrent_constraint = None
    bias_constraint = None
    dropout = 0.5
    recurrent_dropout = 0.3
    seed = 123
    return_sequences = False
    return_state = False
    go_backwards = False
    stateful = False
    unroll = False
    use_cudnn = 'auto'
    input_tensor = np.random.random((64, 5, 32)).astype(np.float32)
    mask = None
    training = False
    initial_state = None

    input_dict = {
        "units": units,
        "activation": activation,
        "recurrent_activation": recurrent_activation,
        "use_bias": use_bias,
        "kernel_initializer": kernel_initializer,
        "recurrent_initializer": recurrent_initializer,
        "bias_initializer": bias_initializer,
        "unit_forget_bias": unit_forget_bias,
        "kernel_regularizer": kernel_regularizer,
        "recurrent_regularizer": recurrent_regularizer,
        "bias_regularizer": bias_regularizer,
        "activity_regularizer": activity_regularizer,
        "kernel_constraint": kernel_constraint,
        "recurrent_constraint": recurrent_constraint,
        "bias_constraint": bias_constraint,
        "dropout": dropout,
        "recurrent_dropout": recurrent_dropout,
        "seed": seed,
        "return_sequences": return_sequences,
        "return_state": return_state,
        "go_backwards": go_backwards,
        "stateful": stateful,
        "unroll": unroll,
        "use_cudnn": use_cudnn,
        "input": input_tensor,
        "mask": mask,
        "training": training,
        "initial_state": initial_state
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    units = 16
    activation = 'sigmoid'
    recurrent_activation = 'tanh'
    use_bias = True
    kernel_initializer = 'lecun_uniform'
    recurrent_initializer = 'orthogonal'
    bias_initializer = 'zeros'
    unit_forget_bias = True
    kernel_regularizer = None
    recurrent_regularizer = None
    bias_regularizer = None
    activity_regularizer = None
    kernel_constraint = None
    recurrent_constraint = None
    bias_constraint = None
    dropout = 0.1
    recurrent_dropout = 0.05
    seed = 7
    return_sequences = True
    return_state = False
    go_backwards = True
    stateful = False
    unroll = False
    use_cudnn = 'auto'
    input_tensor = np.random.random((8, 15, 4)).astype(np.float32)
    mask = None
    training = True
    initial_state = None

    input_dict = {
        "units": units,
        "activation": activation,
        "recurrent_activation": recurrent_activation,
        "use_bias": use_bias,
        "kernel_initializer": kernel_initializer,
        "recurrent_initializer": recurrent_initializer,
        "bias_initializer": bias_initializer,
        "unit_forget_bias": unit_forget_bias,
        "kernel_regularizer": kernel_regularizer,
        "recurrent_regularizer": recurrent_regularizer,
        "bias_regularizer": bias_regularizer,
        "activity_regularizer": activity_regularizer,
        "kernel_constraint": kernel_constraint,
        "recurrent_constraint": recurrent_constraint,
        "bias_constraint": bias_constraint,
        "dropout": dropout,
        "recurrent_dropout": recurrent_dropout,
        "seed": seed,
        "return_sequences": return_sequences,
        "return_state": return_state,
        "go_backwards": go_backwards,
        "stateful": stateful,
        "unroll": unroll,
        "use_cudnn": use_cudnn,
        "input": input_tensor,
        "mask": mask,
        "training": training,
        "initial_state": initial_state
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    units = 256
    activation = 'hard_sigmoid'
    recurrent_activation = 'relu'
    use_bias = False
    kernel_initializer = 'zeros'
    recurrent_initializer = 'orthogonal'
    bias_initializer = 'zeros' # changed from 'ones' to 'zeros'
    unit_forget_bias = False
    kernel_regularizer = None
    recurrent_regularizer = None
    bias_regularizer = None
    activity_regularizer = None
    kernel_constraint = None
    recurrent_constraint = None
    bias_constraint = None
    dropout = 0.0
    recurrent_dropout = 0.0
    seed = None
    return_sequences = False
    return_state = True
    go_backwards = False
    stateful = True
    unroll = False
    use_cudnn = 'auto'
    input_tensor = np.random.random((1, 30, 64)).astype(np.float32)
    mask = np.random.randint(0, 2, size=(1, 30)).astype(np.bool_)
    training = False
    initial_state = None

    input_dict = {
        "units": units,
        "activation": activation,
        "recurrent_activation": recurrent_activation,
        "use_bias": use_bias,
        "kernel_initializer": kernel_initializer,
        "recurrent_initializer": recurrent_initializer,
        "bias_initializer": bias_initializer,
        "unit_forget_bias": unit_forget_bias,
        "kernel_regularizer": kernel_regularizer,
        "recurrent_regularizer": recurrent_regularizer,
        "bias_regularizer": bias_regularizer,
        "activity_regularizer": activity_regularizer,
        "kernel_constraint": kernel_constraint,
        "recurrent_constraint": recurrent_constraint,
        "bias_constraint": bias_constraint,
        "dropout": dropout,
        "recurrent_dropout": recurrent_dropout,
        "seed": seed,
        "return_sequences": return_sequences,
        "return_state": return_state,
        "go_backwards": go_backwards,
        "stateful": stateful,
        "unroll": unroll,
        "use_cudnn": use_cudnn,
        "input": input_tensor,
        "mask": mask,
        "training": training,
        "initial_state": initial_state
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    units = 8
    activation = 'tanh'
    recurrent_activation = 'sigmoid'
    use_bias = True
    kernel_initializer = 'glorot_uniform'
    recurrent_initializer = 'orthogonal'
    bias_initializer = 'zeros'
    unit_forget_bias = True
    kernel_regularizer = None
    recurrent_regularizer = None
    bias_regularizer = None
    activity_regularizer = None
    kernel_constraint = None
    recurrent_constraint = None
    bias_constraint = None
    dropout = 0.0
    recurrent_dropout = 0.0
    seed = None
    return_sequences = False
    return_state = False
    go_backwards = False
    stateful = False
    unroll = False
    use_cudnn = 'auto'
    input_tensor = np.random.random((128, 3, 2)).astype(np.float32)
    mask = None
    training = False
    initial_state = None

    input_dict = {
        "units": units,
        "activation": activation,
        "recurrent_activation": recurrent_activation,
        "use_bias": use_bias,
        "kernel_initializer": kernel_initializer,
        "recurrent_initializer": recurrent_initializer,
        "bias_initializer": bias_initializer,
        "unit_forget_bias": unit_forget_bias,
        "kernel_regularizer": kernel_regularizer,
        "recurrent_regularizer": recurrent_regularizer,
        "bias_regularizer": bias_regularizer,
        "activity_regularizer": activity_regularizer,
        "kernel_constraint": kernel_constraint,
        "recurrent_constraint": recurrent_constraint,
        "bias_constraint": bias_constraint,
        "dropout": dropout,
        "recurrent_dropout": recurrent_dropout,
        "seed": seed,
        "return_sequences": return_sequences,
        "return_state": return_state,
        "go_backwards": go_backwards,
        "stateful": stateful,
        "unroll": unroll,
        "use_cudnn": use_cudnn,
        "input": input_tensor,
        "mask": mask,
        "training": training,
        "initial_state": initial_state
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    units = 4
    activation = 'relu'
    recurrent_activation = 'hard_sigmoid'
    use_bias = False
    kernel_initializer = 'he_normal'
    recurrent_initializer = 'identity'
    bias_initializer = 'zeros' # changed from 'ones' to 'zeros'
    unit_forget_bias = False
    kernel_regularizer = None
    recurrent_regularizer = None
    bias_regularizer = None
    activity_regularizer = None
    kernel_constraint = None
    recurrent_constraint = None
    bias_constraint = None
    dropout = 0.2
    recurrent_dropout = 0.1
    seed = 42
    return_sequences = True
    return_state = True
    go_backwards = True
    stateful = True
    unroll = False
    use_cudnn = 'auto'
    input_tensor = np.random.random((2, 100, 2)).astype(np.float32)
    mask = np.random.randint(0, 2, size=(2, 100)).astype(np.bool_)
    training = True
    initial_state = None

    input_dict = {
        "units": units,
        "activation": activation,
        "recurrent_activation": recurrent_activation,
        "use_bias": use_bias,
        "kernel_initializer": kernel_initializer,
        "recurrent_initializer": recurrent_initializer,
        "bias_initializer": bias_initializer,
        "unit_forget_bias": unit_forget_bias,
        "kernel_regularizer": kernel_regularizer,
        "recurrent_regularizer": recurrent_regularizer,
        "bias_regularizer": bias_regularizer,
        "activity_regularizer": activity_regularizer,
        "kernel_constraint": kernel_constraint,
        "recurrent_constraint": recurrent_constraint,
        "bias_constraint": bias_constraint,
        "dropout": dropout,
        "recurrent_dropout": recurrent_dropout,
        "seed": seed,
        "return_sequences": return_sequences,
        "return_state": return_state,
        "go_backwards": go_backwards,
        "stateful": stateful,
        "unroll": unroll,
        "use_cudnn": use_cudnn,
        "input": input_tensor,
        "mask": mask,
        "training": training,
        "initial_state": initial_state
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    units = 512
    activation = None
    recurrent_activation = None
    use_bias = True
    kernel_initializer = 'truncated_normal'
    recurrent_initializer = 'glorot_normal'
    bias_initializer = 'zeros' # changed from 'constant' to 'zeros'
    unit_forget_bias = True
    kernel_regularizer = None
    recurrent_regularizer = None
    bias_regularizer = None
    activity_regularizer = None
    kernel_constraint = None
    recurrent_constraint = None
    bias_constraint = None
    dropout = 0.5
    recurrent_dropout = 0.3
    seed = 123
    return_sequences = False
    return_state = False
    go_backwards = False
    stateful = False
    unroll = False
    use_cudnn = 'auto'
    input_tensor = np.random.random((4, 1, 128)).astype(np.float32)
    mask = None
    training = False
    initial_state = None

    input_dict = {
        "units": units,
        "activation": activation,
        "recurrent_activation": recurrent_activation,
        "use_bias": use_bias,
        "kernel_initializer": kernel_initializer,
        "recurrent_initializer": recurrent_initializer,
        "bias_initializer": bias_initializer,
        "unit_forget_bias": unit_forget_bias,
        "kernel_regularizer": kernel_regularizer,
        "recurrent_regularizer": recurrent_regularizer,
        "bias_regularizer": bias_regularizer,
        "activity_regularizer": activity_regularizer,
        "kernel_constraint": kernel_constraint,
        "recurrent_constraint": recurrent_constraint,
        "bias_constraint": bias_constraint,
        "dropout": dropout,
        "recurrent_dropout": recurrent_dropout,
        "seed": seed,
        "return_sequences": return_sequences,
        "return_state": return_state,
        "go_backwards": go_backwards,
        "stateful": stateful,
        "unroll": unroll,
        "use_cudnn": use_cudnn,
        "input": input_tensor,
        "mask": mask,
        "training": training,
        "initial_state": initial_state
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    units = 1
    activation = 'sigmoid'
    recurrent_activation = 'tanh'
    use_bias = True
    kernel_initializer = 'lecun_uniform'
    recurrent_initializer = 'orthogonal'
    bias_initializer = 'zeros'
    unit_forget_bias = True
    kernel_regularizer = None
    recurrent_regularizer = None
    bias_regularizer = None
    activity_regularizer = None
    kernel_constraint = None
    recurrent_constraint = None
    bias_constraint = None
    dropout = 0.1
    recurrent_dropout = 0.05
    seed = 7
    return_sequences = True
    return_state = False
    go_backwards = True
    stateful = False
    unroll = False
    use_cudnn = 'auto'
    input_tensor = np.random.random((256, 2, 1)).astype(np.float32)
    mask = None
    training = True
    initial_state = None

    input_dict = {
        "units": units,
        "activation": activation,
        "recurrent_activation": recurrent_activation,
        "use_bias": use_bias,
        "kernel_initializer": kernel_initializer,
        "recurrent_initializer": recurrent_initializer,
        "bias_initializer": bias_initializer,
        "unit_forget_bias": unit_forget_bias,
        "kernel_regularizer": kernel_regularizer,
        "recurrent_regularizer": recurrent_regularizer,
        "bias_regularizer": bias_regularizer,
        "activity_regularizer": activity_regularizer,
        "kernel_constraint": kernel_constraint,
        "recurrent_constraint": recurrent_constraint,
        "bias_constraint": bias_constraint,
        "dropout": dropout,
        "recurrent_dropout": recurrent_dropout,
        "seed": seed,
        "return_sequences": return_sequences,
        "return_state": return_state,
        "go_backwards": go_backwards,
        "stateful": stateful,
        "unroll": unroll,
        "use_cudnn": use_cudnn,
        "input": input_tensor,
        "mask": mask,
        "training": training,
        "initial_state": initial_state
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    units = 1024
    activation = 'hard_sigmoid'
    recurrent_activation = 'relu'
    use_bias = False
    kernel_initializer = 'zeros'
    recurrent_initializer = 'orthogonal'
    bias_initializer = 'zeros' # changed from 'ones' to 'zeros'
    unit_forget_bias = False
    kernel_regularizer = None
    recurrent_regularizer = None
    bias_regularizer = None
    activity_regularizer = None
    kernel_constraint = None
    recurrent_constraint = None
    bias_constraint = None
    dropout = 0.0
    recurrent_dropout = 0.0
    seed = None
    return_sequences = False
    return_state = True
    go_backwards = False
    stateful = True
    unroll = False
    use_cudnn = 'auto'
    input_tensor = np.random.random((1, 1, 1024)).astype(np.float32)
    mask = np.random.randint(0, 2, size=(1, 1)).astype(np.bool_)
    training = False
    initial_state = None

    input_dict = {
        "units": units,
        "activation": activation,
        "recurrent_activation": recurrent_activation,
        "use_bias": use_bias,
        "kernel_initializer": kernel_initializer,
        "recurrent_initializer": recurrent_initializer,
        "bias_initializer": bias_initializer,
        "unit_forget_bias": unit_forget_bias,
        "kernel_regularizer": kernel_regularizer,
        "recurrent_regularizer": recurrent_regularizer,
        "bias_regularizer": bias_regularizer,
        "activity_regularizer": activity_regularizer,
        "kernel_constraint": kernel_constraint,
        "recurrent_constraint": recurrent_constraint,
        "bias_constraint": bias_constraint,
        "dropout": dropout,
        "recurrent_dropout": recurrent_dropout,
        "seed": seed,
        "return_sequences": return_sequences,
        "return_state": return_state,
        "go_backwards": go_backwards,
        "stateful": stateful,
        "unroll": unroll,
        "use_cudnn": use_cudnn,
        "input": input_tensor,
        "mask": mask,
        "training": training,
        "initial_state": initial_state
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.keras.layers.LSTM"] = tf_keras_layers_lstm_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_keras_layers_batchnormalization_inputs():
    list_of_inputs = []

    # Input 1
    axis = -1
    momentum = 0.99
    epsilon = 0.001
    center = True
    scale = True
    beta_initializer = 'zeros'
    gamma_initializer = 'ones'
    moving_mean_initializer = 'zeros'
    moving_variance_initializer = 'ones'
    beta_regularizer = None
    gamma_regularizer = None
    beta_constraint = None
    gamma_constraint = None
    synchronized = False
    name = 'batch_norm_1'
    dtype = tf.float32
    input_tensor = np.random.rand(10, 5).astype(np.float32)
    training = True
    mask = None

    input_dict = {
        "axis": axis,
        "momentum": momentum,
        "epsilon": epsilon,
        "center": center,
        "scale": scale,
        "beta_initializer": beta_initializer,
        "gamma_initializer": gamma_initializer,
        "moving_mean_initializer": moving_mean_initializer,
        "moving_variance_initializer": moving_variance_initializer,
        "beta_regularizer": beta_regularizer,
        "gamma_regularizer": gamma_regularizer,
        "beta_constraint": beta_constraint,
        "gamma_constraint": gamma_constraint,
        "synchronized": synchronized,
        "name": name,
        "dtype": dtype,
        "input": input_tensor,
        "training": training,
        "mask": mask
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    axis = 1
    momentum = 0.9
    epsilon = 0.01
    center = False
    scale = False
    beta_initializer = 'glorot_uniform'
    gamma_initializer = 'he_normal'
    moving_mean_initializer = 'zeros'
    moving_variance_initializer = 'ones'
    beta_regularizer = None
    gamma_regularizer = None
    beta_constraint = None
    gamma_constraint = None
    synchronized = True
    name = 'batch_norm_2'
    dtype = tf.float64
    input_tensor = np.random.rand(5, 10, 5).astype(np.float64)
    training = False
    mask = np.random.randint(0, 2, size=(5, 10)).astype(bool)

    input_dict = {
        "axis": axis,
        "momentum": momentum,
        "epsilon": epsilon,
        "center": center,
        "scale": scale,
        "beta_initializer": beta_initializer,
        "gamma_initializer": gamma_initializer,
        "moving_mean_initializer": moving_mean_initializer,
        "moving_variance_initializer": moving_variance_initializer,
        "beta_regularizer": beta_regularizer,
        "gamma_regularizer": gamma_regularizer,
        "beta_constraint": beta_constraint,
        "gamma_constraint": gamma_constraint,
        "synchronized": synchronized,
        "name": name,
        "dtype": dtype,
        "input": input_tensor,
        "training": training,
        "mask": mask
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    axis = -1
    momentum = 0.999
    epsilon = 0.00001
    center = True
    scale = False
    beta_initializer = 'zeros'
    gamma_initializer = 'ones'
    moving_mean_initializer = 'zeros'
    moving_variance_initializer = 'ones'
    beta_regularizer = None
    gamma_regularizer = None
    beta_constraint = None
    gamma_constraint = None
    synchronized = False
    name = 'batch_norm_3'
    dtype = tf.float32
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    training = True
    mask = None

    input_dict = {
        "axis": axis,
        "momentum": momentum,
        "epsilon": epsilon,
        "center": center,
        "scale": scale,
        "beta_initializer": beta_initializer,
        "gamma_initializer": gamma_initializer,
        "moving_mean_initializer": moving_mean_initializer,
        "moving_variance_initializer": moving_variance_initializer,
        "beta_regularizer": beta_regularizer,
        "gamma_regularizer": gamma_regularizer,
        "beta_constraint": beta_constraint,
        "gamma_constraint": gamma_constraint,
        "synchronized": synchronized,
        "name": name,
        "dtype": dtype,
        "input": input_tensor,
        "training": training,
        "mask": mask
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    axis = 2
    momentum = 0.8
    epsilon = 0.1
    center = False
    scale = True
    beta_initializer = 'he_uniform'
    gamma_initializer = 'glorot_normal'
    moving_mean_initializer = 'zeros'
    moving_variance_initializer = 'ones'
    beta_regularizer = None
    gamma_regularizer = None
    beta_constraint = None
    gamma_constraint = None
    synchronized = True
    name = 'batch_norm_4'
    dtype = tf.float64
    input_tensor = np.random.rand(1, 5, 10, 5, 2).astype(np.float64)
    training = False
    mask = np.random.randint(0, 2, size=(1, 5, 10, 5)).astype(bool)

    input_dict = {
        "axis": axis,
        "momentum": momentum,
        "epsilon": epsilon,
        "center": center,
        "scale": scale,
        "beta_initializer": beta_initializer,
        "gamma_initializer": gamma_initializer,
        "moving_mean_initializer": moving_mean_initializer,
        "moving_variance_initializer": moving_variance_initializer,
        "beta_regularizer": beta_regularizer,
        "gamma_regularizer": gamma_regularizer,
        "beta_constraint": beta_constraint,
        "gamma_constraint": gamma_constraint,
        "synchronized": synchronized,
        "name": name,
        "dtype": dtype,
        "input": input_tensor,
        "training": training,
        "mask": mask
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    axis = -1
    momentum = 0.5
    epsilon = 0.0000001
    center = True
    scale = True
    beta_initializer = 'zeros'
    gamma_initializer = 'ones'
    moving_mean_initializer = 'zeros'
    moving_variance_initializer = 'ones'
    beta_regularizer = None
    gamma_regularizer = None
    beta_constraint = None
    gamma_constraint = None
    synchronized = False
    name = 'batch_norm_5'
    dtype = tf.float32
    input_tensor = np.random.rand(20,).astype(np.float32)
    training = True
    mask = None

    input_dict = {
        "axis": axis,
        "momentum": momentum,
        "epsilon": epsilon,
        "center": center,
        "scale": scale,
        "beta_initializer": beta_initializer,
        "gamma_initializer": gamma_initializer,
        "moving_mean_initializer": moving_mean_initializer,
        "moving_variance_initializer": moving_variance_initializer,
        "beta_regularizer": beta_regularizer,
        "gamma_regularizer": gamma_regularizer,
        "beta_constraint": beta_constraint,
        "gamma_constraint": gamma_constraint,
        "synchronized": synchronized,
        "name": name,
        "dtype": dtype,
        "input": input_tensor,
        "training": training,
        "mask": mask
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    axis = 0
    momentum = 0.7
    epsilon = 0.0001
    center = False
    scale = False
    beta_initializer = 'he_uniform'
    gamma_initializer = 'glorot_normal'
    moving_mean_initializer = 'zeros'
    moving_variance_initializer = 'ones'
    beta_regularizer = None
    gamma_regularizer = None
    beta_constraint = None
    gamma_constraint = None
    synchronized = True
    name = 'batch_norm_6'
    dtype = tf.float64
    input_tensor = np.random.rand(10,).astype(np.float64)
    training = False
    mask = None

    input_dict = {
        "axis": axis,
        "momentum": momentum,
        "epsilon": epsilon,
        "center": center,
        "scale": scale,
        "beta_initializer": beta_initializer,
        "gamma_initializer": gamma_initializer,
        "moving_mean_initializer": moving_mean_initializer,
        "moving_variance_initializer": moving_variance_initializer,
        "beta_regularizer": beta_regularizer,
        "gamma_regularizer": gamma_regularizer,
        "beta_constraint": beta_constraint,
        "gamma_constraint": gamma_constraint,
        "synchronized": synchronized,
        "name": name,
        "dtype": dtype,
        "input": input_tensor,
        "training": training,
        "mask": mask
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    axis = 1
    momentum = 0.99
    epsilon = 0.001
    center = True
    scale = True
    beta_initializer = 'zeros'
    gamma_initializer = 'ones'
    moving_mean_initializer = 'zeros'
    moving_variance_initializer = 'ones'
    beta_regularizer = None
    gamma_regularizer = None
    beta_constraint = None
    gamma_constraint = None
    synchronized = False
    name = 'batch_norm_7'
    dtype = tf.float32
    input_tensor = np.random.rand(5, 10).astype(np.float32)
    training = True
    mask = None

    input_dict = {
        "axis": axis,
        "momentum": momentum,
        "epsilon": epsilon,
        "center": center,
        "scale": scale,
        "beta_initializer": beta_initializer,
        "gamma_initializer": gamma_initializer,
        "moving_mean_initializer": moving_mean_initializer,
        "moving_variance_initializer": moving_variance_initializer,
        "beta_regularizer": beta_regularizer,
        "gamma_regularizer": gamma_regularizer,
        "beta_constraint": beta_constraint,
        "gamma_constraint": gamma_constraint,
        "synchronized": synchronized,
        "name": name,
        "dtype": dtype,
        "input": input_tensor,
        "training": training,
        "mask": mask
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    axis = -1
    momentum = 0.95
    epsilon = 0.005
    center = False
    scale = True
    beta_initializer = 'glorot_uniform'
    gamma_initializer = 'he_normal'
    moving_mean_initializer = 'zeros'
    moving_variance_initializer = 'ones'
    beta_regularizer = None
    gamma_regularizer = None
    beta_constraint = None
    gamma_constraint = None
    synchronized = True
    name = 'batch_norm_8'
    dtype = tf.float64
    input_tensor = np.random.rand(2, 3, 4).astype(np.float64)
    training = False
    mask = np.random.randint(0, 2, size=(2, 3)).astype(bool)

    input_dict = {
        "axis": axis,
        "momentum": momentum,
        "epsilon": epsilon,
        "center": center,
        "scale": scale,
        "beta_initializer": beta_initializer,
        "gamma_initializer": gamma_initializer,
        "moving_mean_initializer": moving_mean_initializer,
        "moving_variance_initializer": moving_variance_initializer,
        "beta_regularizer": beta_regularizer,
        "gamma_regularizer": gamma_regularizer,
        "beta_constraint": beta_constraint,
        "gamma_constraint": gamma_constraint,
        "synchronized": synchronized,
        "name": name,
        "dtype": dtype,
        "input": input_tensor,
        "training": training,
        "mask": mask
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    axis = 3
    momentum = 0.9
    epsilon = 1e-06
    center = True
    scale = False
    beta_initializer = 'zeros'
    gamma_initializer = 'ones'
    moving_mean_initializer = 'zeros'
    moving_variance_initializer = 'ones'
    beta_regularizer = None
    gamma_regularizer = None
    beta_constraint = None
    gamma_constraint = None
    synchronized = False
    name = 'batch_norm_9'
    dtype = tf.float32
    input_tensor = np.random.rand(1, 5, 5, 10).astype(np.float32)
    training = True
    mask = None

    input_dict = {
        "axis": axis,
        "momentum": momentum,
        "epsilon": epsilon,
        "center": center,
        "scale": scale,
        "beta_initializer": beta_initializer,
        "gamma_initializer": gamma_initializer,
        "moving_mean_initializer": moving_mean_initializer,
        "moving_variance_initializer": moving_variance_initializer,
        "beta_regularizer": beta_regularizer,
        "gamma_regularizer": gamma_regularizer,
        "beta_constraint": beta_constraint,
        "gamma_constraint": gamma_constraint,
        "synchronized": synchronized,
        "name": name,
        "dtype": dtype,
        "input": input_tensor,
        "training": training,
        "mask": mask
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    axis = -1
    momentum = 0.999
    epsilon = 0.00001
    center = False
    scale = True
    beta_initializer = 'glorot_uniform'
    gamma_initializer = 'he_normal'
    moving_mean_initializer = 'zeros'
    moving_variance_initializer = 'ones'
    beta_regularizer = None
    gamma_regularizer = None
    beta_constraint = None
    gamma_constraint = None
    synchronized = True
    name = 'batch_norm_10'
    dtype = tf.float64
    input_tensor = np.random.rand(5, 2, 3, 4, 1).astype(np.float64)
    training = False
    mask = np.random.randint(0, 2, size=(5, 2, 3, 4)).astype(bool)

    input_dict = {
        "axis": axis,
        "momentum": momentum,
        "epsilon": epsilon,
        "center": center,
        "scale": scale,
        "beta_initializer": beta_initializer,
        "gamma_initializer": gamma_initializer,
        "moving_mean_initializer": moving_mean_initializer,
        "moving_variance_initializer": moving_variance_initializer,
        "beta_regularizer": beta_regularizer,
        "gamma_regularizer": gamma_regularizer,
        "beta_constraint": beta_constraint,
        "gamma_constraint": gamma_constraint,
        "synchronized": synchronized,
        "name": name,
        "dtype": dtype,
        "input": input_tensor,
        "training": training,
        "mask": mask
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.keras.layers.BatchNormalization"] = tf_keras_layers_batchnormalization_inputs()

