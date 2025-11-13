generated_inputs = {}
import tensorflow as tf
import numpy as np
import copy

def tf_convert_to_tensor_4_inputs():
    list_of_inputs = []
    
    input_dict = {
        "value": 3.14,
        "dtype": np.float32,
        "dtype_hint": np.float64,
        "name": "test_tensor_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": -2.718,
        "dtype": np.float32,
        "dtype_hint": np.float64,
        "name": "test_tensor_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": 0.0,
        "dtype": np.float64,
        "dtype_hint": np.float32,
        "name": "test_tensor_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": 1000000.0,
        "dtype": np.float32,
        "dtype_hint": np.float64,
        "name": "test_tensor_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": -0.0001,
        "dtype": np.float16,
        "dtype_hint": np.float32,
        "name": "test_tensor_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": 1.0,
        "dtype": np.float32,
        "dtype_hint": np.float16,
        "name": "test_tensor_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": 0.5,
        "dtype": np.float32,
        "dtype_hint": np.float16,
        "name": "test_tensor_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": -42.0,
        "dtype": np.float64,
        "dtype_hint": np.float32,
        "name": "test_tensor_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": 1e-10,
        "dtype": np.float64,
        "dtype_hint": np.float32,
        "name": "test_tensor_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": -999999.99,
        "dtype": np.float32,
        "dtype_hint": np.float64,
        "name": "test_tensor_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.convert_to_tensor_4"] = tf_convert_to_tensor_4_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_counter_inputs():
    list_of_inputs = []
    
    # Input 1: Default values
    input_dict = {
        "start": 0,
        "step": 1,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Custom start value
    input_dict = {
        "start": 10,
        "step": 1,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Custom step value
    input_dict = {
        "start": 0,
        "step": 5,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Negative step
    input_dict = {
        "start": 10,
        "step": -1,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: int32 dtype
    input_dict = {
        "start": 0,
        "step": 1,
        "dtype": tf.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Negative start
    input_dict = {
        "start": -5,
        "step": 1,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Large step
    input_dict = {
        "start": 0,
        "step": 100,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Negative start and step
    input_dict = {
        "start": -10,
        "step": -2,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Large start value
    input_dict = {
        "start": 1000,
        "step": 10,
        "dtype": tf.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Custom start, step, and int32 dtype
    input_dict = {
        "start": 2,
        "step": 5,
        "dtype": tf.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Zero start with negative step
    input_dict = {
        "start": 0,
        "step": -10,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Large negative start
    input_dict = {
        "start": -1000,
        "step": 50,
        "dtype": tf.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.data.experimental.Counter"] = tf_data_experimental_counter_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_eigh_inputs():
    list_of_inputs = []
    
    tensor = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": "eigh_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[[1.0, 0.5], [0.5, 2.0]], [[3.0, 1.0], [1.0, 4.0]]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.eye(3, dtype=np.float64)
    input_dict = {"tensor": tensor, "name": "identity_eigh"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[1.0, 2.0], [2.0, -1.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[5.0, 1.0, 0.0, 0.5], [1.0, 4.0, 0.5, 0.0], [0.0, 0.5, 3.0, 1.0], [0.5, 0.0, 1.0, 2.0]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": "large_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.random.randn(2, 2, 3, 3).astype(np.float32)
    tensor = (tensor + np.transpose(tensor, (0, 1, 3, 2))) / 2
    input_dict = {"tensor": tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.float64)
    input_dict = {"tensor": tensor, "name": "diagonal_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[0.01, 0.005], [0.005, 0.02]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor = np.array([[[5.0]], [[10.0]], [[15.0]]], dtype=np.float32)
    input_dict = {"tensor": tensor, "name": "batch_1x1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.linalg.eigh"] = tf_linalg_eigh_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_bessel_i1e_inputs():
    list_of_inputs = []
    
    x = np.array([1.0], dtype=np.float32)
    input_dict = {"x": x, "name": "bessel_i1e_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-1.0], dtype=np.float32)
    input_dict = {"x": x, "name": "bessel_i1e_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
    input_dict = {"x": x, "name": "bessel_i1e_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.0], dtype=np.float32)
    input_dict = {"x": x, "name": "bessel_i1e_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "bessel_i1e_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    input_dict = {"x": x, "name": "bessel_i1e_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    input_dict = {"x": x, "name": "bessel_i1e_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-10.0, -20.0, -30.0], dtype=np.float32)
    input_dict = {"x": x, "name": "bessel_i1e_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.001, -0.001, 0.01, -0.01], dtype=np.float64)
    input_dict = {"x": x, "name": "bessel_i1e_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.5, 2.5, 3.5], dtype=np.float16)
    input_dict = {"x": x, "name": "bessel_i1e_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.linspace(-5.0, 5.0, 20, dtype=np.float32)
    input_dict = {"x": x, "name": "bessel_i1e_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.bessel_i1e"] = tf_math_bessel_i1e_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_lgamma_inputs():
    list_of_inputs = []
    
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    input_dict = {"x": x, "name": "lgamma_positive_ints"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    input_dict = {"x": x, "name": "lgamma_half_ints"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.5, 1, 4.5, -4, -5.6], dtype=np.float32)
    input_dict = {"x": x, "name": "lgamma_mixed"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "lgamma_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"x": x, "name": "lgamma_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([10.0, 20.0, 50.0, 100.0], dtype=np.float32)
    input_dict = {"x": x, "name": "lgamma_large"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32)
    input_dict = {"x": x, "name": "lgamma_small"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0, 2.5, 5.0, 10.0], dtype=np.float64)
    input_dict = {"x": x, "name": "lgamma_float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    input_dict = {"x": x, "name": "lgamma_float16"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([5.0], dtype=np.float32)
    input_dict = {"x": x, "name": "lgamma_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-0.5, -1.5, -2.5, -3.5], dtype=np.float32)
    input_dict = {"x": x, "name": "lgamma_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.lgamma"] = tf_math_lgamma_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_log1p_inputs():
    list_of_inputs = []
    
    x = np.array([0, 0.5, 1, 5], dtype=np.float32)
    name = "log1p_op1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    name = "log1p_op2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(2.0, dtype=np.float32)
    name = "log1p_op3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    name = "log1p_op4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-0.5, -0.3, -0.1, 0.0], dtype=np.float32)
    name = "log1p_op5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([10.0, 100.0, 1000.0], dtype=np.float64)
    name = "log1p_op6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1e-10, 1e-8, 1e-6, 1e-4], dtype=np.float64)
    name = "log1p_op7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1+2j, 3+4j, 0+1j], dtype=np.complex64)
    name = "log1p_op8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    name = "log1p_op9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.random.rand(2, 2, 2, 2).astype(np.float32)
    name = "log1p_op10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.5, 1.5, 2.5], dtype=np.float16)
    name = "log1p_op11"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.log1p"] = tf_math_log1p_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_logical_not_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 1D array with True/False values
    x = np.array([True, False], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: All True values
    x = np.array([True, True, True], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: All False values
    x = np.array([False, False, False, False], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D array
    x = np.array([[True, False], [False, True]], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D array
    x = np.array([[[True, False], [True, True]], [[False, False], [True, False]]], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Single scalar value (0D)
    x = np.array(True, dtype=bool)
    input_dict = {"x": x, "name": "logical_not_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single scalar False
    x = np.array(False, dtype=bool)
    input_dict = {"x": x, "name": "logical_not_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Larger 1D array
    x = np.array([True, False, True, False, True, False, True, False], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D array with different shape
    x = np.array([[True, True, True], [False, False, False]], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D array
    x = np.array([[[[True, False], [False, True]]]], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Large 2D array
    x = np.array([[True, False, True, False], [False, True, False, True], [True, True, False, False]], dtype=bool)
    input_dict = {"x": x, "name": "logical_not_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Empty name parameter
    x = np.array([True, False], dtype=bool)
    input_dict = {"x": x, "name": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.logical_not"] = tf_math_logical_not_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_polygamma_inputs():
    list_of_inputs = []
    
    a = np.array(0.0, dtype=np.float32)
    x = np.array(1.0, dtype=np.float32)
    name = "polygamma_0"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([1.0], dtype=np.float32)
    x = np.array([2.5], dtype=np.float32)
    name = "polygamma_1"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    name = "polygamma_2"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float32)
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    name = "polygamma_3"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    x = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    name = "polygamma_large"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array(4.0, dtype=np.float32)
    x = np.array(0.5, dtype=np.float32)
    name = "polygamma_order4"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.ones((2, 2, 2), dtype=np.float32)
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    name = "polygamma_3d"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([5.0, 5.0], dtype=np.float64)
    x = np.array([2.0, 3.0], dtype=np.float64)
    name = "polygamma_order5"
    input_dict = {"a": a, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.polygamma"] = tf_math_polygamma_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_real_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([-2.25 + 4.75j, 3.25 + 5.75j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": "real_part"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    input_dict = {"input": input_tensor, "name": "real_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "already_real"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-5-3j, -2+1j, -7-9j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": "negative_real"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex128)
    input_dict = {"input": input_tensor, "name": "complex_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array(3+4j, dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": "scalar_complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-1.0, -2.5, -3.7], dtype=np.float64)
    input_dict = {"input": input_tensor, "name": "negative_floats"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([5+0j, 10+0j, 15+0j], dtype=np.complex64)
    input_dict = {"input": input_tensor, "name": "zero_imaginary"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": "integers"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0+1j, 0+2j, 0+3j], dtype=np.complex128)
    input_dict = {"input": input_tensor, "name": "zero_real"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": "large_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.real"] = tf_math_real_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_crelu_inputs():
    list_of_inputs = []
    
    features = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    axis = -1
    name = "crelu_1d"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[1.5, -2.5], [3.5, -4.5]], dtype=np.float64)
    axis = -1
    name = "crelu_2d"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.int32)
    axis = 0
    name = "crelu_3d_axis0"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[10, -20, 30], [-40, 50, -60]], dtype=np.int64)
    axis = 1
    name = "crelu_int64"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[1000, -2000], [-3000, 4000]], dtype=np.int16)
    axis = 0
    name = "crelu_int16"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[10, -20], [-30, 40]], dtype=np.int8)
    axis = 1
    name = "crelu_int8"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = -1
    name = "crelu_4d"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[[1.0, -1.0], [2.0, -2.0]], [[3.0, -3.0], [4.0, -4.0]]], dtype=np.float32)
    axis = 2
    name = "crelu_3d_axis2"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[5.0, -10.0, 15.0]], dtype=np.float32)
    axis = -1
    name = "crelu_single_row"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float32)
    axis = 0
    name = "crelu_zeros"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.crelu"] = tf_nn_crelu_inputs()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_fact_inputs():
    list_of_inputs = []
    
    # Input 1: No name parameter
    input_dict = {
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Simple name
    input_dict = {
        "name": "fact_op"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Name with underscores
    input_dict = {
        "name": "factorial_fact_operation"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Name with numbers
    input_dict = {
        "name": "fact123"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Short name
    input_dict = {
        "name": "f"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Longer descriptive name
    input_dict = {
        "name": "factorial_fact_output_operation"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Name with forward slash
    input_dict = {
        "name": "my_scope/fact_op"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Name with multiple slashes
    input_dict = {
        "name": "scope1/scope2/fact"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Name with mixed case
    input_dict = {
        "name": "MyFactOperation"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Name with prefix
    input_dict = {
        "name": "op_fact_tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Empty string name
    input_dict = {
        "name": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Fact"] = tf_raw_ops_fact_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_log_inputs():
    list_of_inputs = []
    
    x = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)
    input_dict = {"x": x, "name": "log_op_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "log_op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"x": x, "name": "log_op_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(2.718281828, dtype=np.float32)
    input_dict = {"x": x, "name": "log_op_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0, 10.0, 100.0], dtype=np.float32)
    input_dict = {"x": x, "name": "log_op_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.1, 0.5, 1.5, 10.0], dtype=np.float16)
    input_dict = {"x": x, "name": "log_op_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict = {"x": x, "name": "log_op_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1+0j, 2+0j, 3+0j], dtype=np.complex128)
    input_dict = {"x": x, "name": "log_op_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.linspace(0.1, 100.0, 50, dtype=np.float32)
    input_dict = {"x": x, "name": "log_op_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.ones((2, 2, 2, 2), dtype=np.float64)
    input_dict = {"x": x, "name": "log_op_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.001, 0.01, 0.1, 1.0], dtype=np.float32)
    input_dict = {"x": x, "name": "log_op_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Log"] = tf_raw_ops_log_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_round_inputs():
    list_of_inputs = []
    
    x = np.array([1.2, 2.5, 3.7, 4.1], dtype=np.float32)
    input_dict = {"x": x, "name": "round_op_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[-1.5, -2.3], [-3.7, -4.9]], dtype=np.float64)
    input_dict = {"x": x, "name": "round_op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]], dtype=np.float32)
    input_dict = {"x": x, "name": "round_op_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.1, 0.9, 1.1, 1.9], dtype=np.float32)
    input_dict = {"x": x, "name": "round_op_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(2.5, dtype=np.float32)
    input_dict = {"x": x, "name": "round_op_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[0.0, 0.5], [-0.5, 0.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "round_op_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {"x": x, "name": "round_op_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[10, 20], [30, 40]], dtype=np.int64)
    input_dict = {"x": x, "name": "round_op_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[[1.1, 1.9], [2.1, 2.9]], [[3.1, 3.9], [4.1, 4.9]]]], dtype=np.float32)
    input_dict = {"x": x, "name": "round_op_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([100.5, 200.5, 300.5, 400.5], dtype=np.float64)
    input_dict = {"x": x, "name": "round_op_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.001, 0.0001, -0.001, -0.0001], dtype=np.float32)
    input_dict = {"x": x, "name": "round_op_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Round"] = tf_raw_ops_round_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_softplus_inputs():
    list_of_inputs = []
    
    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "softplus_1"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    name = "softplus_2"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    name = "softplus_3"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    name = "softplus_4"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([-5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    name = "softplus_5"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    name = "softplus_6"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    name = "softplus_7"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array(5.0, dtype=np.float32)
    name = "softplus_8"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    name = "softplus_9"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float64)
    name = "softplus_10"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = "softplus_11"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Softplus"] = tf_raw_ops_softplus_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Tan_inputs():
    list_of_inputs = []
    
    x = np.array([-9.0, -0.5, 0.0, 1.0, 1.2], dtype=np.float32)
    input_dict = {"name": "tan_op_1", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    input_dict = {"name": "tan_op_2", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    input_dict = {"name": "tan_op_3", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(0.785398, dtype=np.float32)
    input_dict = {"name": "tan_op_4", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-1.5, -2.0, -3.0, -4.0], dtype=np.float32)
    input_dict = {"name": "tan_op_5", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    input_dict = {"name": "tan_op_6", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    input_dict = {"name": "tan_op_7", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0 + 2.0j, 0.5 + 0.5j, -1.0 + 1.0j], dtype=np.complex64)
    input_dict = {"name": "tan_op_8", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [0.0 + 1.0j, 1.0 + 1.0j]], dtype=np.complex128)
    input_dict = {"name": "tan_op_9", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-3.14, -1.57, 0.0, 1.57, 3.14, 4.71], dtype=np.float64)
    input_dict = {"name": "tan_op_10", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    input_dict = {"name": "tan_op_11", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Tan"] = tf_raw_ops_Tan_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_UnicodeScript_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([65, 66, 67], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1, 31, 38], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[65, 945, 1040], [20013, 12354, 1488]], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([945], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-1, -100, 65], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0x1F600, 0x1F601, 0x1F602], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[65, 66], [67, 68]], [[945, 946], [947, 948]]], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0, 32, 127], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([20013, 22269, 26085, 26412], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1575, 1576, 1577, 1578], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1488, 1489, 1490], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.UnicodeScript"] = tf_raw_ops_UnicodeScript_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_where_inputs():
    list_of_inputs = []
    
    condition = np.array([[True, False], [True, False]], dtype=np.bool_)
    input_dict = {"condition": condition, "name": "where_op_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    condition = np.array([[[True, False], [True, False]], [[False, True], [False, True]], [[False, False], [False, True]]], dtype=np.bool_)
    input_dict = {"condition": condition, "name": "where_op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    condition = np.array([True, False, True, True, False], dtype=np.bool_)
    input_dict = {"condition": condition, "name": "where_op_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    condition = np.array([[1.5, 0.0], [-0.5, 0.0]], dtype=np.float32)
    input_dict = {"condition": condition, "name": "where_op_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    condition = np.array([[[1.5, 0.0], [-0.5, 0.0]], [[0.0, 0.25], [0.0, 0.75]], [[0.0, 0.0], [0.0, 0.01]]], dtype=np.float64)
    input_dict = {"condition": condition, "name": "where_op_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    condition = np.array([[1, 0, -3], [0, 5, 0], [7, 0, 9]], dtype=np.int32)
    input_dict = {"condition": condition, "name": "where_op_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    condition = np.array([10, 0, -20, 30, 0], dtype=np.int64)
    input_dict = {"condition": condition, "name": "where_op_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    condition = np.array([[255, 0], [128, 1]], dtype=np.uint8)
    input_dict = {"condition": condition, "name": "where_op_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    condition = np.array([[1.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.5j, 0.0 + 0.0j]], dtype=np.complex64)
    input_dict = {"condition": condition, "name": "where_op_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    condition = np.array([[[1.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.5j, 0.0 + 0.0j]], [[0.0 + 0.0j, 0.25 + 1.5j], [0.0 + 0.0j, 0.75 + 0.0j]]], dtype=np.complex128)
    input_dict = {"condition": condition, "name": "where_op_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Where"] = tf_raw_ops_where_inputs()

import tensorflow as tf
import copy

def tf_raw_ops_WriteFile_inputs():
    list_of_inputs = []
    
    input_dict = {
        "filename": "/tmp/test1.txt",
        "contents": "Hello World",
        "name": "write_op1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test2.txt",
        "contents": "",
        "name": "write_op2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test3.txt",
        "contents": "Line 1\nLine 2\nLine 3",
        "name": "write_op3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test4.bin",
        "contents": "Binary content",
        "name": "write_op4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/nested/dir/test5.txt",
        "contents": "Nested directory content",
        "name": "write_op5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test6.txt",
        "contents": "Special chars: !@#$%^&*()",
        "name": "write_op6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test7.txt",
        "contents": "Unicode content",
        "name": "write_op7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test8.json",
        "contents": '{"key": "value", "number": 42}',
        "name": "write_op8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test9.txt",
        "contents": "A" * 1000,
        "name": "write_op9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test10.txt",
        "contents": "Tab\there\tand\nspaces   here",
        "name": "write_op10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test11.txt",
        "contents": "No operation name",
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "filename": "/tmp/test12.csv",
        "contents": "col1,col2,col3\n1,2,3\n4,5,6",
        "name": "write_op12"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.WriteFile"] = tf_raw_ops_WriteFile_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_fresnel_cos_inputs():
    list_of_inputs = []
    
    x = np.array(1.0, dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(-1.0, dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-1.0, -0.1, 0.1, 1.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(0.0, dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(10.0, dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(2.5, dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.001, 0.01, 0.1], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[-3.0, -2.5], [-2.0, -1.5]], dtype=np.float64)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0], dtype=np.float32)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.special.fresnel_cos"] = tf_math_special_fresnel_cos_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_softsign_inputs():
    list_of_inputs = []
    
    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_op1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_op2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_op3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict = {"features": features, "name": "softsign_op4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_op5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([0.001, -0.001, 0.0001], dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_op6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([100.0, -100.0, 1000.0], dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_op7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {"features": features, "name": "softsign_op8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array(5.0, dtype=np.float32)
    input_dict = {"features": features, "name": "softsign_op9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[1.5, -2.5, 3.5], [4.5, -5.5, 6.5]], dtype=np.float16)
    input_dict = {"features": features, "name": "softsign_op10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.random.randn(100, 50).astype(np.float32)
    input_dict = {"features": features, "name": "softsign_op11"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.softsign"] = tf_nn_softsign_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_erf_inputs():
    list_of_inputs = []
    
    x = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, -2.0]], dtype=np.float32)
    name = "erf_op_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.5, -0.5, 1.5, -1.5, 2.5], dtype=np.float64)
    name = "erf_op_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array(1.0, dtype=np.float32)
    name = "erf_op_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[0.0, 0.5], [1.0, 1.5]], [[2.0, 2.5], [3.0, 3.5]]], dtype=np.float32)
    name = "erf_op_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-1.0, -2.0, -3.0, -0.5], dtype=np.float32)
    name = "erf_op_5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    name = "erf_op_6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([5.0, 10.0, -5.0, -10.0], dtype=np.float32)
    name = "erf_op_7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.001, -0.001, 0.01, -0.01], dtype=np.float64)
    name = "erf_op_8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]], dtype=np.float32)
    name = "erf_op_9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    name = "erf_op_10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Erf"] = tf_raw_ops_erf_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_RandomUniform_inputs():
    list_of_inputs = []
    
    input_dict = {
        "shape": np.array([10], dtype=np.int32),
        "dtype": tf.float32,
        "seed": 42,
        "seed2": 1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([5, 5], dtype=np.int32),
        "dtype": tf.float32,
        "seed": 42,
        "seed2": 123,
        "name": "random_uniform_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([2, 3, 4], dtype=np.int64),
        "dtype": tf.float64,
        "seed": 1,
        "seed2": 2,
        "name": "random_3d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([2, 2, 2, 2], dtype=np.int32),
        "dtype": tf.half,
        "seed": 100,
        "seed2": 200,
        "name": "random_half"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([100, 50], dtype=np.int64),
        "dtype": tf.bfloat16,
        "seed": 999,
        "seed2": 888,
        "name": "random_bfloat16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([1], dtype=np.int32),
        "dtype": tf.float32,
        "seed": 5,
        "seed2": 10,
        "name": "single_element"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([2, 3, 4, 5, 6], dtype=np.int32),
        "dtype": tf.float32,
        "seed": 7,
        "seed2": 14,
        "name": "random_5d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([8, 8], dtype=np.int64),
        "dtype": tf.float64,
        "seed": 50,
        "seed2": 60,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([1000], dtype=np.int32),
        "dtype": tf.float32,
        "seed": 12345,
        "seed2": 67890,
        "name": "large_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([4, 5, 6], dtype=np.int32),
        "dtype": tf.float64,
        "seed": 33,
        "seed2": 44,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.RandomUniform"] = tf_raw_ops_RandomUniform_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_eye_inputs():
    list_of_inputs = []
    
    input_dict = {
        "num_rows": 5,
        "num_columns": 5,
        "dtype": np.float32,
        "name": "eye1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 10,
        "num_columns": 5,
        "dtype": np.float32,
        "name": "eye2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 3,
        "num_columns": 8,
        "dtype": np.float64,
        "name": "eye3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 1,
        "num_columns": 1,
        "dtype": np.float32,
        "name": "eye4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 100,
        "num_columns": 100,
        "dtype": np.float32,
        "name": "eye5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 7,
        "num_columns": 7,
        "dtype": np.int32,
        "name": "eye6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 4,
        "num_columns": 6,
        "dtype": np.int64,
        "name": "eye7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 2,
        "num_columns": 20,
        "dtype": np.float32,
        "name": "eye8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 15,
        "num_columns": 15,
        "dtype": np.float16,
        "name": "eye9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 25,
        "num_columns": 10,
        "dtype": np.float64,
        "name": "eye10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 50,
        "num_columns": 3,
        "dtype": np.float32,
        "name": "eye11"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "num_rows": 10,
        "num_columns": 1,
        "dtype": np.float32,
        "name": "eye12"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.sparse.eye"] = tf_sparse_eye_inputs()

