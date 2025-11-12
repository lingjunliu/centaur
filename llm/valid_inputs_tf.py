generated_inputs = {}
import numpy as np
import tensorflow as tf
import copy

def tf_compat_forward_compatible_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_dict = {
        "year": np.int32(2023),
        "month": np.int32(10),
        "day": np.int32(15)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input_dict = {
        "year": np.int32(2024),
        "month": np.int32(1),
        "day": np.int32(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input_dict = {
        "year": np.int32(2023),
        "month": np.int32(12),
        "day": np.int32(31)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input_dict = {
        "year": np.int32(2025),
        "month": np.int32(6),
        "day": np.int32(15)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input_dict = {
        "year": np.int32(2023),
        "month": np.int32(2),
        "day": np.int32(29)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - negative values
    input_dict = {
        "year": np.int32(-1),
        "month": np.int32(12),
        "day": np.int32(31)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - zero values
    input_dict = {
        "year": np.int32(0),
        "month": np.int32(1),
        "day": np.int32(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - different dimensions
    input_dict = {
        "year": np.int32(2024),
        "month": np.int32(12),
        "day": np.int32(31)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - large values
    input_dict = {
        "year": np.int32(2050),
        "month": np.int32(12),
        "day": np.int32(31)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - small values
    input_dict = {
        "year": np.int32(2023),
        "month": np.int32(1),
        "day": np.int32(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.compat.forward_compatible"] = tf_compat_forward_compatible_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_counter_inputs():
    list_of_inputs = []
    
    # Input 1: start=0, step=1, dtype=tf.int64
    input_dict = {
        "start": np.int64(0),
        "step": np.int64(1),
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: start=2, step=1, dtype=tf.int32
    input_dict = {
        "start": np.int32(2),
        "step": np.int32(1),
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: start=10, step=-1, dtype=tf.int64
    input_dict = {
        "start": np.int64(10),
        "step": np.int64(-1),
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: start=5, step=2, dtype=tf.int32
    input_dict = {
        "start": np.int32(5),
        "step": np.int32(2),
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: start=0, step=3, dtype=tf.int64
    input_dict = {
        "start": np.int64(0),
        "step": np.int64(3),
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: start=-5, step=1, dtype=tf.int64
    input_dict = {
        "start": np.int64(-5),
        "step": np.int64(1),
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: start=0, step=-2, dtype=tf.int32
    input_dict = {
        "start": np.int32(0),
        "step": np.int32(-2),
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: start=100, step=5, dtype=tf.int64
    input_dict = {
        "start": np.int64(100),
        "step": np.int64(5),
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: start=2, step=7, dtype=tf.int32
    input_dict = {
        "start": np.int32(2),
        "step": np.int32(7),
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: start=-10, step=-3, dtype=tf.int64
    input_dict = {
        "start": np.int64(-10),
        "step": np.int64(-3),
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.data.experimental.Counter"] = tf_data_experimental_counter_inputs()

import tensorflow as tf
import numpy as np
import copy

def tfexperimental_numpy_append_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    arr = np.array([1, 2, 3])
    values = np.array([4, 5, 6])
    axis = None
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    arr = np.array([[1, 2], [3, 4]])
    values = np.array([[5, 6], [7, 8]])
    axis = 0
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid - single element arrays
    arr = np.array([1])
    values = np.array([2])
    axis = None
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid - scalar values
    arr = np.array(1)
    values = np.array(2)
    axis = None
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid - float arrays
    arr = np.array([1.1, 2.2, 3.3])
    values = np.array([4.4, 5.5, 6.6])
    axis = None
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - negative values
    arr = np.array([-1, -2, -3])
    values = np.array([-4, -5, -6])
    axis = None
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - same dimension arrays
    arr = np.array([[1, 2], [3, 4]])
    values = np.array([[5, 6], [7, 8]])
    axis = 1
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - one dimensional arrays
    arr = np.array([1, 2, 3])
    values = np.array([4, 5, 6])
    axis = 0
    
    input_dict = {
        "arr": arr,
        "values": values,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.append"] = tfexperimental_numpy_append_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_experiment_numpy_argmin_inputs():
    list_of_inputs = []
    
    # Input 1: 2D array with axis=0
    a = np.array([[1, 2, 3], [4, 5, 6]])
    input_dict = {"a": a, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D array with axis=1
    a = np.array([[1, 2, 3], [4, 5, 6]])
    input_dict = {"a": a, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D array
    a = np.array([1, 2, 3, 4])
    input_dict = {"a": a, "axis": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D array with axis=0
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"a": a, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D array with axis=1
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"a": a, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 3D array with axis=2
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"a": a, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 1D array with negative values
    a = np.array([-1, 2, -3, 4])
    input_dict = {"a": a, "axis": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 2D array with negative values
    a = np.array([[-1, 2], [3, -4]])
    input_dict = {"a": a, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D array with negative values and axis=1
    a = np.array([[-1, 2], [3, -4]])
    input_dict = {"a": a, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 1D array with float values
    a = np.array([1.5, 2.7, 3.1])
    input_dict = {"a": a, "axis": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.argmin"] = tf_experiment_numpy_argmin_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_bitwise_not_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([-1, -2, -3], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([0, 1, 0], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([1, 0, 1], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([-1, -2, -3], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([0, 0, 0], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([[0, 1], [1, 0]], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([[-1, 0, 1], [0, -1, 1]], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.bitwise_not"] = tf_bitwise_not_inputs()

import numpy as np
import tensorflow as tf

def conj_inputs():
    list_of_inputs = []
    
    # Input 1: Real tensor
    x = np.array([1, 2, 3, 4])
    input_dict = {"x": x}
    list_of_inputs.append(input_dict)
    
    # Input 2: Complex tensor
    x = np.array([1+2j, 3+4j, 5+6j])
    input_dict = {"x": x}
    list_of_inputs.append(input_dict)
    
    # Input 3: Real tensor with negative values
    x = np.array([-1, -2, -3, -4])
    input_dict = {"x": x}
    list_of_inputs.append(input_dict)
    
    # Input 4: Complex tensor with negative real part
    x = np.array([-1+2j, -3+4j, -5+6j])
    input_dict = {"x": x}
    list_of_inputs.append(input_dict)
    
    # Input 5: 2D tensor
    x = np.array([[1, 2], [3, 4]])
    input_dict = {"x": x}
    list_of_inputs.append(input_dict)
    
    # Input 6: 2D complex tensor
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
    input_dict = {"x": x}
    list_of_inputs.append(input_dict)
    
    # Input 7: 3D tensor
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"x": x}
    list_of_inputs.append(input_dict)
    
    # Input 8: 3D complex tensor
    x = np.array([[[1+2j, 3+4j], [5+6j, 7+8j]], [[9+10j, 11+12j], [13+14j, 15+16j]]])
    input_dict = {"x": x}
    list_of_inputs.append(input_dict)
    
    # Input 9: Scalar tensor
    x = np.array(1)
    input_dict = {"x": x}
    list_of_inputs.append(input_dict)
    
    # Input 10: Scalar complex tensor
    x = np.array(1+2j)
    input_dict = {"x": x}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.experimental.numpy.conj"] = conj_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_diag_inputs():
    list_of_inputs = []
    
    # Input 1: 1D tensor with k=0
    v = np.array([1, 2, 3, 4])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 1D tensor with k=1
    v = np.array([1, 2, 3, 4])
    k = 1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D tensor with k=-1
    v = np.array([1, 2, 3, 4])
    k = -1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D tensor with k=0
    v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor with k=1
    v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensor with k=-1
    v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = -1
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 1D tensor with negative values
    v = np.array([-1, 2, -3, 4])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 2D tensor with negative values
    v = np.array([[-1, 2], [3, -4]])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D tensor with float values
    v = np.array([[1.5, 2.7], [3.1, 4.8]])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D tensor with complex values
    v = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
    k = 0
    input_dict = {"v": v, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.diag"] = tf_diag_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_fix_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([1.2, -2.7, 3.9])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([[1.2, -2.7], [3.9, -4.1]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([[[1.2, -2.7], [3.9, -4.1]], [[5.6, -6.8], [7.9, -8.2]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([1.0, -2.0, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([-1.5, -2.5, -3.5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([0.0, 0.0, 0.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([1.123, -2.456, 3.789])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([[1.1, -2.2], [3.3, -4.4]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([[-1.1, -2.2], [3.3, -4.4]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([-1.5, 2.7, -3.9])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.fix"] = tf_experimental_numpy_fix_inputs()

import tensorflow as tf
import copy

def tf_floor_divide_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x1 = tf.constant([[10, 20, 30], [40, 50, 60]])
    x2 = tf.constant([[2, 4, -6], [5, 7, 9]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x1 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    x2 = tf.constant([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x1 = tf.constant([1, 2, 3, 4, 5])
    x2 = tf.constant([2, 2, 2, 2, 2])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x1 = tf.constant([-10, -20, -30])
    x2 = tf.constant([2, 4, -6])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x1 = tf.constant([[1.5, 2.7], [3.9, 4.1]])
    x2 = tf.constant([[0.5, 1.3], [2.1, 1.8]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x1 = tf.constant([[[[1, 2]], [[3, 4]]]])
    x2 = tf.constant([[[[2, 3]], [[4, 5]]]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x1 = tf.constant([[10, -20], [30, -40]])
    x2 = tf.constant([[2, -4], [-6, 8]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x1 = tf.constant([-10, 20, -30])
    x2 = tf.constant([2, -4, 6])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x1 = tf.constant([[1.5, 2.7], [3.9, 4.1]])
    x2 = tf.constant([[2.5, 1.3], [2.1, 1.8]])
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x1 = tf.constant([[[[1, 2]], [[3, 4]]]], dtype=tf.float32)
    x2 = tf.constant([[[[2, 3]], [[4, 5]]]], dtype=tf.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.floor_divide"] = tf_floor_divide_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_isfinite_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([1.0, 2.0, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([-1.0, -2.0, -3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([np.inf, np.nan, 1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([[-1.0, 2.0], [3.0, -4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([[np.inf, np.nan], [1.0, 2.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([[[np.inf, np.nan], [1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([[[[-1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([[[-1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.isfinite"] = tf_isfinite_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_isinf_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([1.0, 2.0, np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([-np.inf, 0.0, 1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([np.inf, np.nan, 1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([np.inf, np.inf, np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([-np.inf, -np.inf, -np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([1.0, 2.0, np.nan])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([1.0, np.inf, 2.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([[1.0, np.inf], [2.0, np.nan]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([[-np.inf, 1.0], [np.nan, 2.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.isinf"] = tf_experimental_numpy_isinf_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_isnan_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([1.0, 2.0, np.nan, 4.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([[1.0, np.nan], [3.0, 4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([np.nan, np.nan, np.nan])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([[-1.0, 2.0, np.nan], [3.0, -4.0, 5.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [7.0, np.nan]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([np.nan, 1.0, np.nan, 2.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([[[np.nan, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, np.nan]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.isnan"] = tf_experimental_numpy_isnan_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_isposinf_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([1.0, 2.0, np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([-1.0, 2.0, np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([np.inf, np.inf, np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([-np.inf, np.inf, -1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([1.0, 2.0, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([[1.0, 2.0], [3.0, np.inf]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, np.inf]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([np.inf, -1.0, 2.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([1.0, 2.0, np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([-np.inf, -1.0, np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.isposinf"] = tf_experimental_numpy_isposinf_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_moveaxis_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    a = np.array([[1, 2, 3], [4, 5, 6]])
    source = 0
    destination = 1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = -1
    destination = 0
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = 0
    destination = -1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = -2
    destination = -1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = -1
    destination = 0
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = 1
    destination = -1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = 0
    destination = -2
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = -1
    destination = 1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = 0
    destination = -1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    source = -2
    destination = 0
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.moveaxis"] = tf_moveaxis_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_signbit_inputs():
    list_of_inputs = []
    
    # Input 1, valid - scalar tensor
    x = np.array(5.0)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid - 1D array
    x = np.array([1.0, -2.0, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid - 2D array
    x = np.array([[1.0, -2.0], [3.0, -4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid - 3D array
    x = np.array([[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid - negative values
    x = np.array([-1.0, -2.0, -3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - float values with zero
    x = np.array([0.0, -0.0, 1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - mixed positive/negative values
    x = np.array([1.0, -2.0, 3.0, -4.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - single element array
    x = np.array([-1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - large values
    x = np.array([1e10, -1e10, 1e-10])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - small values
    x = np.array([1e-10, -1e-10, 1e-5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.signbit"] = tf_signbit_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_central_crop_inputs():
    list_of_inputs = []
    
    # Input 1: 3D tensor with central_fraction=0.5
    image_3d = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], 
                         [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]], dtype=np.float32)
    central_fraction = 0.5
    input_dict = {
        "image": image_3d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor with central_fraction=0.25
    image_4d = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]],
                         [[[13., 14., 15.], [16., 17., 18.]], [[19., 20., 21.], [22., 23., 24.]]]], dtype=np.float32)
    central_fraction = 0.25
    input_dict = {
        "image": image_4d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor with central_fraction=0.75
    image_3d = np.array([[[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.]], 
                         [[16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.], [26., 27., 28., 29., 30.]]], dtype=np.float32)
    central_fraction = 0.75
    input_dict = {
        "image": image_3d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor with central_fraction=0.1
    image_4d = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]],
                         [[[13., 14., 15.], [16., 17., 18.]], [[19., 20., 21.], [22., 23., 24.]]],
                         [[[25., 26., 27.], [28., 29., 30.]], [[31., 32., 33.], [34., 35., 36.]]]], dtype=np.float32)
    central_fraction = 0.1
    input_dict = {
        "image": image_4d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D tensor with central_fraction=1.0
    image_3d = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], 
                         [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]], dtype=np.float32)
    central_fraction = 1.0
    input_dict = {
        "image": image_3d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with central_fraction=0.9
    image_4d = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]],
                         [[[13., 14., 15.], [16., 17., 18.]], [[19., 20., 21.], [22., 23., 24.]]],
                         [[[25., 26., 27.], [28., 29., 30.]], [[31., 32., 33.], [34., 35., 36.]]]], dtype=np.float32)
    central_fraction = 0.9
    input_dict = {
        "image": image_4d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D tensor with central_fraction=0.33
    image_3d = np.array([[[1., 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12.], [13., 14., 15., 16., 17., 18.]], 
                         [[19., 20., 21., 22., 23., 24.], [25., 26., 27., 28., 29., 30.], [31., 32., 33., 34., 35., 36.]]], dtype=np.float32)
    central_fraction = 0.33
    input_dict = {
        "image": image_3d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor with central_fraction=0.4
    image_4d = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]],
                         [[[13., 14., 15.], [16., 17., 18.]], [[19., 20., 21.], [22., 23., 24.]]],
                         [[[25., 26., 27.], [28., 29., 30.]], [[31., 32., 33.], [34., 35., 36.]]]], dtype=np.float32)
    central_fraction = 0.4
    input_dict = {
        "image": image_4d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with central_fraction=0.6
    image_3d = np.array([[[1., 2., 3., 4., 5., 6., 7., 8., 9.], [10., 11., 12., 13., 14., 15., 16., 17., 18.]], 
                         [[19., 20., 21., 22., 23., 24., 25., 26., 27.], [28., 29., 30., 31., 32., 33., 34., 35., 36.]]], dtype=np.float32)
    central_fraction = 0.6
    input_dict = {
        "image": image_3d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with central_fraction=0.01
    image_4d = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]],
                         [[[13., 14., 15.], [16., 17., 18.]], [[19., 20., 21.], [22., 23., 24.]]],
                         [[[25., 26., 27.], [28., 29., 30.]], [[31., 32., 33.], [34., 35., 36.]]]], dtype=np.float32)
    central_fraction = 0.01
    input_dict = {
        "image": image_4d,
        "central_fraction": central_fraction
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.central_crop"] = tf_image_central_crop_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_grayscale_to_rgb_inputs():
    list_of_inputs = []
    
    # Input 1: Single channel image (3D)
    a = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Single channel image with negative values (3D)
    a = np.array([[[ -1.0], [0.0], [1.0]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Multi-channel image (4D)
    a = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Multi-channel image with negative values (4D)
    a = np.array([[[[ -1.0], [0.0]], [[1.0], [2.0]]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Single channel image with float values (3D)
    a = np.array([[[1.5], [2.7], [3.9]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Single channel image with mixed values (3D)
    a = np.array([[[0.0], [1.0], [2.0]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single channel image with zero values (3D)
    a = np.array([[[0.0], [0.0], [0.0]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Single channel image with large values (3D)
    a = np.array([[[100.0], [200.0], [300.0]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Single channel image with decimal values (3D)
    a = np.array([[[1.1], [2.2], [3.3]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Single channel image with negative values (3D)
    a = np.array([[[ -5.0], [ -10.0], [ -15.0]]], dtype=np.float32)
    input_dict = {
        "images": a,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.grayscale_to_rgb"] = generate_grayscale_to_rgb_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_random_hue_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    max_delta = 0.2
    seed = 42
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    image = np.array([[[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]], [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]]], dtype=np.float32)
    max_delta = 0.1
    seed = 100
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    image = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=np.float32)
    max_delta = 0.3
    seed = 200
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]], dtype=np.float32)
    max_delta = 0.4
    seed = 300
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    max_delta = 0.05
    seed = 400
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    max_delta = 0.5
    seed = 500
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    image = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=np.float32)
    max_delta = 0.0
    seed = 600
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    max_delta = 0.3
    seed = 700
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    max_delta = 0.05
    seed = 800
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    max_delta = 0.2
    seed = 900
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.random_hue"] = tf_image_random_hue_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_rgb_to_hsv_inputs():
    list_of_inputs = []
    
    # Input 1: 3D tensor with RGB values in [0,1]
    a = tf.constant([[[0.5, 0.5, 0.5], [0.8, 0.8, 0.8]], [[0.2, 0.2, 0.2], [0.9, 0.9, 0.9]]], dtype=tf.float32)
    input_dict = {
        "images": a,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor with RGB values in [0,1]
    b = tf.random.uniform([2, 3, 4, 3], minval=0, maxval=1, dtype=tf.float32)
    input_dict = {
        "images": b,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D tensor with RGB values in [0,1]
    c = tf.constant([[0.1, 0.2, 0.3]], dtype=tf.float32)
    input_dict = {
        "images": c,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D tensor with RGB values in [0,1]
    d = tf.constant([[0.5, 0.5, 0.5], [0.8, 0.8, 0.8]], dtype=tf.float32)
    input_dict = {
        "images": d,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D tensor with RGB values in [0,1] and different dtypes
    f = tf.constant([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]], dtype=tf.float64)
    input_dict = {
        "images": f,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with RGB values in [0,1] and different dtypes
    g = tf.random.uniform([3, 2, 5, 3], minval=0, maxval=1, dtype=tf.half)
    input_dict = {
        "images": g,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 4D tensor with RGB values in [0,1] and different dtypes
    h = tf.random.uniform([2, 3, 4, 3], minval=0, maxval=1, dtype=tf.float32)
    input_dict = {
        "images": h,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 3D tensor with RGB values in [0,1] and different dtypes
    i = tf.constant([[[0.5, 0.5, 0.5], [0.8, 0.8, 0.8]], [[0.2, 0.2, 0.2], [0.9, 0.9, 0.9]]], dtype=tf.float64)
    input_dict = {
        "images": i,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D tensor with RGB values in [0,1] and different dtypes
    j = tf.random.uniform([2, 3, 4, 3], minval=0, maxval=1, dtype=tf.float32)
    input_dict = {
        "images": j,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 3D tensor with RGB values in [0,1] and different dtypes
    k = tf.constant([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]], dtype=tf.float64)
    input_dict = {
        "images": k,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.rgb_to_hsv"] = tf_image_rgb_to_hsv_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_transpose_inputs():
    list_of_inputs = []
    
    # Input 1: 3D tensor with shape (2, 3, 2)
    image = np.array([[[1, 2], [5, 6], [9, 10]], [[3, 4], [7, 8], [11, 12]]], dtype=np.int32)
    input_dict = {
        "image": image,
        "name": "transpose_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor with shape (1, 2, 3, 2) 
    image = np.array([[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]], dtype=np.int32)
    input_dict = {
        "image": image,
        "name": "transpose_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor with shape (3, 2, 2)
    image = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=np.int32)
    input_dict = {
        "image": image,
        "name": "transpose_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor with shape (2, 3, 2, 2)
    image = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.int32)
    input_dict = {
        "image": image,
        "name": "transpose_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D tensor with shape (2, 2, 3) 
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    input_dict = {
        "image": image,
        "name": "transpose_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with shape (1, 3, 2, 3)
    image = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.float32)
    input_dict = {
        "image": image,
        "name": "transpose_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D tensor with shape (4, 2, 2)
    image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32)
    input_dict = {
        "image": image,
        "name": "transpose_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor with shape (2, 2, 2, 2)
    image = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.int32)
    input_dict = {
        "image": image,
        "name": "transpose_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with shape (3, 3, 3)
    image = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]], dtype=np.float32)
    input_dict = {
        "image": image,
        "name": "transpose_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with shape (1, 2, 2, 2)
    image = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    input_dict = {
        "image": image,
        "name": "transpose_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.transpose"] = tf_image_transpose_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_linear_operator_householder_inputs():
    list_of_inputs = []
    
    # Input 1, valid - 1D vector
    reflection_axis = np.array([1.0, 0.0], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "test1"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid - 1D vector with negative values
    reflection_axis = np.array([-1.0, 0.0], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "test2"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid - 2D vector
    reflection_axis = np.array([1.0, 1.0], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "test3"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid - 2D vector with negative values
    reflection_axis = np.array([-1.0, -1.0], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "test4"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid - 3D vector
    reflection_axis = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "test5"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - 3D vector with negative values
    reflection_axis = np.array([-1.0, 0.0, -1.0], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "test6"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - 4D vector
    reflection_axis = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "test7"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - 4D vector with negative values
    reflection_axis = np.array([-1.0, -1.0, 0.0, -1.0], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "test8"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - 5D vector
    reflection_axis = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "test9"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - 5D vector with negative values
    reflection_axis = np.array([-1.0, 0.0, -1.0, 0.0, -1.0], dtype=np.float32)
    is_non_singular = True
    is_self_adjoint = True
    is_positive_definite = False
    is_square = True
    name = "test10"
    
    input_dict = {
        "reflection_axis": reflection_axis,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorHouseholder"] = generate_linear_operator_householder_inputs()

import numpy as np
import tensorflow as tf

def matrix_transpose_inputs():
    list_of_inputs = []
    
    # Input 1: 2D real tensor
    a = np.array([[1, 2, 3], [4, 5, 6]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: 2D complex tensor with conjugate=True
    a = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": True
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: 3D tensor (batched matrix)
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: 2D tensor with negative values
    a = np.array([[-1, -2, -3], [4, 5, 6]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: 2D tensor with zero values
    a = np.array([[0, 2, 3], [4, 5, 6]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: 3D tensor with different shapes
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: 2D tensor with float values
    a = np.array([[1.5, 2.7], [3.1, 4.8]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: 3D tensor with mixed types
    a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": True
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: 2D tensor with complex values
    a = np.array([[1+1j, 2+2j], [3+3j, 4+4j]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": True
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: 2D tensor with mixed values (float and int)
    a = np.array([[1.0, 2], [3.0, 4]])
    input_dict = {
        "a": a,
        "name": "matrix_transpose",
        "conjugate": False
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.linalg.matrix_transpose"] = matrix_transpose_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_trace_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 3D tensor
    x = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 2D tensor with float values
    x = np.array([[1.5, 2.7], [3.1, 4.9]], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "trace_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D tensor with negative values
    x = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor with zero values
    x = np.array([[0, 0], [0, 0]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensor with large values
    x = np.array([[100, 200], [300, 400]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 2D tensor with complex numbers
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    input_dict = {
        "x": x,
        "name": "trace_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 2D tensor with mixed values
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with varying shapes
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D tensor with mixed values
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.linalg.trace"] = tf_linalg_trace_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_atan2_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    y = np.array([1., 1.])
    x = np.array([1., 1.])
    name = "test1"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    y = np.array([1., -1.])
    x = np.array([1., 1.])
    name = "test2"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    y = np.array([-1., 1.])
    x = np.array([1., -1.])
    name = "test3"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    y = np.array([-1., -1.])
    x = np.array([-1., -1.])
    name = "test4"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    y = np.array([0., 1.])
    x = np.array([1., 0.])
    name = "test5"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    y = np.array([0., 1.])
    x = np.array([-1., 0.])
    name = "test6"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    y = np.array([0., -1.])
    x = np.array([-1., 0.])
    name = "test7"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    y = np.array([0., -1.])
    x = np.array([1., 0.])
    name = "test8"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    y = np.array([[1., 1.], [1., 1.]])
    x = np.array([[1., 1.], [1., 1.]])
    name = "test9"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    y = np.array([[1., -1.], [-1., 1.]])
    x = np.array([[1., 1.], [-1., -1.]])
    name = "test10"
    
    input_dict = {
        "y": y,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.atan2"] = tf_math_atan2_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_bessel_i1e_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([-1., -0.5, 0.5, 1.], dtype=np.float32)
    name = "test1"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([-2.5, -1.2, 0.7, 1.8], dtype=np.float64)
    name = "test2"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([0.0], dtype=np.float32)
    name = "test3"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([-0.1, -0.01, 0.01, 0.1], dtype=np.float64)
    name = "test4"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([[-1., 0.], [1., -2.]], dtype=np.float32)
    name = "test5"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([[-10., -5.], [5., 10.]], dtype=np.float64)
    name = "test6"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([0.001, 0.0001, -0.001], dtype=np.float32)
    name = "test7"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([[0.5, -0.5], [1.0, -1.0]], dtype=np.float64)
    name = "test8"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([-3.14, 2.71, -0.5, 0.5], dtype=np.float32)
    name = "test9"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([[-1.1, -0.1], [0.1, 1.1]], dtype=np.float64)
    name = "test10"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.bessel_i1e"] = generate_bessel_i1e_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_cos_inputs():
    list_of_inputs = []
    
    # Input 1: scalar tensor
    x = np.array(1.5, dtype=np.float32)
    input_dict = {"x": x, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 1D tensor with negative values
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {"x": x, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 2D tensor with mixed values
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"x": x, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: tensor with inf values
    x = np.array([np.inf, -np.inf], dtype=np.float32)
    input_dict = {"x": x, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: complex tensor
    x = np.array([1+2j, 3+4j], dtype=np.complex64)
    input_dict = {"x": x, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: scalar tensor with zero
    x = np.array(0.0, dtype=np.float32)
    input_dict = {"x": x, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 1D tensor with nan values
    x = np.array([np.nan, np.nan], dtype=np.float32)
    input_dict = {"x": x, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 1D tensor with float64 type
    x = np.array([1.5, 2.5], dtype=np.float64)
    input_dict = {"x": x, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 1D tensor with half type
    x = np.array([1.5, 2.5], dtype=np.float16)
    input_dict = {"x": x, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.cos"] = generate_cos_inputs()

import numpy as np
import tensorflow as tf

def tf_math_cumprod_inputs():
    list_of_inputs = []
    
    # Input 1: Basic tensor with axis=0, exclusive=False, reverse=False
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    axis = 0
    exclusive = False
    reverse = False
    name = "cumprod_1"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: Basic tensor with axis=0, exclusive=True, reverse=False
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    axis = 0
    exclusive = True
    reverse = False
    name = "cumprod_2"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: Basic tensor with axis=0, exclusive=False, reverse=True
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    axis = 0
    exclusive = False
    reverse = True
    name = "cumprod_3"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: Basic tensor with axis=0, exclusive=True, reverse=True
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    axis = 0
    exclusive = True
    reverse = True
    name = "cumprod_4"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: Multi-dimensional tensor with axis=1, exclusive=False, reverse=False
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis = 1
    exclusive = False
    reverse = False
    name = "cumprod_5"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: Multi-dimensional tensor with axis=1, exclusive=True, reverse=False
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis = 1
    exclusive = True
    reverse = False
    name = "cumprod_6"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: Multi-dimensional tensor with axis=1, exclusive=False, reverse=True
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis = 1
    exclusive = False
    reverse = True
    name = "cumprod_7"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: Multi-dimensional tensor with axis=1, exclusive=True, reverse=True
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis = 1
    exclusive = True
    reverse = True
    name = "cumprod_8"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: 3D tensor with axis=0, exclusive=False, reverse=False
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axis = 0
    exclusive = False
    reverse = False
    name = "cumprod_9"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: 3D tensor with axis=2, exclusive=True, reverse=True
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axis = 2
    exclusive = True
    reverse = True
    name = "cumprod_10"
    
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.math.cumprod"] = tf_math_cumprod_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_erf_inputs():
    list_of_inputs = []
    
    # Input 1: Single element array
    x = np.array([1.0], dtype=np.float32)
    input_dict = {"x": x, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Multi-dimensional array with positive values
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Multi-dimensional array with negative values
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Mixed positive and negative values
    x = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Single dimension with zero values
    x = np.array([0.0], dtype=np.float32)
    input_dict = {"x": x, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Large values
    x = np.array([10.0, 20.0], dtype=np.float32)
    input_dict = {"x": x, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Float64 array
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Negative values with different shape
    x = np.array([[-1.0], [-2.0], [-3.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Very small values
    x = np.array([0.001, 0.002], dtype=np.float32)
    input_dict = {"x": x, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Mixed values including zero and negative
    x = np.array([[0.0, -1.0], [2.0, -3.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.erf"] = generate_erf_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_floormod_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([2, 2, 2, 2], dtype=np.int32)
    name = "test1"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([10, -5, 8], dtype=np.float32)
    y = np.array([3.0, -2.0, 4.0], dtype=np.float32)
    name = "test2"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[2, 3], [4, 5]], dtype=np.int64)
    name = "test3"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([0.5, -1.5, 2.5], dtype=np.float64)
    y = np.array([1.0, -1.0, 3.0], dtype=np.float64)
    name = "test4"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([100, -200, 300], dtype=np.int8)
    y = np.array([7, -8, 9], dtype=np.int8)
    name = "test5"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    y = np.array([3, 3, 3, 3, 3], dtype=np.int16)
    name = "test6"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([[-10, 20], [-30, 40]], dtype=np.float32)
    y = np.array([5.0, -10.0], dtype=np.float32)
    name = "test7"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([3, 3, 3], dtype=np.uint8)
    name = "test8"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.int32)
    name = "test9"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([1, 2, 3], dtype=np.float32)
    y = np.array([3, 3, 3], dtype=np.float32)
    name = "test10"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.floormod"] = tf_math_floormod_inputs()

import tensorflow as tf
import copy

def tf_math_invert_permutation_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = tf.constant([3, 4, 0, 2, 1], dtype=tf.int32)
    name = "test_1"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = tf.constant([0, 1, 2, 3], dtype=tf.int64)
    name = "test_2"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = tf.constant([4, 3, 2, 1, 0], dtype=tf.int32)
    name = "test_3"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = tf.constant([1, 0, 3, 2], dtype=tf.int64)
    name = "test_4"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = tf.constant([2, 0, 1], dtype=tf.int32)
    name = "test_5"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = tf.constant([5, 4, 3, 2, 1, 0], dtype=tf.int64)
    name = "test_6"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = tf.constant([3, 2, 1, 0], dtype=tf.int32)
    name = "test_7"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = tf.constant([0, 2, 1], dtype=tf.int64)
    name = "test_8"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = tf.constant([1, 3, 2, 0], dtype=tf.int32)
    name = "test_9"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = tf.constant([4, 0, 2, 3, 1], dtype=tf.int64)
    name = "test_10"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.invert_permutation"] = tf_math_invert_permutation_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_is_inf_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([5.0, np.inf, 6.8, np.inf], dtype=np.float64)
    name = "test1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([np.inf, np.nan, -np.inf, 0.0], dtype=np.float32)
    name = "test2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    name = "test3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    name = "test4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([-np.inf, -np.inf, 0.0], dtype=np.float64)
    name = "test5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([1.0, np.nan, 2.0], dtype=np.float32)
    name = "test6"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([[np.inf, 1.0], [2.0, np.inf]], dtype=np.float64)
    name = "test7"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([[-np.inf, np.inf], [np.inf, -np.inf]], dtype=np.float32)
    name = "test8"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    name = "test9"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    name = "test10"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.is_inf"] = tf_math_is_inf_inputs()

import tensorflow as tf
import numpy as np

def generate_lgamma_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([0, 0.5, 1, 4.5, -4, -5.6], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test1"
    }
    list_of_inputs.append(input_dict)

    # Input 2, valid
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test2"
    }
    list_of_inputs.append(input_dict)

    # Input 3, valid
    x = np.array([0.1, 0.5, 1.5, 2.5], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test3"
    }
    list_of_inputs.append(input_dict)

    # Input 4, valid
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test4"
    }
    list_of_inputs.append(input_dict)

    # Input 5, valid
    x = np.array([10.0, 11.0, 12.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test5"
    }
    list_of_inputs.append(input_dict)

    # Input 6, valid
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test6"
    }
    list_of_inputs.append(input_dict)

    # Input 7, valid
    x = np.array([-0.5, -1.5, -2.5], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test7"
    }
    list_of_inputs.append(input_dict)

    # Input 8, valid
    x = np.array([0.25, 0.75, 1.25, 1.75], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test8"
    }
    list_of_inputs.append(input_dict)

    # Input 9, valid
    x = np.array([0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test9"
    }
    list_of_inputs.append(input_dict)

    # Input 10, valid
    x = np.array([5.5, 6.5, 7.5], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test10"
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.math.lgamma"] = generate_lgamma_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_log1p_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with positive values
    x = np.array([0., 0.5, 1., 5.], dtype=np.float32)
    input_dict = {"x": x, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64 tensor with positive values
    x = np.array([0., 0.5, 1., 5.], dtype=np.float64)
    input_dict = {"x": x, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: float32 tensor with negative values
    x = np.array([-0.9, -0.5, 0., 0.5], dtype=np.float32)
    input_dict = {"x": x, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: float64 tensor with negative values
    x = np.array([-0.9, -0.5, 0., 0.5], dtype=np.float64)
    input_dict = {"x": x, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: complex64 tensor with positive values
    x = np.array([1.+0j, 2.+0j, 3.+0j], dtype=np.complex64)
    input_dict = {"x": x, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: complex128 tensor with positive values
    x = np.array([1.+0j, 2.+0j, 3.+0j], dtype=np.complex128)
    input_dict = {"x": x, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: half tensor with positive values
    x = np.array([0., 0.5, 1., 5.], dtype=np.float16)
    input_dict = {"x": x, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: bfloat16 tensor with positive values
    x = np.array([0., 0.5, 1., 5.], dtype=np.float32)
    input_dict = {"x": x, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: scalar tensor (1D array) with positive values
    x = np.array([0.5], dtype=np.float32)
    input_dict = {"x": x, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D tensor with positive values
    x = np.array([[0., 0.5], [1., 5.]], dtype=np.float32)
    input_dict = {"x": x, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.log1p"] = tf_math_log1p_inputs()

import tensorflow as tf
import copy

def tf_math_logical_not_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = tf.constant([True, False])
    name = "test_1"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = tf.constant(False)
    name = "test_2"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = tf.constant([True, False, True])
    name = "test_3"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = tf.constant([[True, False], [False, True]])
    name = "test_4"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = tf.constant([[[True, False], [True, False]], [[False, True], [False, True]]])
    name = "test_5"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - with negative values (though bool type doesn't support negatives, but can be converted)
    x = tf.constant([True, False, True])
    name = "test_6"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = tf.constant([False, False, False])
    name = "test_7"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = tf.constant([True, True, True])
    name = "test_8"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = tf.constant([True, False, True, False])
    name = "test_9"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = tf.constant([False, True, False, True])
    name = "test_10"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.logical_not"] = tf_math_logical_not_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_maximum_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([0., 0., 0., 0.], dtype=np.float32)
    y = np.array([-2., 0., 2., 5.], dtype=np.float32)
    name = "test1"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([-5., 0., 0., 0.], dtype=np.float32)
    y = np.array([-3.], dtype=np.float32)
    name = "test2"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([3, 2, 1], dtype=np.int32)
    name = "test3"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([[-1, -2], [3, 4]], dtype=np.int64)
    y = np.array([[-3, -4], [5, 6]], dtype=np.int64)
    name = "test4"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    y = np.array([1.5, 0.5, 3.5], dtype=np.float64)
    name = "test5"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([10, 20], dtype=np.int32)
    y = np.array([5, 15], dtype=np.int32)
    name = "test6"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([[-10, -20], [-30, -40]], dtype=np.int16)
    y = np.array([[-5, -15], [-25, -35]], dtype=np.int16)
    name = "test7"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    name = "test8"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([3, 2, 1], dtype=np.int32)
    name = "test9"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([[-5, -4, -3], [-2, -1, 0]], dtype=np.float64)
    y = np.array([[-3, -2, -1], [0, 1, 2]], dtype=np.float64)
    name = "test10"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.maximum"] = tf_math_maximum_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_polygamma_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    a = np.array([0.0], dtype=np.float32)
    x = np.array([1.0], dtype=np.float32)
    name = "polygamma_1"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    a = np.array([1.0], dtype=np.float32)
    x = np.array([2.0], dtype=np.float32)
    name = "polygamma_2"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    a = np.array([2.0], dtype=np.float32)
    x = np.array([3.0], dtype=np.float32)
    name = "polygamma_3"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    a = np.array([0.0], dtype=np.float64)
    x = np.array([1.5], dtype=np.float64)
    name = "polygamma_4"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    a = np.array([1.0], dtype=np.float64)
    x = np.array([2.5], dtype=np.float64)
    name = "polygamma_5"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    a = np.array([2.0], dtype=np.float64)
    x = np.array([3.5], dtype=np.float64)
    name = "polygamma_6"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    a = np.array([0.0], dtype=np.float32)
    x = np.array([1.0], dtype=np.float32)
    name = "polygamma_7"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    a = np.array([1.0], dtype=np.float32)
    x = np.array([2.0], dtype=np.float32)
    name = "polygamma_8"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    a = np.array([2.0], dtype=np.float32)
    x = np.array([3.0], dtype=np.float32)
    name = "polygamma_9"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    a = np.array([0.0], dtype=np.float64)
    x = np.array([1.5], dtype=np.float64)
    name = "polygamma_10"
    
    input_dict = {
        "a": a,
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.polygamma"] = tf_math_polygamma_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_real_inputs():
    list_of_inputs = []
    
    # Input 1, valid - real tensor
    input_1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict_1 = {"input": input_1, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    
    # Input 2, valid - complex tensor with real part
    input_2 = np.array([1+2j, 3+4j], dtype=np.complex64)
    input_dict_2 = {"input": input_2, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    
    # Input 3, valid - complex tensor with negative real part
    input_3 = np.array([-1-2j, -3-4j], dtype=np.complex64)
    input_dict_3 = {"input": input_3, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    # Input 4, valid - complex tensor with mixed real and imaginary parts
    input_4 = np.array([1.5+2.5j, 3.7+4.8j], dtype=np.complex64)
    input_dict_4 = {"input": input_4, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5, valid - real tensor with negative values
    input_5 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict_5 = {"input": input_5, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6, valid - complex tensor with zero imaginary part
    input_6 = np.array([1.0+0j, 2.0+0j], dtype=np.complex64)
    input_dict_6 = {"input": input_6, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7, valid - complex tensor with real part in float64
    input_7 = np.array([1.0+2.0j, 3.0+4.0j], dtype=np.complex128)
    input_dict_7 = {"input": input_7, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8, valid - real tensor with multiple dimensions
    input_8 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict_8 = {"input": input_8, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9, valid - complex tensor with real part in float64
    input_9 = np.array([1.0+2.0j, 3.0+4.0j], dtype=np.complex128)
    input_dict_9 = {"input": input_9, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10, valid - real tensor with complex numbers
    input_10 = np.array([1.0+0j, 2.0+0j], dtype=np.complex64)
    input_dict_10 = {"input": input_10, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    return list_of_inputs

generated_inputs["tf.math.real"] = tf_math_real_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def generate_bessel_j1_inputs():
    list_of_inputs = []
    
    # Input 1: Positive float32 values
    x = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Negative float32 values
    x = np.array([-0.5, -1.0, -2.0, -4.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Mixed positive and negative float32 values
    x = np.array([0.5, -1.0, 2.0, -4.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Single element array
    x = np.array([1.5], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Zero values
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Float64 values
    x = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Large values
    x = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Small values
    x = np.array([0.001, 0.0001, 0.00001], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Complex array (1D)
    x = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Multiple dimensions
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.special.bessel_j1"] = generate_bessel_j1_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_squared_difference_inputs():
    list_of_inputs = []
    
    # Input 1: Valid - float32 tensors
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Valid - int32 tensors
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[2, 3], [4, 5]], dtype=np.int32)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Valid - float64 tensors
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Valid - complex64 tensors
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    y = np.array([[2+4j, 6+8j], [10+12j, 14+16j]], dtype=np.complex64)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Valid - complex128 tensors
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    y = np.array([[2+4j, 6+8j], [10+12j, 14+16j]], dtype=np.complex128)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Valid - mixed negative values
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([-2, -4], dtype=np.int32)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Valid - broadcasting tensors
    x = np.array([1, 2], dtype=np.float32)
    y = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Valid - scalar tensors
    x = np.array(5, dtype=np.int32)
    y = np.array(3, dtype=np.int32)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Valid - float64 with negative values
    x = np.array([-1.0, -2.0], dtype=np.float64)
    y = np.array([1.0, 2.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Valid - mixed dimensions
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    y = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.squared_difference"] = tf_math_squared_difference_inputs()

import numpy as np
import tensorflow as tf

def tf_math_zero_fraction_inputs():
    list_of_inputs = []
    
    # Input 1: Empty tensor
    value = np.array([], dtype=np.float32)
    name = "empty_tensor"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 2: Tensor with all zeros
    value = np.array([0, 0, 0, 0], dtype=np.float32)
    name = "all_zeros"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 3: Tensor with mixed values including zeros
    value = np.array([1, 0, 3, 0, 5], dtype=np.float32)
    name = "mixed_values"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 4: Tensor with negative values
    value = np.array([-1, 0, -3, 0, -5], dtype=np.float32)
    name = "negative_values"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 5: 2D tensor with zeros
    value = np.array([[1, 0], [0, 3]], dtype=np.float32)
    name = "2d_tensor"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 6: 3D tensor with zeros
    value = np.array([[[1, 0], [0, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
    name = "3d_tensor"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 7: Tensor with many zeros
    value = np.array([0, 0, 0, 1, 2, 3], dtype=np.float32)
    name = "many_zeros"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 8: Tensor with all non-zero values
    value = np.array([1, 2, 3, 4], dtype=np.float32)
    name = "all_non_zero"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 9: Tensor with only one zero
    value = np.array([1, 2, 3, 0], dtype=np.float32)
    name = "single_zero"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: Tensor with multiple zeros and negative values
    value = np.array([0, -1, 0, 2, -3], dtype=np.float32)
    name = "multiple_zeros_negative"
    input_dict = {
        "value": value,
        "name": name
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.math.zero_fraction"] = tf_math_zero_fraction_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_crelu_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    features = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float32)
    axis = -1
    name = "test_1"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    features = np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.float32)
    axis = 2
    name = "test_2"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    features = np.array([[-1, -2, -3], [4, 5, 6]], dtype=np.float32)
    axis = 0
    name = "test_3"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    features = np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.float32)
    axis = -1
    name = "test_4"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    features = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.float32)
    axis = -1
    name = "test_5"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    features = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.float32)
    axis = 1
    name = "test_6"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    features = np.array([[-1, -2], [3, -4], [-5, -6]], dtype=np.float32)
    axis = 0
    name = "test_7"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    features = np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.float32)
    axis = -1
    name = "test_8"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    features = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis = -1
    name = "test_9"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    features = np.array([[-1, -2], [3, -4], [-5, -6]], dtype=np.float32)
    axis = 1
    name = "test_10"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.crelu"] = tf_nn_crelu_inputs()

import tensorflow as tf
import copy

def tf_isotonic_regression_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_tensor = tf.constant([[3, 1, 2], [1, 3, 4]], dtype=tf.float32)
    decreasing = True
    axis = 1
    
    input_dict = {
        "inputs": input_tensor,
        "decreasing": decreasing,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input_tensor = tf.constant([3, 1, 2], dtype=tf.float32)
    decreasing = True
    axis = 0
    
    input_dict = {
        "inputs": input_tensor,
        "decreasing": decreasing,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input_tensor = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    decreasing = False
    axis = 1
    
    input_dict = {
        "inputs": input_tensor,
        "decreasing": decreasing,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input_tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32)
    decreasing = True
    axis = 0
    
    input_dict = {
        "inputs": input_tensor,
        "decreasing": decreasing,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input_tensor = tf.constant([[-1, -2, -3], [1, 2, 3]], dtype=tf.float32)
    decreasing = True
    axis = 0
    
    input_dict = {
        "inputs": input_tensor,
        "decreasing": decreasing,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input_tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
    decreasing = True
    axis = 1
    
    input_dict = {
        "inputs": input_tensor,
        "decreasing": decreasing,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input_tensor = tf.constant([[-1, 2, -3], [4, -5, 6]], dtype=tf.float32)
    decreasing = False
    axis = 0
    
    input_dict = {
        "inputs": input_tensor,
        "decreasing": decreasing,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input_tensor = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    decreasing = False
    axis = 1
    
    input_dict = {
        "inputs": input_tensor,
        "decreasing": decreasing,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input_tensor = tf.constant([[3, 2, 1], [6, 5, 4]], dtype=tf.float32)
    decreasing = True
    axis = 1
    
    input_dict = {
        "inputs": input_tensor,
        "decreasing": decreasing,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input_tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32)
    decreasing = False
    axis = 0
    
    input_dict = {
        "inputs": input_tensor,
        "decreasing": decreasing,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.isotonic_regression"] = tf_isotonic_regression_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_softmax_inputs():
    list_of_inputs = []
    
    # Input 1
    logits = np.array([-1, 0., 1.], dtype=np.float32)
    axis = -1
    name = "softmax_1"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    logits = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    axis = -1
    name = "softmax_2"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    logits = np.array([[-1, 0., 1.], [-2, 1., 0.]], dtype=np.float32)
    axis = -1
    name = "softmax_3"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    logits = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axis = -1
    name = "softmax_4"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    logits = np.array([[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]], dtype=np.float32)
    axis = -1
    name = "softmax_5"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    logits = np.array([[-1, 0., 1.], [-2, 1., 0.], [-3, 2., 1.]], dtype=np.float32)
    axis = -1
    name = "softmax_6"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    logits = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    axis = -1
    name = "softmax_7"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    logits = np.array([[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]], dtype=np.float32)
    axis = -1
    name = "softmax_8"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    logits = np.array([-1, 0., 1.], dtype=np.float32)
    axis = -1
    name = "softmax_9"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    logits = np.array([[-1, 0., 1.], [-2, 1., 0.], [-3, 2., 1.], [-4, 3., 2.]], dtype=np.float32)
    axis = -1
    name = "softmax_10"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.softmax"] = tf_nn_softmax_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_cosh_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with negative values
    x = np.array([-float('inf'), -9, -0.5, 1, 1.2, 2, 10, float('inf')], dtype=np.float32)
    input_dict = {
        'name': 'Cosh',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64 tensor with negative values
    x = np.array([-float('inf'), -9, -0.5, 1, 1.2, 2, 10, float('inf')], dtype=np.float64)
    input_dict = {
        'name': 'Cosh',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: half tensor with negative values
    x = np.array([-float('inf'), -9, -0.5, 1, 1.2, 2, 10, float('inf')], dtype=np.float16)
    input_dict = {
        'name': 'Cosh',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: bfloat16 tensor with negative values
    x = np.array([-float('inf'), -9, -0.5, 1, 1.2, 2, 10, float('inf')], dtype=np.float32)
    input_dict = {
        'name': 'Cosh',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: complex64 tensor with negative values
    x = np.array([-complex('inf'), -9, -0.5, 1, 1.2, 2, 10, complex('inf')], dtype=np.complex64)
    input_dict = {
        'name': 'Cosh',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: complex128 tensor with negative values
    x = np.array([-complex('inf'), -9, -0.5, 1, 1.2, 2, 10, complex('inf')], dtype=np.complex128)
    input_dict = {
        'name': 'Cosh',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: float32 tensor with positive values
    x = np.array([0.5, 1.2, 2, 10, 100], dtype=np.float32)
    input_dict = {
        'name': 'Cosh',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: float64 tensor with positive values
    x = np.array([0.5, 1.2, 2, 10, 100], dtype=np.float64)
    input_dict = {
        'name': 'Cosh',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: half tensor with positive values
    x = np.array([0.5, 1.2, 2, 10, 100], dtype=np.float16)
    input_dict = {
        'name': 'Cosh',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: bfloat16 tensor with positive values
    x = np.array([0.5, 1.2, 2, 10, 100], dtype=np.float32)
    input_dict = {
        'name': 'Cosh',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Cosh"] = generate_cosh_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_div_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    
    input_dict = {
        "name": "div_1",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([1, 2, 3], dtype=np.int32)
    
    input_dict = {
        "name": "div_2",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    
    input_dict = {
        "name": "div_3",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    y = np.array([[2, 4], [6, 8]], dtype=np.uint8)
    
    input_dict = {
        "name": "div_4",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    y = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    
    input_dict = {
        "name": "div_5",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    
    input_dict = {
        "name": "div_6",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([[1, 2], [3, 4]], dtype=np.int16)
    y = np.array([[1, 2], [3, 4]], dtype=np.int16)
    
    input_dict = {
        "name": "div_7",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([1.0, 2.0], dtype=np.float64)
    
    input_dict = {
        "name": "div_8",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    
    input_dict = {
        "name": "div_9",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    
    input_dict = {
        "name": "div_10",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Div"] = tf_raw_ops_div_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_draw_bounding_boxes_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with 4D image and 3D boxes
    images = np.random.rand(2, 10, 10, 3).astype(np.float32)
    boxes = np.random.rand(2, 3, 4).astype(np.float32)
    
    input_dict = {
        "name": "test1",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Different batch size
    images = np.random.rand(1, 5, 5, 3).astype(np.float32)
    boxes = np.random.rand(1, 2, 4).astype(np.float32)
    
    input_dict = {
        "name": "test2",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input  3: Different image size
    images = np.random.rand(1, 20, 30, 4).astype(np.float32)
    boxes = np.random.rand(1, 1, 4).astype(np.float32)
    
    input_dict = {
        "name": "test3",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Different number of bounding boxes
    images = np.random.rand(3, 15, 15, 3).astype(np.float32)
    boxes = np.random.rand(3, 5, 4).astype(np.float32)
    
    input_dict = {
        "name": "test4",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: With negative values in boxes (valid since API does not restrict negative values)
    images = np.random.rand(2, 10, 10, 3).astype(np.float32)
    boxes = np.random.rand(2, 3, 4).astype(np.float32) - 0.5
    
    input_dict = {
        "name": "test5",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Float values in boxes
    images = np.random.rand(1, 20, 20, 3).astype(np.float32)
    boxes = np.random.rand(1, 2, 4).astype(np.float32)
    
    input_dict = {
        "name": "test6",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Different depth
    images = np.random.rand(2, 10, 10, 1).astype(np.float32)
    boxes = np.random.rand(2, 2, 4).astype(np.float32)
    
    input_dict = {
        "name": "test7",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Different image dimensions (height/width)
    images = np.random.rand(1, 25, 30, 3).astype(np.float32)
    boxes = np.random.rand(1, 1, 4).astype(np.float32)
    
    input_dict = {
        "name": "test8",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: With multiple bounding boxes per image
    images = np.random.rand(3, 10, 10, 3).astype(np.float32)
    boxes = np.random.rand(3, 4, 4).astype(np.float32)
    
    input_dict = {
        "name": "test9",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Different image types (e.g. half)
    images = np.random.rand(2, 15, 15, 3).astype(np.float16)
    boxes = np.random.rand(2, 3, 4).astype(np.float32)
    
    input_dict = {
        "name": "test10",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.DrawBoundingBoxes"] = generate_draw_bounding_boxes_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Elu_inputs():
    list_of_inputs = []
    
    # Input 1: scalar tensor
    features = np.array(1.0, dtype=np.float32)
    input_dict = {"name": "elu_scalar", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: scalar tensor with negative value
    features = np.array(-1.0, dtype=np.float32)
    input_dict = {"name": "elu_negative_scalar", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D tensor
    features = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    input_dict = {"name": "elu_1d", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 1D tensor with negative values
    features = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {"name": "elu_1d_negative", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor
    features = np.array([[1.0, -1.0], [0.0, -2.0]], dtype=np.float32)
    input_dict = {"name": "elu_2d", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensor with negative values
    features = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {"name": "elu_2d_negative", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D tensor
    features = np.array([[[1.0, -1.0], [0.0, -2.0]], [[-3.0, -4.0], [-5.0, -6.0]]], dtype=np.float32)
    input_dict = {"name": "elu_3d", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 3D tensor with negative values
    features = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {"name": "elu_3d_negative", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: float64 tensor
    features = np.array([1.0, -1.0], dtype=np.float64)
    input_dict = {"name": "elu_float64", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: bfloat16 tensor
    features = np.array([1.0, -1.0], dtype=np.float32)  # bfloat16 is not directly supported in numpy so we use float32
    input_dict = {"name": "elu_bfloat16", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Elu"] = tf_raw_ops_Elu_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fact_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_dict = {
        "name": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input_dict = {
        "name": "factorial"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input_dict = {
        "name": "gamma"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input_dict = {
        "name": "beta"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input_dict = {
        "name": "binomial_coefficient"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input_dict = {
        "name": "rising_factorial"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input_dict = {
        "name": "falling_factorial"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input_dict = {
        "name": "double_factorial"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input_dict = {
        "name": "multinomial_coefficient"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input_dict = {
        "name": "partition_function"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Fact"] = tf_raw_ops_fact_inputs()

import numpy as np
import tensorflow as tf

def tf_greater_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([5, 4, 6], dtype=np.int32)
    y = np.array([5, 2, 5], dtype=np.int32)
    input_dict = {
        "name": "test1",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2, valid
    x = np.array([5, 4, 6], dtype=np.float32)
    y = np.array([5, 2, 5], dtype=np.float32)
    input_dict = {
        "name": "test2",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3, valid
    x = np.array([5.0, 4.0, 6.0], dtype=np.float64)
    y = np.array([5.0, 2.0, 5.0], dtype=np.float64)
    input_dict = {
        "name": "test3",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4, valid
    x = np.array([1, 2, 3], dtype=np.int16)
    y = np.array([5, 2, 5], dtype=np.int16)
    input_dict = {
        "name": "test4",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5, valid
    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([5, 2, 5], dtype=np.int8)
    input_dict = {
        "name": "test5",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6, valid
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([5, 2, 5], dtype=np.int64)
    input_dict = {
        "name": "test6",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7, valid
    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([5, 2, 5], dtype=np.uint8)
    input_dict = {
        "name": "test7",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8, valid
    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([5, 2, 5], dtype=np.float16)
    input_dict = {
        "name": "test8",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Greater"] = tf_greater_inputs()

import numpy as np
import tensorflow as tf

def tf_raw_ops_L2Loss_inputs():
    list_of_inputs = []
    
    # Input 1 - 2D tensor with positive values
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"name": "test1", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 2 - 2D tensor with negative values
    t = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {"name": "test2", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 3 - 1D tensor with mixed values
    t = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    input_dict = {"name": "test3", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 4 - 3D tensor
    t = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"name": "test4", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 5 - 1D tensor with float64
    t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"name": "test5", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 6 - 2D tensor with bfloat16
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"name": "test6", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 7 - 2D tensor with half
    t = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"name": "test7", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 8 - 2D tensor with zero values
    t = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    input_dict = {"name": "test8", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 9 - 1D tensor with large values
    t = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    input_dict = {"name": "test9", "t": t}
    list_of_inputs.append(input_dict)
    
    # Input 10 - 2D tensor with decimal values
    t = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    input_dict = {"name": "test10", "t": t}
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.raw_ops.L2Loss"] = tf_raw_ops_L2Loss_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_log_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with negative values
    x = np.array([-1.0, -0.5, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {
        'name': 'log_input_1',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64 tensor with negative values
    x = np.array([-1.0, -0.5, 0.0, 1.0, 2.0], dtype=np.float64)
    input_dict = {
        'name': 'log_input_2',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: complex64 tensor with negative real part
    x = np.array([-1.0+1j, -0.5+1j, 0.0+1j, 1.0+1j, 2.0+1j], dtype=np.complex64)
    input_dict = {
        'name': 'log_input_3',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: complex128 tensor with negative real part
    x = np.array([-1.0+1j, -0.5+1j, 0.0+1j, 1.0+1j, 2.0+1j], dtype=np.complex128)
    input_dict = {
        'name': 'log_input_4',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: float32 tensor with zero values
    x = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    input_dict = {
        'name': 'log_input_5',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: float64 tensor with zero values
    x = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float64)
    input_dict = {
        'name': 'log_input_6',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: float32 tensor with negative values (single dimension)
    x = np.array([-1.0, -0.5, 0.0], dtype=np.float32)
    input_dict = {
        'name': 'log_input_7',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: float64 tensor with negative values (single dimension)
    x = np.array([-1.0, -0.5, 0.0], dtype=np.float64)
    input_dict = {
        'name': 'log_input_8',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: complex64 tensor with negative values (single dimension)
    x = np.array([-1.0+1j, -0.5+1j, 0.0+1j], dtype=np.complex64)
    input_dict = {
        'name': 'log_input_9',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: complex128 tensor with negative values (single dimension)
    x = np.array([-1.0+1j, -0.5+1j, 0.0+1j], dtype=np.complex128)
    input_dict = {
        'name': 'log_input_10',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Log"] = generate_log_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Mean_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis_tensor = np.array([1], dtype=np.int32)
    keep_dims = False
    name = "test1"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    axis_tensor = np.array([0], dtype=np.int32)
    keep_dims = True
    name = "test2"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis_tensor = np.array([1, 2], dtype=np.int64)
    keep_dims = False
    name = "test3"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int8)
    axis_tensor = np.array([0], dtype=np.int32)
    keep_dims = False
    name = "test4"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    axis_tensor = np.array([0, 1], dtype=np.int64)
    keep_dims = True
    name = "test5"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axis_tensor = np.array([-1], dtype=np.int32)
    keep_dims = False
    name = "test6"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axis_tensor = np.array([0, 2], dtype=np.int32)
    keep_dims = False
    name = "test7"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input_tensor = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int16)
    axis_tensor = np.array([0], dtype=np.int64)
    keep_dims = True
    name = "test8"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int64)
    axis_tensor = np.array([0], dtype=np.int32)
    keep_dims = False
    name = "test9"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.float32)
    axis_tensor = np.array([0], dtype=np.int64)
    keep_dims = False
    name = "test10"
    
    input_dict = {
        "keep_dims": keep_dims,
        "name": name,
        "input": input_tensor,
        "axis": axis_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Mean"] = tf_raw_ops_Mean_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_pow_inputs():
    list_of_inputs = []
    
    # Input 1: float32
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    y = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    input_dict = {
        "name": "test1",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    y = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    input_dict = {
        "name": "test2",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: int32
    x = np.array([[2, 3], [4, 5]], dtype=np.int32)
    y = np.array([[2, 3], [4, 5]], dtype=np.int32)
    input_dict = {
        "name": "test3",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: int64
    x = np.array([[2, 3], [4, 5]], dtype=np.int64)
    y = np.array([[2, 3], [4, 5]], dtype=np.int64)
    input_dict = {
        "name": "test4",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: complex64
    x = np.array([[2+3j, 4+5j], [6+7j, 8+9j]], dtype=np.complex64)
    y = np.array([[2+3j, 4+5j], [6+7j, 8+9j]], dtype=np.complex64)
    input_dict = {
        "name": "test5",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: int8
    x = np.array([[2, 3], [4, 5]], dtype=np.int8)
    y = np.array([[2, 3], [4, 5]], dtype=np.int8)
    input_dict = {
        "name": "test6",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: float32 with negative values
    x = np.array([[-2.0, 3.0], [-4.0, 5.0]], dtype=np.float32)
    y = np.array([[2.0, -3.0], [4.0, 5.0]], dtype=np.float32)
    input_dict = {
        "name": "test7",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: int32 with scalar values
    x = np.array(2, dtype=np.int32)
    y = np.array(3, dtype=np.int32)
    input_dict = {
        "name": "test8",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: int32 with 3D array
    x = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.int32)
    y = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.int32)
    input_dict = {
        "name": "test9",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: float32 with float64 mixed types (corrected version)
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    y = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    input_dict = {
        "name": "test10",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Pow"] = tf_raw_ops_pow_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Prod_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 2D tensor with keep_dims=False
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis_tensor = np.array([1], dtype=np.int32)
    keep_dims = False
    name = "test1"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Simple 2D tensor with keep_dims=True
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.float64)
    axis_tensor = np.array([0], dtype=np.int32)
    keep_dims = True
    name = "test2"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor with keep_dims=False
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis_tensor = np.array([0], dtype=np.int64)
    keep_dims = False
    name = "test3"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor with keep_dims=True
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    axis_tensor = np.array([1], dtype=np.int32)
    keep_dims = True
    name = "test4"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 1D tensor with keep_dims=False
    input_tensor = np.array([1, 2, 3], dtype=np.float32)
    axis_tensor = np.array([0], dtype=np.int64)
    keep_dims = False
    name = "test5"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with negative axis values
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.float32)
    axis_tensor = np.array([-1], dtype=np.int32)
    keep_dims = False
    name = "test6"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D tensor with multiple axes
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis_tensor = np.array([0, 1], dtype=np.int64)
    keep_dims = False
    name = "test7"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 2D tensor with keep_dims=True and negative axis
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    axis_tensor = np.array([-1], dtype=np.int32)
    keep_dims = True
    name = "test8"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with all dimensions reduced
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int8)
    axis_tensor = np.array([0, 1, 2], dtype=np.int64)
    keep_dims = True
    name = "test9"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D tensor with complex numbers
    input_tensor = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    axis_tensor = np.array([1], dtype=np.int32)
    keep_dims = False
    name = "test10"
    
    input_dict = {
        "input": input_tensor,
        "axis": axis_tensor,
        "keep_dims": keep_dims,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Prod"] = tf_raw_ops_Prod_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_real_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_tensor = np.array([-2.25 + 4.75j, 3.25 + 5.75j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input_tensor = np.array([1.5 + 2.5j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input_tensor = np.array([[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input_tensor = np.array([0.0 + 0.0j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input_tensor = np.array([-1.0 - 1.0j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input_tensor = np.array([1.0 + 1.0j, -1.0 - 1.0j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input_tensor = np.array([1.0 + 1.0j, -1.0 - 1.0j], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "real7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input_tensor = np.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "real8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input_tensor = np.array([0.5 + 1.5j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input_tensor = np.array([-3.0 - 4.0j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "real10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Real"] = tf_real_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_selu_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    features = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    input_dict = {
        "name": "selu_1",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    features = np.array([[-1., -2., -3.], [4., 5., 6.]], dtype=np.float32)
    input_dict = {
        "name": "selu_2",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    features = np.array([[1., -2., 3.], [-4., 5., -6.]], dtype=np.float32)
    input_dict = {
        "name": "selu_3",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    features = np.array([[-1., -2., -3.], [-4., -5., -6.]], dtype=np.float32)
    input_dict = {
        "name": "selu_4",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    features = np.array([1., 2., 3., 4., 5.], dtype=np.float32)
    input_dict = {
        "name": "selu_5",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    features = np.array([[-1., -2., -3., -4., -5.], [1., 2., 3., 4., 5.]], dtype=np.float32)
    input_dict = {
        "name": "selu_6",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    features = np.array([[-1., -2., -3., -4., -5.], [1., 2., 3., 4., 5.]], dtype=np.float64)
    input_dict = {
        "name": "selu_7",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    features = np.array([[-1., -2., -3., -4., -5.], [1., 2., 3., 4., 5.]], dtype=np.float64)
    input_dict = {
        "name": "selu_8",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    features = np.array([[-1., -2., -3., -4., -5.], [1., 2., 3., 4., 5.]], dtype=np.float64)
    input_dict = {
        "name": "selu_9",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    features = np.array([[-1., -2., -3., -4., -5.], [1., 2., 3., 4., 5.]], dtype=np.float64)
    input_dict = {
        "name": "selu_10",
        "features": tf.constant(features)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Selu"] = tf_raw_ops_selu_inputs()

import numpy as np
import tensorflow as tf

def generate_sinh_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with negative values
    x = np.array([-np.inf, -9, -0.5, 1, 1.2, 2, 10, np.inf], dtype=np.float32)
    input_dict = {"name": "test1", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 2: float64 tensor with negative values
    x = np.array([-np.inf, -9, -0.5, 1, 1.2, 2, 10, np.inf], dtype=np.float64)
    input_dict = {"name": "test2", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 3: half tensor with negative values
    x = np.array([-np.inf, -9, -0.5, 1, 1.2, 2, 10, np.inf], dtype=np.float16)
    input_dict = {"name": "test3", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 4: bfloat16 tensor with negative values
    x = np.array([-np.inf, -9, -0.5, 1, 1.2, 2, 10, np.inf], dtype=np.float32)
    input_dict = {"name": "test4", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 5: complex64 tensor with negative values
    x = np.array([-np.inf, -9, -0.5, 1, 1.2, 2, 10, np.inf], dtype=np.complex64)
    input_dict = {"name": "test5", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 6: complex128 tensor with negative values
    x = np.array([-np.inf, -9, -0.5, 1, 1.2, 2, 10, np.inf], dtype=np.complex128)
    input_dict = {"name": "test6", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 7: float32 tensor with single element
    x = np.array([5.5], dtype=np.float32)
    input_dict = {"name": "test7", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 8: float64 tensor with single element
    x = np.array([5.5], dtype=np.float64)
    input_dict = {"name": "test8", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 9: half tensor with single element
    x = np.array([5.5], dtype=np.float16)
    input_dict = {"name": "test9", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 10: bfloat16 tensor with single element
    x = np.array([5.5], dtype=np.float32)
    input_dict = {"name": "test10", "x": x}
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Sinh"] = generate_sinh_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_softplus_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        "name": "softplus_1",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    features = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {
        "name": "softplus_2",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    features = np.array([1.0], dtype=np.float32)
    input_dict = {
        "name": "softplus_3",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    input_dict = {
        "name": "softplus_4",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    features = np.array([[-1.0, -2.0, -3.0]], dtype=np.float64)
    input_dict = {
        "name": "softplus_5",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    features = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {
        "name": "softplus_6",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    features = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict = {
        "name": "softplus_7",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    features = np.array([[-1.0, -2.0, -3.0, -4.0]], dtype=np.float32)
    input_dict = {
        "name": "softplus_8",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    features = np.array([[-1.0], [-2.0], [-3.0]], dtype=np.float32)
    input_dict = {
        "name": "softplus_9",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    features = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float64)
    input_dict = {
        "name": "softplus_10",
        "features": features
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Softplus"] = generate_softplus_inputs()

import numpy as np
import tensorflow as tf

def generate_sparse_segment_mean_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32 data, int32 indices and segment_ids
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "test_1",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 2: With negative values in data
    data = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "test_2",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 3: Single dimension data with int64 indices and segment_ids
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    input_dict = {
        "sparse_gradient": False,
        "name": "test_3",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 4: Data with multiple dimensions and mixed segment_ids
    data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "test_4",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 5: With repeated segments and different indices
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "test_5",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 6: Data with float64 type and different indices
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "test_6",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 7: Single dimension with different number of elements
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int64)
    segment_ids = np.array([0, 1], dtype=np.int64)
    input_dict = {
        "sparse_gradient": True,
        "name": "test_7",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 8: Mixed data types and dimensions
    data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "test_8",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 9: Non-sequential segments
    data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "test_9",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    # Input 10: Single dimension with different indices and segment_ids
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 1, 2], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "test_10",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(input_dict.copy())

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSegmentMean"] = generate_sparse_segment_mean_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_segment_sum_inputs():
    list_of_inputs = []
    
    # Input 1, valid - 2D tensor with 3 rows
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid - 2D tensor with 3 rows
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid - 2D tensor with 3 rows
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid - 3D tensor with 2 rows
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid - 3D tensor with 3 rows
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 1, 2], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - 1D tensor with 4 elements
    data = np.array([1, 2, 3, 4], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - 2D tensor with 4 rows
    data = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 1, 2, 3], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - 2D tensor with 3 rows, negative values
    data = np.array([[-1, -2, -3], [-4, -5, -6], [7, 8, 9]], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - 2D tensor with 3 rows, different dtypes
    data = np.array([[1.5, 2.7, 3.8], [4.1, 5.2, 6.3], [7.9, 8.1, 9.2]], dtype=np.float64)
    indices = np.array([0, 1], dtype=np.int64)
    segment_ids = np.array([0, 1], dtype=np.int64)
    
    input_dict = {
        "sparse_gradient": False,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - 2D tensor with 2 rows, sparse gradient = True
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    
    input_dict = {
        "sparse_gradient": True,
        "name": "test",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSegmentSum"] = tf_sparse_segment_sum_inputs()

import numpy as np
import tensorflow as tf

def generate_tan_inputs():
    list_of_inputs = []
    
    # Input 1: scalar tensor
    input_tensor = np.array(1.0, dtype=np.float32)
    input_dict = {"name": "tan_1", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 2: 1D tensor with negative values
    input_tensor = np.array([-1.5, -0.5, 0.5], dtype=np.float32)
    input_dict = {"name": "tan_2", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 3: 2D tensor
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"name": "tan_3", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 4: 3D tensor
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"name": "tan_4", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 5: tensor with inf values
    input_tensor = np.array([np.inf, -np.inf, 0.0], dtype=np.float32)
    input_dict = {"name": "tan_5", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 6: tensor with nan values
    input_tensor = np.array([np.nan, np.nan, 1.0], dtype=np.float32)
    input_dict = {"name": "tan_6", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 7: complex tensor
    input_tensor = np.array([1+2j, 2+1j, 3+4j], dtype=np.complex64)
    input_dict = {"name": "tan_7", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 8: half precision tensor
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input_dict = {"name": "tan_8", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 9: bfloat16 tensor
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # Note: numpy doesn't support bfloat16 directly, but using float32 for simulation
    input_dict = {"name": "tan_9", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    # Input 10: tensor with mixed values including zero
    input_tensor = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    input_dict = {"name": "tan_10", "x": input_tensor}
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Tan"] = generate_tan_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_unicode_script_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_tensor = np.array([1, 31, 38], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input_tensor = np.array([0, 1, 2, 3], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input_tensor = np.array([-1, -2, -3], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input_tensor = np.array([100, 200, 300], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    input_tensor = np.array([1000, 2000, 3000], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input_tensor = np.array([10000, 20000, 30000], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    input_tensor = np.array([100000, 200000], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    input_tensor = np.array([1, 31, 38, 100, 200], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    input_tensor = np.array([1, 31, 38, 100, 200, 300], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    input_tensor = np.array([1, 31, 38, 100, 200, 300, 400], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.UnicodeScript"] = tf_raw_ops_unicode_script_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_where_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor with True values
    condition = np.array([[True, False], [True, False]], dtype=bool)
    input_dict = {
        "name": "test1",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 3D tensor with True values
    condition = np.array([[[True, False], [True, False]], [[False, True], [False, True]]], dtype=bool)
    input_dict = {
        "name": "test2",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D tensor with True values
    condition = np.array([True, False, True, False], dtype=bool)
    input_dict = {
        "name": "test3",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor with True values
    condition = np.array([[[[True, False], [False, True]], [[False, True], [True, False]]], [[[False, False], [True, False]], [[True, False], [False, True]]]], dtype=bool)
    input_dict = {
        "name": "test4",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor with all True values
    condition = np.array([[True, True], [True, True]], dtype=bool)
    input_dict = {
        "name": "test5",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensor with all False values
    condition = np.array([[False, False], [False, False]], dtype=bool)
    input_dict = {
        "name": "test6",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D tensor with negative values
    condition = np.array([[[1.5, -0.5], [0.0, 0.0]], [[0.0, 0.25], [0.0, 0.75]]], dtype=float)
    input_dict = {
        "name": "test7",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 3D tensor with complex values
    condition = np.array([[[1.5+0.j, 0.0+0.j], [0.0+0.5j, 0.0+0.j]], [[0.0+0.j, 0.25+1.5j], [0.0+0.j, 0.75+0.j]]], dtype=complex)
    input_dict = {
        "name": "test8",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D tensor with negative values
    condition = np.array([[1.5, -0.5], [0.0, 0.0]], dtype=float)
    input_dict = {
        "name": "test9",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 3D tensor with float values
    condition = np.array([[[1.5, 0.0], [-0.5, 0.0]], [[0.0, 0.25], [0.0, 0.75]]], dtype=float)
    input_dict = {
        "name": "test10",
        "condition": condition
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Where"] = tf_raw_ops_where_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_inputs():
    list_of_inputs = []
    
    # Input 1: Basic string content
    filename = np.array("test_file.txt", dtype=np.string_)
    contents = np.array("Hello World!", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: File with newline characters
    filename = np.array("file_with_newlines.txt", dtype=np.string_)
    contents = np.array("Line1\nLine2\nLine3", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: File with special characters
    filename = np.array("special_chars.txt", dtype=np.string_)
    contents = np.array("Hello@#$%^&*()", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Empty content file
    filename = np.array("empty_file.txt", dtype=np.string_)
    contents = np.array("", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: File with numbers
    filename = np.array("numbers_file.txt", dtype=np.string_)
    contents = np.array("123456789", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: File with spaces
    filename = np.array("spaces_file.txt", dtype=np.string_)
    contents = np.array("   ", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: File with multiple lines
    filename = np.array("multi_line_file.txt", dtype=np.string_)
    contents = np.array("Line1\nLine2\nLine3\nLine4", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: File with tab characters
    filename = np.array("tab_file.txt", dtype=np.string_)
    contents = np.array("Tab\tCharacter\tHere", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: File with mixed content
    filename = np.array("mixed_file.txt", dtype=np.string_)
    contents = np.array("Mixed content\nWith\nNumbers: 123456789", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: File with binary content
    filename = np.array("binary_file.bin", dtype=np.string_)
    contents = np.array(b"\x00\x01\x02\x03", dtype=np.string_)
    input_dict = {
        "name": "string",
        "filename": "string",
        "contents": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.WriteFile"] = generate_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_reverse_inputs():
    list_of_inputs = []
    
    # Input 1: Reverse last dimension (axis = [3])
    tensor_1 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_1 = np.array([3], dtype=np.int32)
    input_dict = {
        "tensor": tensor_1,
        "axis": axis_1,
        "name": "reverse_last_dim"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Reverse first dimension (axis = [0])
    tensor_2 = np.array([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]], dtype=np.int32)
    axis_2 = np.array([0], dtype=np.int32)
    input_dict = {
        "tensor": tensor_2,
        "axis": axis_2,
        "name": "reverse_first_dim"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Reverse middle dimension (axis = [1])
    tensor_3 = np.array([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]], dtype=np.int32)
    axis_3 = np.array([1], dtype=np.int32)
    input_dict = {
        "tensor": tensor_3,
        "axis": axis_3,
        "name": "reverse_middle_dim"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Reverse last two dimensions (axis = [2, 3])
    tensor_4 = np.array([[[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]], dtype=np.int32)
    axis_4 = np.array([2, 3], dtype=np.int32)
    input_dict = {
        "tensor": tensor_4,
        "axis": axis_4,
        "name": "reverse_last_two_dims"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Reverse first two dimensions (axis = [0, 1])
    tensor_5 = np.array([[[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]], dtype=np.int32)
    axis_5 = np.array([0, 1], dtype=np.int32)
    input_dict = {
        "tensor": tensor_5,
        "axis": axis_5,
        "name": "reverse_first_two_dims"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Reverse negative indices (axis = [-1])
    tensor_6 = np.array([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]], dtype=np.int32)
    axis_6 = np.array([-1], dtype=np.int32)
    input_dict = {
        "tensor": tensor_6,
        "axis": axis_6,
        "name": "reverse_negative_index"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Reverse negative indices (axis = [-2])
    tensor_7 = np.array([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]], dtype=np.int32)
    axis_7 = np.array([-2], dtype=np.int32)
    input_dict = {
        "tensor": tensor_7,
        "axis": axis_7,
        "name": "reverse_negative_index_neg"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Reverse multiple dimensions with negative indices (axis = [-1, -2])
    tensor_8 = np.array([[[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]], dtype=np.int32)
    axis_8 = np.array([-1, -2], dtype=np.int32)
    input_dict = {
        "tensor": tensor_8,
        "axis": axis_8,
        "name": "reverse_multiple_negatives"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Reverse zero dimensions (axis = []) with 1D tensor
    tensor_9 = np.array([[0, 1, 2, 3]], dtype=np.int32)
    axis_9 = np.array([], dtype=np.int32)
    input_dict = {
        "tensor": tensor_9,
        "axis": axis_9,
        "name": "reverse_zero_dims"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Reverse last dimension with string tensor
    tensor_10 = np.array([["a", "b", "c"], ["d", "e", "f"]], dtype=np.object_)
    axis_10 = np.array([1], dtype=np.int32)
    input_dict = {
        "tensor": tensor_10,
        "axis": axis_10,
        "name": "reverse_string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.reverse"] = tf_reverse_inputs()

import numpy as np
import tensorflow as tf

def generate_sparse_tensor_inputs():
    list_of_inputs = []
    
    # Input 1: Basic 2D sparse tensor with integer values
    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    dense_shape = np.array([3, 4], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 2: 3D sparse tensor with float values
    indices = np.array([[0, 1, 2], [1, 0, 1]], dtype=np.int64)
    values = np.array([3.14, 2.71], dtype=np.float32)
    dense_shape = np.array([2, 3, 4], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 3: 1D sparse tensor with negative values
    indices = np.array([[0], [2], [4]], dtype=np.int64)
    values = np.array([-1, -2, -3], dtype=np.int32)
    dense_shape = np.array([5], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 4: 2D sparse tensor with mixed types (int and float)
    indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    values = np.array([1, 2.5], dtype=np.float32)
    dense_shape = np.array([2, 3], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 5: 3D sparse tensor with zero values
    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    values = np.array([0, 0], dtype=np.int32)
    dense_shape = np.array([2, 2, 2], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 6: 1D sparse tensor with string values
    indices = np.array([[0], [2]], dtype=np.int64)
    values = np.array(['hello', 'world'], dtype=np.object_)
    dense_shape = np.array([3], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 7: Large sparse tensor (10x10)
    indices = np.array([[0, 0], [9, 9]], dtype=np.int64)
    values = np.array([100, 200], dtype=np.int32)
    dense_shape = np.array([10, 10], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 8: 2D sparse tensor with duplicate indices (but valid for this API)
    indices = np.array([[0, 0], [0, 0]], dtype=np.int64)
    values = np.array([5, 5], dtype=np.int32)
    dense_shape = np.array([3, 3], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 9: 4D sparse tensor (high dimensional)
    indices = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    values = np.array([1000, 2000], dtype=np.int32)
    dense_shape = np.array([2, 3, 4, 5], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: 3D sparse tensor with negative index values (valid for this API)
    indices = np.array([[0, 1, 2], [1, 0, 1]], dtype=np.int64)
    values = np.array([-10, -20], dtype=np.int32)
    dense_shape = np.array([2, 3, 4], dtype=np.int64)
    
    input_dict = {
        "indices": indices,
        "values": values,
        "dense_shape": dense_shape
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.sparse.SparseTensor"] = generate_sparse_tensor_inputs()

import numpy as np
import tensorflow as tf

def tf_train_coordinator_inputs():
    list_of_inputs = []
    
    # Input 1: Empty list
    input_dict = {
        "clean_stop_exception_types": []
    }
    list_of_inputs.append(input_dict)
    
    # Input 2: Single exception type
    input_dict = {
        "clean_stop_exception_types": [ValueError]
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.train.Coordinator"] = tf_train_coordinator_inputs()

import tensorflow as tf
import copy

def tf_autodiff_forward_accumulator_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 2D tensor with 2x2 matrix
    primals = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    tangents = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 1D tensor with 3 elements
    primals = tf.constant([1.0, 2.0, 3.0])
    tangents = tf.constant([1.0, 0.0, 0.0])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor with 2x2x2 matrix
    primals = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    tangents = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Negative values in tensor
    primals = tf.constant([[-1.0, -2.0], [-3.0, -4.0]])
    tangents = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Scalar tensor
    primals = tf.constant(5.0)
    tangents = tf.constant(1.0)
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Large matrix with 3x3 elements
    primals = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    tangents = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Single element tensor
    primals = tf.constant([1.0])
    tangents = tf.constant([1.0])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Mixed tensor with negative values
    primals = tf.constant([[-1.0, 2.0], [3.0, -4.0]])
    tangents = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Tensor with float values
    primals = tf.constant([[1.5, 2.5], [3.5, 4.5]])
    tangents = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Different dimensions in tensor
    primals = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tangents = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    x = tf.constant(1.0)
    
    input_dict = {
        "primals": primals,
        "tangents": tangents,
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.autodiff.ForwardAccumulator"] = tf_autodiff_forward_accumulator_inputs()

import tensorflow as tf
import copy

def tf_experimental_numpy_array_inputs():
    list_of_inputs = []
    
    # Input 1: tensor with dtype specified
    val = tf.constant([1, 2, 3])
    dtype = tf.int32
    copy = True
    ndmin = 0
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: tensor with different dtype, no copy needed
    val = tf.constant([1.0, 2.0, 3.0])
    dtype = tf.float32
    copy = False
    ndmin = 0
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: tensor with ndmin = 1
    val = tf.constant([1, 2, 3])
    dtype = None
    copy = True
    ndmin = 1
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: tensor with ndmin = 2
    val = tf.constant([1, 2, 3])
    dtype = None
    copy = False
    ndmin = 2
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: tensor with negative values
    val = tf.constant([-1, -2, -3])
    dtype = None
    copy = True
    ndmin = 0
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: tensor with float values and copy=True
    val = tf.constant([1.5, 2.7, 3.9])
    dtype = tf.float64
    copy = True
    ndmin = 0
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: tensor with complex values
    val = tf.constant([1+2j, 3+4j])
    dtype = tf.complex64
    copy = False
    ndmin = 0
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: tensor with ndmin = 3
    val = tf.constant([[[1, 2], [3, 4]]])
    dtype = None
    copy = True
    ndmin = 3
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: tensor with zero values
    val = tf.constant([0, 0, 0])
    dtype = None
    copy = False
    ndmin = 0
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: tensor with mixed dimensions
    val = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dtype = None
    copy = True
    ndmin = 1
    
    input_dict = {
        "val": val,
        "dtype": dtype,
        "copy": copy,
        "ndmin": ndmin
    }
    list_of_inputs.append(input_dict.copy())

    return list_of_inputs

generated_inputs["tf.experimental.numpy.array"] = tf_experimental_numpy_array_inputs()

import numpy as np
import tensorflow as tf

def tf_identity_inputs():
    list_of_inputs = []
    
    # Input 1, valid - integer n=3 with float dtype
    input_dict = {
        "n": 3,
        "dtype": np.float64
    }
    list_of_inputs.append(input_dict)
    
    # Input 2, valid - integer n=5 with int dtype
    input_dict = {
        "n": 5,
        "dtype": np.int32
    }
    list_of_inputs.append(input_dict)
    
    # Input 3, valid - integer n=1 with bool dtype
    input_dict = {
        "n": 1,
        "dtype": np.bool_
    }
    list_of_inputs.append(input_dict)
    
    # Input 4, valid - integer n=0 with float dtype (edge case)
    input_dict = {
        "n": 0,
        "dtype": np.float32
    }
    list_of_inputs.append(input_dict)
    
    # Input 5, valid - integer n=2 with float dtype
    input_dict = {
        "n": 2,
        "dtype": np.float32
    }
    list_of_inputs.append(input_dict)
    
    # Input 6, valid - integer n=4 with int dtype
    input_dict = {
        "n": 4,
        "dtype": np.int64
    }
    list_of_inputs.append(input_dict)
    
    # Input 7, valid - integer n=10 with float dtype
    input_dict = {
        "n": 10,
        "dtype": np.float64
    }
    list_of_inputs.append(input_dict)
    
    # Input 8, valid - integer n=7 with bool dtype
    input_dict = {
        "n": 7,
        "dtype": np.bool_
    }
    list_of_inputs.append(input_dict)
    
    # Input 9, valid - integer n=100 with float dtype
    input_dict = {
        "n": 100,
        "dtype": np.float64
    }
    list_of_inputs.append(input_dict)
    
    # Input 10, valid - integer n=50 with int dtype
    input_dict = {
        "n": 50,
        "dtype": np.int32
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.identity"] = tf_identity_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_logaddexp_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([4.0, 5.0, 6.0])
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    x2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x1 = np.array([1.0])
    x2 = np.array([2.0])
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x1 = np.array([-1.0, -2.0, -3.0])
    x2 = np.array([-4.0, -5.0, -6.0])
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x1 = np.array([0.0, 1.0, 2.0])
    x2 = np.array([3.0, 4.0, 5.0])
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x1 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    x2 = np.array([[5.0, -6.0], [-7.0, 8.0]])
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x1 = np.array([1.0, 2.0, 3.0, 4.0])
    x2 = np.array([5.0, 6.0, 7.0, 8.0])
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x1 = np.array([-1.0, -2.0, -3.0])
    x2 = np.array([4.0, 5.0, 6.0])
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x1 = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    x2 = np.array([[[5.0, 6.0], [7.0, 8.0]]])
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x1 = np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    x2 = np.array([[[[5.0, 6.0]], [[7.0, 8.0]]]])
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.logaddexp"] = tf_logaddexp_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_reciprocal_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([1, 2, 3, 4, 5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([[1, 2], [3, 4]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([-1, -2, -3, -4, -5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([0.5, 1.5, 2.5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([[0.5, 1.5], [2.5, 3.5]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([[-0.5, -1.5], [-2.5, -3.5]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([1, 2, 3, 4, 5])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.reciprocal"] = tf_reciprocal_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_rot90_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    m = np.array([[1, 2, 3], [4, 5, 6]])
    k = 1
    axes = (0, 1)
    
    input_dict = {
        "m": m,
        "k": k,
        "axes": axes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = 2
    axes = (0, 1)
    
    input_dict = {
        "m": m,
        "k": k,
        "axes": axes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    m = np.array([[1, 2], [3, 4]])
    k = -1
    axes = (0, 1)
    
    input_dict = {
        "m": m,
        "k": k,
        "axes": axes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    m = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    k = 3
    axes = (1, 2)
    
    input_dict = {
        "m": m,
        "k": k,
        "axes": axes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = -2
    axes = (0, 1)
    
    input_dict = {
        "m": m,
        "k": k,
        "axes": axes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    m = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    k = 1
    axes = (0, 2)
    
    input_dict = {
        "m": m,
        "k": k,
        "axes": axes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = 0
    axes = (0, 1)
    
    input_dict = {
        "m": m,
        "k": k,
        "axes": axes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k = 0
    axes = (1, 0)
    
    input_dict = {
        "m": m,
        "k": k,
        "axes": axes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    m = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    k = 1
    axes = (0, 1)
    
    input_dict = {
        "m": m,
        "k": k,
        "axes": axes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    m = np.array([[1, 2], [3, 4]])
    k = -1
    axes = (1, 0)
    
    input_dict = {
        "m": m,
        "k": k,
        "axes": axes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.rot90"] = tf_rot90_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_zeros_like_inputs():
    list_of_inputs = []
    
    # Input 1: 2D array with float dtype
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"a": a, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 3D array with int dtype
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {"a": a, "dtype": np.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D array with bool dtype
    a = np.array([True, False, True], dtype=np.bool_)
    input_dict = {"a": a, "dtype": np.bool_}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D array with complex dtype
    a = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    input_dict = {"a": a, "dtype": np.complex64}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 1D array with float64 dtype
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"a": a, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D array with uint8 dtype
    a = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    input_dict = {"a": a, "dtype": np.uint8}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D array with int64 dtype
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    input_dict = {"a": a, "dtype": np.int64}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 1D array with complex128 dtype
    a = np.array([1+2j, 3+4j], dtype=np.complex128)
    input_dict = {"a": a, "dtype": np.complex128}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D array with float32 dtype
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"a": a, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D array with int32 dtype
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {"a": a, "dtype": np.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.zeros_like"] = tf_zeros_like_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_adjust_gamma_inputs():
    list_of_inputs = []
    
    # Input 1, valid - simple 2D image
    image = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    gamma = 0.5
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid - 3D image with different gamma value
    image = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    gamma = 2.0
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid - 4D image with gamma = 0.2
    image = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]], dtype=np.float32)
    gamma = 0.2
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid - 4D image with gamma = 3.0
    image = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]], dtype=np.float32)
    gamma = 3.0
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid - 2D image with negative gamma (not allowed but we try)
    image = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    gamma = 0.5
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - 3D image with gamma = 1.5
    image = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    gamma = 1.5
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - 3D image with gamma = 0.9
    image = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    gamma = 0.9
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - 4D image with gamma = 0.5
    image = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]], dtype=np.float32)
    gamma = 0.5
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - 4D image with gamma = 1.0
    image = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]], dtype=np.float32)
    gamma = 1.0
    gain = 1.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - 3D image with gamma = 2.5 and gain = 2.0
    image = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    gamma = 2.5
    gain = 2.0
    input_dict = {
        "image": image,
        "gamma": gamma,
        "gain": gain
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.adjust_gamma"] = tf_image_adjust_gamma_inputs()

import tensorflow as tf
import copy

def tf_image_random_jpeg_quality_inputs():
    list_of_inputs = []
    
    # Input 1
    image = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.uint8)
    min_jpeg_quality = 75
    max_jpeg_quality = 95
    seed = 42
    
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2
    image = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.uint8)
    min_jpeg_quality = 50
    max_jpeg_quality = 100
    seed = 123
    
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    image = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.uint8)
    min_jpeg_quality = 0
    max_jpeg_quality = 50
    seed = 456
    
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    image = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.uint8)
    min_jpeg_quality = 10
    max_jpeg_quality = 80
    seed = 789
    
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    image = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.uint8)
    min_jpeg_quality = 10
    max_jpeg_quality = 100
    seed = 100
    
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    image = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.uint8)
    min_jpeg_quality = 75
    max_jpeg_quality = 95
    seed = 200
    
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    image = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.uint8)
    min_jpeg_quality = 20
    max_jpeg_quality = 30
    seed = 300
    
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    image = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.uint8)
    min_jpeg_quality = 10
    max_jpeg_quality = 50
    seed = 400
    
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    image = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.uint8)
    min_jpeg_quality = 60
    max_jpeg_quality = 90
    seed = 500
    
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    image = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.uint8)
    min_jpeg_quality = 30
    max_jpeg_quality = 70
    seed = 600
    
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.random_jpeg_quality"] = tf_image_random_jpeg_quality_inputs()

import tensorflow as tf
import numpy as np
import copy

def sobel_edges_inputs():
    list_of_inputs = []
    
    # Input 1: Basic 3D tensor (batch_size=1, height=28, width=28, channels=3)
    image = np.random.uniform(0, 255, [1, 28, 28, 3]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor with batch_size=2
    image = np.random.uniform(0, 255, [2, 16, 16, 3]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 4D tensor with batch_size=1, height=32, width=32, channel=1
    image = np.random.uniform(0, 255, [1, 32, 32, 1]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor with batch_size=1, height=64, width=64, channel=4
    image = np.random.uniform(0, 255, [1, 64, 64, 4]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 4D tensor with batch_size=1, height=10, width=10, channel=3
    image = np.random.uniform(0, 255, [1, 10, 10, 3]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with batch_size=1, height=32, width=32, channel=5
    image = np.random.uniform(0, 255, [1, 32, 32, 5]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 4D tensor with batch_size=3, height=8, width=8, channel=3
    image = np.random.uniform(0, 255, [3, 8, 8, 3]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor with batch_size=1, height=20, width=20, channel=6
    image = np.random.uniform(0, 255, [1, 20, 20, 6]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D tensor with batch_size=1, height=15, width=15, channel=2
    image = np.random.uniform(0, 255, [1, 15, 15, 2]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with batch_size=1, height=30, width=30, channel=7
    image = np.random.uniform(0, 255, [1, 30, 30, 7]).astype(np.float32)
    input_dict = {"image": tf.convert_to_tensor(image)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.sobel_edges"] = sobel_edges_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 3D tensor with seed
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    seed = np.array([1, 2], dtype=np.int32)
    max_delta = 0.2
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor with seed
    image = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]], dtype=np.float32)
    seed = np.array([3, 4], dtype=np.int32)
    max_delta = 0.1
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Zero delta value (valid)
    image = np.array([[[0.5, 0.4, 0.3], [0.2, 0.1, 0.0]], [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]], dtype=np.float32)
    seed = np.array([5, 6], dtype=np.int32)
    max_delta = 0.0
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Large delta value (valid)
    image = np.array([[[0.5, 0.4, 0.3], [0.2, 0.1, 0.0]], [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]], dtype=np.float32)
    seed = np.array([7, 8], dtype=np.int32)
    max_delta = 0.4
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Single pixel image
    image = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    seed = np.array([9, 10], dtype=np.int32)
    max_delta = 0.2
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Different color channels (RGB)
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    seed = np.array([11, 12], dtype=np.int32)
    max_delta = 0.5
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Large number of channels
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    seed = np.array([13, 14], dtype=np.int32)
    max_delta = 0.4
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Float values with different dimensions
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    seed = np.array([15, 16], dtype=np.int32)
    max_delta = 0.3
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Large delta value with different seed
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    seed = np.array([17, 18], dtype=np.int32)
    max_delta = 0.5
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Another valid case with zero delta
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    seed = np.array([19, 20], dtype=np.int32)
    max_delta = 0.0
    
    input_dict = {
        "image": image,
        "max_delta": max_delta,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.stateless_random_hue"] = generate_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_linear_operator_circulant2d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic 2D spectrum with complex dtype
    spectrum = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float32)
    input_output_dtype = tf.complex64
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - 3D spectrum with real dtype
    spectrum = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]], dtype=np.float32)
    input_output_dtype = tf.float32
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - 4D spectrum with complex dtype and non-singular flag
    spectrum = np.array([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], [[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]]], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = True
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - 2D spectrum with real dtype and self-adjoint flag
    spectrum = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float32)
    input_output_dtype = tf.float32
    is_non_singular = None
    is_self_adjoint = True
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - 2D spectrum with complex dtype and positive definite flag
    spectrum = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = True
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - 3D spectrum with real dtype and square flag set to True (this is valid)
    spectrum = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=np.float32)
    input_output_dtype = tf.float32
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - 2D spectrum with negative values and complex dtype
    spectrum = np.array([[-1., 2., 3.], [4., -5., 6.], [7., 8., -9.]], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - 4D spectrum with complex dtype and different dimensions
    spectrum = np.array([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], [[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]]], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - 3D spectrum with real dtype and different batch dimensions
    spectrum = np.array([[[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]], dtype=np.float32)
    input_output_dtype = tf.float32
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - 2D spectrum with complex dtype and different name
    spectrum = np.array([[1., 2.], [3., 4.]], dtype=np.complex64)
    input_output_dtype = tf.complex64
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None
    is_square = True
    name = "LinearOperatorCirculant2D"
    
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": input_output_dtype,
        "is_non_singular": is_non_singular,
        "is_self_adjoint": is_self_adjoint,
        "is_positive_definite": is_positive_definite,
        "is_square": is_square,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorCirculant2D"] = generate_linear_operator_circulant2d_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_bessel_i0_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([1., 2., 3.], dtype=np.float32)
    name = "test1"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([-1., -2., -3.], dtype=np.float32)
    name = "test2"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    name = "test3"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([-0.5, -1.5, -2.5], dtype=np.float64)
    name = "test4"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([0.0], dtype=np.float32)
    name = "test5"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    name = "test6"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([[-1., -2.], [3., 4.]], dtype=np.float32)
    name = "test7"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    name = "test8"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([[-0.5, -1.5], [2.5, 3.5]], dtype=np.float64)
    name = "test9"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([[-1., 0., 1.], [2., 3., 4.]], dtype=np.float32)
    name = "test10"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.bessel_i0"] = generate_bessel_i0_inputs()

import tensorflow as tf
import numpy as np

def tf_math_floor_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with positive values
    x = np.array([1.3324, -1.5, 5.555, -2.532, 0.99, float("inf")], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "floor_1"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2: float64 tensor with negative values
    x = np.array([-1.3324, -1.5, -5.555, -2.532, -0.99, float("-inf")], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "floor_2"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3: bfloat16 tensor with mixed values
    x = np.array([1.33, -1.5, 5.55, -2.53, 0.99, float("inf")], dtype=np.float16)
    input_dict = {
        "x": x,
        "name": "floor_3"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4: float32 tensor with one dimension
    x = np.array([1.33, -1.5, 5.55, -2.53, 0.99], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "floor_4"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5: float64 tensor with two dimensions
    x = np.array([[1.33, -1.5], [5.55, -2.53]], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "floor_5"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6: half tensor with one dimension
    x = np.array([1.33, -1.5, 5.55, -2.53, 0.99], dtype=np.float16)
    input_dict = {
        "x": x,
        "name": "floor_6"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7: float32 tensor with scalar value
    x = np.array(1.33, dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "floor_7"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8: float64 tensor with negative values
    x = np.array([-1.33, -1.5, -5.55, -2.53], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "floor_8"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9: float32 tensor with decimal values
    x = np.array([1.33, -1.5, 5.55, -2.53, 0.99], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "floor_9"
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10: bfloat16 tensor with large values
    x = np.array([1.33, -1.5, 5.55, -2.53, 0.99], dtype=np.float16)
    input_dict = {
        "x": x,
        "name": "floor_10"
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.math.floor"] = tf_math_floor_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_nextafter_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with positive values
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64 tensor with negative values
    x1 = np.array([-1.0, -2.0], dtype=np.float64)
    x2 = np.array([-2.0, -3.0], dtype=np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: float32 tensor with zero values
    x1 = np.array([0.0, 0.0], dtype=np.float32)
    x2 = np.array([1e-45, 1e-45], dtype=np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: float64 tensor with subnormal values
    x1 = np.array([1e-30, 1e-30], dtype=np.float64)
    x2 = np.array([1e-31, 1e-31], dtype=np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: float32 tensor with mixed values
    x1 = np.array([1.5, 2.7, 3.1], dtype=np.float32)
    x2 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: float64 tensor with single element
    x1 = np.array([1.0], dtype=np.float64)
    x2 = np.array([2.0], dtype=np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: float32 tensor with high precision values
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: float64 tensor with different dimensions
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x2 = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: float32 tensor with nan values
    x1 = np.array([np.nan, 1.0], dtype=np.float32)
    x2 = np.array([1.0, 2.0], dtype=np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: float64 tensor with large values
    x1 = np.array([1e30, 2e30], dtype=np.float64)
    x2 = np.array([2e30, 3e30], dtype=np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.nextafter"] = tf_math_nextafter_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_special_fresnel_cos_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([-1., -0.1, 0.1, 1.], dtype=np.float32)
    name = "test1"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x = np.array([0.5, -0.5, 1.5, -1.5], dtype=np.float32)
    name = "test2"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x = np.array([0.0], dtype=np.float32)
    name = "test3"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x = np.array([-10., 10., -5., 5.], dtype=np.float32)
    name = "test4"

    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x = np.array([0.99, -0.99, 1.99, -1.99], dtype=np.float32)
    name = "test5"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    name = "test6"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x = np.array([[-1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    name = "test7"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    name = "test8"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x = np.array([1e-5, -1e-5, 1e-3, -1e-3], dtype=np.float32)
    name = "test9"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float64)
    name = "test10"
    
    input_dict = {
        "x": x,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.special.fresnel_cos"] = tf_math_special_fresnel_cos_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_unsorted_segment_max_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], dtype=np.int32)
    segment_ids = np.array([0, 1, 0], dtype=np.int32)
    num_segments = 2
    name = "test1"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    segment_ids = np.array([0, 1], dtype=np.int64)
    num_segments = 2
    name = "test2"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    num_segments = 3
    name = "test3"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    segment_ids = np.array([0, 0, 1], dtype=np.int64)
    num_segments = 2
    name = "test4"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int16)
    segment_ids = np.array([0, 1], dtype=np.int32)
    num_segments = 2
    name = "test5"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    data = np.array([1.5, 2.7, 3.9, 4.1], dtype=np.float32)
    segment_ids = np.array([0, 1, 0, 1], dtype=np.int64)
    num_segments = 2
    name = "test6"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    num_segments = 2
    name = "test7"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    data = np.array([[-1, -2, -3], [4, 5, 6], [7, 8, 9]], dtype=np.int8)
    segment_ids = np.array([0, 1, 0], dtype=np.int64)
    num_segments = 2
    name = "test8"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    data = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=np.int16)
    segment_ids = np.array([0, 1], dtype=np.int64)
    num_segments = 2
    name = "test9"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float16)
    segment_ids = np.array([0, 1], dtype=np.int32)
    num_segments = 2
    name = "test10"
    
    input_dict = {
        "data": data,
        "segment_ids": segment_ids,
        "num_segments": num_segments,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.unsorted_segment_max"] = tf_unsorted_segment_max_inputs()

import tensorflow as tf
import numpy as np
import copy

def compute_accidental_hits_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    true_classes = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    sampled_candidates = np.array([2, 4, 5, 6, 7], dtype=np.int64)
    num_true = 3
    seed = 0
    name = "accidental_hit_1"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    true_classes = np.array([[1], [4], [5]], dtype=np.int64)
    sampled_candidates = np.array([1, 4, 5, 7], dtype=np.int64)
    num_true = 1
    seed = 42
    name = "accidental_hit_2"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    true_classes = np.array([[10, 20], [30, 40]], dtype=np.int64)
    sampled_candidates = np.array([10, 20, 30, 40], dtype=np.int64)
    num_true = 2
    seed = 123
    name = "accidental_hit_3"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    true_classes = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    num_true = 5
    seed = 456
    name = "accidental_hit_4"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    true_classes = np.array([[100], [200]], dtype=np.int64)
    sampled_candidates = np.array([100, 200, 300], dtype=np.int64)
    num_true = 1
    seed = 789
    name = "accidental_hit_5"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    true_classes = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    num_true = 2
    seed = 0
    name = "accidental_hit_6"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    true_classes = np.array([[1], [2], [3], [4]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    num_true = 1
    seed = 10
    name = "accidental_hit_7"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    true_classes = np.array([[100, 200, 300], [400, 500, 600]], dtype=np.int64)
    sampled_candidates = np.array([100, 200, 300, 400, 500], dtype=np.int64)
    num_true = 3
    seed = 20
    name = "accidental_hit_8"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    true_classes = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    num_true = 4
    seed = 30
    name = "accidental_hit_9"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    true_classes = np.array([[1], [2], [3]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    num_true = 1
    seed = 40
    name = "accidental_hit_10"
    
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.compute_accidental_hits"] = compute_accidental_hits_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_softsign_inputs():
    list_of_inputs = []
    
    # Input 1: 1D tensor with positive values
    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"features": features, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 1D tensor with negative values
    features = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {"features": features, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 2D tensor with mixed positive and negative values
    features = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    input_dict = {"features": features, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor with positive values
    features = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"features": features, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor with all negative values
    features = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    input_dict = {"features": features, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 1D tensor with float64 values
    features = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"features": features, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 1D tensor with bfloat16 values (simulated as float32)
    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"features": features, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 1D tensor with half values (simulated as float32)
    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"features": features, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 1D tensor with zero values
    features = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"features": features, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 1D tensor with large values
    features = np.array([100.0, -100.0, 50.0], dtype=np.float32)
    input_dict = {"features": features, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.softsign"] = tf_nn_softsign_inputs()

import numpy as np
import tensorflow as tf
import copy

def generate_fake_quant_with_min_max_vars_per_channel_gradient_inputs():
    list_of_inputs = []

    # Input 1, valid
    gradients = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    min_tensor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test1"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    gradients = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    min_tensor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test2"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    gradients = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_tensor = np.array([0.0, 0.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test3"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    gradients = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [0.0, 1.0, 2.0]]], dtype=np.float32)
    input_tensor = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [0.0, 1.0, 2.0]]], dtype=np.float32)
    min_tensor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test4"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    gradients = np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    input_tensor = np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    min_tensor = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test5"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid
    gradients = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_tensor = np.array([-1.0, -1.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0], dtype=np.float32)
    num_bits = 8
    narrow_range = True
    name = "test6"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid
    gradients = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_tensor = np.array([-1.0, -1.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0], dtype=np.float32)
    num_bits = 16
    narrow_range = False
    name = "test7"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    gradients = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_tensor = np.array([-1.0, -1.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0], dtype=np.float32)
    num_bits = 16
    narrow_range = True
    name = "test8"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid
    gradients = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_tensor = np.array([-1.0, -1.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test9"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid
    gradients = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_tensor = np.array([-1.0, -1.0], dtype=np.float32)
    max_tensor = np.array([1.0, 1.0], dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test10"

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "min": min_tensor,
        "max": max_tensor,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient"] = generate_fake_quant_with_min_max_vars_per_channel_gradient_inputs()

import numpy as np
import tensorflow as tf

def tf_raw_ops_cross_inputs():
    list_of_inputs = []
    
    # Input 1, valid - 3-element vector
    a = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([4, 5, 6], dtype=np.float32)
    input_dict = {
        "name": "cross_1",
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2, valid - 3-element vector
    a = np.array([1.5, 2.7, 3.9], dtype=np.float64)
    b = np.array([4.1, 5.2, 6.8], dtype=np.float64)
    input_dict = {
        "name": "cross_2",
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3, valid - 3-element vector with negative values
    a = np.array([-1, 2, -3], dtype=np.int32)
    b = np.array([4, -5, 6], dtype=np.int32)
    input_dict = {
        "name": "cross_3",
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4, valid - 3-element vector with int16 type
    a = np.array([10, 20, 30], dtype=np.int16)
    b = np.array([40, 50, 60], dtype=np.int16)
    input_dict = {
        "name": "cross_4",
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5, valid - 3-element vector with int8 type
    a = np.array([1, 2, 3], dtype=np.int8)
    b = np.array([4, 5, 6], dtype=np.int8)
    input_dict = {
        "name": "cross_5",
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6, valid - 3-element vector with int64 type
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([4, 5, 6], dtype=np.int64)
    input_dict = {
        "name": "cross_6",
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7, valid - 3-element vector with uint8 type
    a = np.array([1, 2, 3], dtype=np.uint8)
    b = np.array([4, 5, 6], dtype=np.uint8)
    input_dict = {
        "name": "cross_7",
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8, valid - 3-element vector with half type
    a = np.array([1, 2, 3], dtype=np.float16)
    b = np.array([4, 5, 6], dtype=np.float16)
    input_dict = {
        "name": "cross_8",
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9, valid - 3-element vector with bfloat16 type
    a = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([4, 5, 6], dtype=np.float32)
    input_dict = {
        "name": "cross_9",
        "a": a,
        "b": b
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Cross"] = tf_raw_ops_cross_inputs()

import numpy as np
import tensorflow as tf
import copy

def generate_diag_part_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor with diagonal elements
    input1 = np.array([[1, 0, 0],
                     [0, 2, 0],
                     [0, 0, 3]], dtype=np.float32)
    input_dict = {
        "name": "diag_part_input_1",
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor with diagonal elements
    input2 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                      [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.float64)
    input_dict = {
        "name": "diag_part_input_2",
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 2D tensor with negative values
    input3 = np.array([[-1, 0],
                      [0, -2]], dtype=np.float32)
    input_dict = {
        "name": "diag_part_input_3",
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor with float values
    input4 = np.array([[[[1.5, 2.7], [3.9, 4.1]], [[5.2, 6.8], [7.3, 8.6]]],
                      [[[9.4, 10.1], [11.7, 12.9]], [[13.2, 14.8], [15.3, 16.4]]]], dtype=np.float64)
    input_dict = {
        "name": "diag_part_input_4",
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor with complex values
    input5 = np.array([[1+2j, 0],
                      [0, 3+4j]], dtype=np.complex64)
    input_dict = {
        "name": "diag_part_input_5",
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensor with zero diagonal elements
    input6 = np.array([[0, 0],
                      [0, 0]], dtype=np.int32)
    input_dict = {
        "name": "diag_part_input_6",
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 4D tensor with mixed values
    input7 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                      [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.float32)
    input_dict = {
        "name": "diag_part_input_7",
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 2D tensor with large values
    input8 = np.array([[100, 0],
                      [0, 200]], dtype=np.int64)
    input_dict = {
        "name": "diag_part_input_8",
        "input": input8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D tensor with negative diagonal elements
    input9 = np.array([[-1, -2],
                      [-3, -4]], dtype=np.float32)
    input_dict = {
        "name": "diag_part_input_9",
        "input": input9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D tensor with zero diagonal elements
    input10 = np.array([[0, 0],
                       [0, 0]], dtype=np.int32)
    input_dict = {
        "name": "diag_part_input_10",
        "input": input10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.DiagPart"] = generate_diag_part_inputs()

import tensorflow as tf
import numpy as np
import copy

def generate_erf_inputs():
    list_of_inputs = []
    
    # Input 1: Single element tensor
    x = np.array([1.0], dtype=np.float32)
    input_dict = {'name': 'test_1', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D tensor with positive values
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {'name': 'test_2', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor with negative values
    x = np.array([[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]], dtype=np.float32)
    input_dict = {'name': 'test_3', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Mixed positive and negative values
    x = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]], dtype=np.float32)
    input_dict = {'name': 'test_4', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Single negative value
    x = np.array([-1.0], dtype=np.float32)
    input_dict = {'name': 'test_5', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Large values
    x = np.array([10.0, 20.0], dtype=np.float32)
    input_dict = {'name': 'test_6', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Float64 type
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {'name': 'test_7', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Half type
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    input_dict = {'name': 'test_8', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Bfloat16 type
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x = x.astype(np.float16)  # This simulates bfloat16
    input_dict = {'name': 'test_9', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Very small values
    x = np.array([0.001, 0.0001], dtype=np.float32)
    input_dict = {'name': 'test_10', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Erf"] = generate_erf_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_imag_inputs():
    list_of_inputs = []
    
    # Input 1: Complex64 with real and imaginary parts
    input_tensor = np.array([1+2j, 3+4j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "imag_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Complex128 with real and imaginary parts
    input_tensor = np.array([1+2j, 3+4j], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "imag_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Complex64 with negative imaginary parts
    input_tensor = np.array([-1-2j, -3-4j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "imag_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Complex128 with mixed real and imaginary parts
    input_tensor = np.array([0+1j, 2+3j], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "imag_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Complex64 with zero imaginary part
    input_tensor = np.array([1+0j, 3+0j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "imag_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Complex128 with zero real part
    input_tensor = np.array([0+1j, 0+3j], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "imag_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Complex64 with large imaginary parts
    input_tensor = np.array([1+100j, 3+200j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "imag_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Complex128 with small imaginary parts
    input_tensor = np.array([1+0.1j, 3+0.2j], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "imag_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Complex64 with complex numbers in 2D array
    input_tensor = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "imag_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Complex128 with complex numbers in 3D array
    input_tensor = np.array([[[1+2j, 3+4j], [5+6j, 7+8j]], [[9+10j, 11+12j], [13+14j, 15+16j]]], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "imag_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Imag"] = tf_imag_inputs()

import numpy as np
import tensorflow as tf
import copy

def generate_lrn_inputs():
    list_of_inputs = []
    
    # Input 1: 4D tensor with float32 dtype
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "depth_radius": 5,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor with half dtype
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float16)
    input_dict = {
        "input": input_tensor,
        "depth_radius": 3,
        "bias": 2.0,
        "alpha": 0.5,
        "beta": 0.75,
        "name": "lrn_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 4D tensor with bfloat16 dtype
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_tensor = input_tensor.astype(np.float16)
    input_dict = {
        "input": input_tensor,
        "depth_radius": 7,
        "bias": 0.5,
        "alpha": 2.0,
        "beta": 0.25,
        "name": "lrn_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor with negative values
    input_tensor = np.random.rand(1, 2, 3, 4).astype(np.float32)
    input_tensor[0, 0, 0, 0] = -1.0
    input_dict = {
        "input": input_tensor,
        "depth_radius": 5,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 4D tensor with zero values
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_tensor[0, 0, 0, 0] = 0.0
    input_dict = {
        "input": input_tensor,
        "depth_radius": 3,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with different shape
    input_tensor = np.random.rand(1, 2, 3, 4).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "depth_radius": 5,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 4D tensor with different shape
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "depth_radius": 2,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor with different shape
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "depth_radius": 1,
        "bias": 0.1,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D tensor with different shape
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "depth_radius": 5,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with different shape
    input_tensor = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "depth_radius": 4,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.LRN"] = generate_lrn_inputs()

import numpy as np
import tensorflow as tf

def lgamma_inputs():
    list_of_inputs = []
    
    # Input 1: Scalar tensor
    x = np.array(5.0, dtype=np.float32)
    input_dict = {"name": "lgamma_1", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 2: 1D array with negative values
    x = np.array([-4.0, -5.6], dtype=np.float32)
    input_dict = {"name": "lgamma_2", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 3: 1D array with positive values
    x = np.array([0.5, 1.0, 4.5], dtype=np.float32)
    input_dict = {"name": "lgamma_3", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 4: 2D array with mixed values
    x = np.array([[0, 0.5], [1, 4.5]], dtype=np.float32)
    input_dict = {"name": "lgamma_4", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 5: Float64 array
    x = np.array([0.5, 1.0, 4.5], dtype=np.float64)
    input_dict = {"name": "lgamma_5", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 6: Half array
    x = np.array([0.5, 1.0, 4.5], dtype=np.float16)
    input_dict = {"name": "lgamma_6", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 7: Scalar tensor with negative value
    x = np.array(-4.0, dtype=np.float32)
    input_dict = {"name": "lgamma_7", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 8: Array with zero values
    x = np.array([0, 0.5], dtype=np.float32)
    input_dict = {"name": "lgamma_8", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 9: 1D array with mixed positive and negative values
    x = np.array([0.5, -4.0, 1.0], dtype=np.float32)
    input_dict = {"name": "lgamma_9", "x": x}
    list_of_inputs.append(input_dict)
    
    # Input 10: 1D array with float64 values
    x = np.array([0.5, 1.0, 4.5], dtype=np.float64)
    input_dict = {"name": "lgamma_10", "x": x}
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Lgamma"] = lgamma_inputs()

import numpy as np
import tensorflow as tf

def generate_logsoftmax_inputs():
    inputs_list = []
    
    # Input 1: 2D tensor with float32 dtype
    logits1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict1 = {"name": "logsoftmax1", "logits": logits1}
    inputs_list.append(input_dict1)
    
    # Input 2: 2D tensor with float64 dtype
    logits2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict2 = {"name": "logsoftmax2", "logits": logits2}
    inputs_list.append(input_dict2)
    
    # Input 3: 2D tensor with half dtype
    logits3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float16)
    input_dict3 = {"name": "logsoftmax3", "logits": logits3}
    inputs_list.append(input_dict3)
    
    # Input 4: 2D tensor with bfloat16 dtype
    logits4 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    input_dict4 = {"name": "logsoftmax4", "logits": logits4}
    inputs_list.append(input_dict4)
    
    # Input 5: 2D tensor with negative values
    logits5 = np.array([[-1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict5 = {"name": "logsoftmax5", "logits": logits5}
    inputs_list.append(input_dict5)
    
    # Input 6: 2D tensor with mixed values including negative
    logits6 = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    input_dict6 = {"name": "logsoftmax6", "logits": logits6}
    inputs_list.append(input_dict6)
    
    # Input 7: 2D tensor with large values
    logits7 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    input_dict7 = {"name": "logsoftmax7", "logits": logits7}
    inputs_list.append(input_dict7)
    
    # Input 8: 2D tensor with zero values
    logits8 = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    input_dict8 = {"name": "logsoftmax8", "logits": logits8}
    inputs_list.append(input_dict8)
    
    # Input 9: 2D tensor with very small values
    logits9 = np.array([[0.001, 0.002], [0.003, 0.004]], dtype=np.float32)
    input_dict9 = {"name": "logsoftmax9", "logits": logits9}
    inputs_list.append(input_dict9)
    
    # Input 10: 2D tensor with float32 dtype and different values
    logits10 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict10 = {"name": "logsoftmax10", "logits": logits10}
    inputs_list.append(input_dict10)

    return inputs_list

generated_inputs["tf.raw_ops.LogSoftmax"] = generate_logsoftmax_inputs()

import numpy as np
import tensorflow as tf
import copy

def generate_realdiv_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensors
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    input_dict = {
        "name": "test1",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 tensors
    x = np.array([10.5, 20.7], dtype=np.float64)
    y = np.array([2.5, 4.0], dtype=np.float64)
    input_dict = {
        "name": "test2",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32 tensors with negative values
    x = np.array([-1.0, -2.0], dtype=np.float32)
    y = np.array([2.0, -4.0], dtype=np.float32)
    input_dict = {
        "name": "test3",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: scalar float32 tensors
    x = np.array(1.0, dtype=np.float32)
    y = np.array(2.0, dtype=np.float32)
    input_dict = {
        "name": "test4",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 tensors with broadcasting
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([2.0], dtype=np.float64)
    input_dict = {
        "name": "test5",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: half tensors
    x = np.array([1.0, 2.0], dtype=np.float16)
    y = np.array([2.0, 4.0], dtype=np.float16)
    input_dict = {
        "name": "test6",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex64 tensors
    x = np.array([1+2j, 3+4j], dtype=np.complex64)
    y = np.array([2+1j, 2+2j], dtype=np.complex64)
    input_dict = {
        "name": "test7",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32 tensors with different dimensions
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    y = np.array([[[2.0, 4.0], [6.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]], dtype=np.float32)
    input_dict = {
        "name": "test8",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32 tensors with broadcasting
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    y = np.array([[[2.0, 4.0], [6.0, 8.0]]], dtype=np.float32)
    input_dict = {
        "name": "test9",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 tensors with various shapes
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    y = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    input_dict = {
        "name": "test10",
        "x": x,
        "y": y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.RealDiv"] = generate_realdiv_inputs()

import numpy as np
import tensorflow as tf

def generate_sparse_softmax_cross_entropy_with_logits_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with float32 features and int32 labels
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    labels = np.array([2, 1], dtype=np.int32)
    
    input_dict = {
        "name": "input_1",
        "features": features,
        "labels": labels
    }
    list_of_inputs.append(input_dict)
    
    # Input 2: Different shape with float64 features and int64 labels
    features = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float64)
    labels = np.array([3, 0], dtype=np.int64)
    
    input_dict = {
        "name": "input_2",
        "features": features,
        "labels": labels
    }
    list_of_inputs.append(input_dict)
    
    # Input 3: With negative values in features (float32)
    features = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)
    labels = np.array([0, 2], dtype=np.int32)
    
    input_dict = {
        "name": "input_3",
        "features": features,
        "labels": labels
    }
    list_of_inputs.append(input_dict)
    
    # Input 4: Single batch with float32 and int32
    features = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    labels = np.array([2], dtype=np.int32)
    
    input_dict = {
        "name": "input_4",
        "features": features,
        "labels": labels
    }
    list_of_inputs.append(input_dict)
    
    # Input 5: With zero values in features (float64)
    features = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float64)
    labels = np.array([0, 1], dtype=np.int64)
    
    input_dict = {
        "name": "input_5",
        "features": features,
        "labels": labels
    }
    list_of_inputs.append(input_dict)
    
    # Input 6: Mixed values (float32) with int64 labels
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    
    input_dict = {
        "name": "input_6",
        "features": features,
        "labels": labels
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits"] = generate_sparse_softmax_cross_entropy_with_logits_inputs()

import numpy as np
import tensorflow as tf

def generate_squared_difference_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensors
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    input_dict = {
        "name": "test1",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 2: float64 tensors
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    input_dict = {
        "name": "test2",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 3: int32 tensors
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[2, 3], [4, 5]], dtype=np.int32)
    input_dict = {
        "name": "test3",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 4: int64 tensors
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[2, 3], [4, 5]], dtype=np.int64)
    input_dict = {
        "name": "test4",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 5: complex64 tensors
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    y = np.array([[2+3j, 4+5j], [6+7j, 8+9j]], dtype=np.complex64)
    input_dict = {
        "name": "test5",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 6: complex128 tensors
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    y = np.array([[2+3j, 4+5j], [6+7j, 8+9j]], dtype=np.complex128)
    input_dict = {
        "name": "test6",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 7: scalar tensors
    x = np.array(5.0, dtype=np.float32)
    y = np.array(3.0, dtype=np.float32)
    input_dict = {
        "name": "test7",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 8: negative values
    x = np.array([-1.0, -2.0], dtype=np.float32)
    y = np.array([-3.0, -4.0], dtype=np.float32)
    input_dict = {
        "name": "test8",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 9: 1D tensors
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {
        "name": "test9",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    # Input 10: broadcasting compatible tensors
    x = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
    y = np.array([[2.0], [5.0]], dtype=np.float32)
    input_dict = {
        "name": "test10",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.raw_ops.SquaredDifference"] = generate_squared_difference_inputs()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_strided_slice_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    input_tensor = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])
    begin_tensor = tf.constant([1, 0, 0])
    end_tensor = tf.constant([2, 1, 3])
    stride_tensor = tf.constant([1, 1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var_tensor = None
    name = "test1"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input_tensor = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])
    begin_tensor = tf.constant([1, 0, 0])
    end_tensor = tf.constant([2, 2, 3])
    stride_tensor = tf.constant([1, 1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var_tensor = None
    name = "test2"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    input_tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    begin_tensor = tf.constant([0, 0])
    end_tensor = tf.constant([2, 3])
    stride_tensor = tf.constant([1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var_tensor = None
    name = "test3"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    input_tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    begin_tensor = tf.constant([0, 0])
    end_tensor = tf.constant([2, 2])
    stride_tensor = tf.constant([1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var_tensor = None
    name = "test4"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid - negative stride
    input_tensor = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])
    begin_tensor = tf.constant([1, -1, 0])
    end_tensor = tf.constant([2, -3, 3])
    stride_tensor = tf.constant([1, -1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var_tensor = None
    name = "test5"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid - with new axis mask
    input_tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    begin_tensor = tf.constant([0, 0])
    end_tensor = tf.constant([2, 2])
    stride_tensor = tf.constant([1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 1
    shrink_axis_mask = 0
    var_tensor = None
    name = "test6"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid - with shrink axis mask
    input_tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    begin_tensor = tf.constant([0, 0])
    end_tensor = tf.constant([2, 2])
    stride_tensor = tf.constant([1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 2
    var_tensor = None
    name = "test7"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid - with ellipsis mask
    input_tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    begin_tensor = tf.constant([0, 0, 0])
    end_tensor = tf.constant([2, 2, 2])
    stride_tensor = tf.constant([1, 1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 4
    new_axis_mask = 0
    shrink_axis_mask = 0
    var_tensor = None
    name = "test8"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid - different dimensionality
    input_tensor = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    begin_tensor = tf.constant([0, 0])
    end_tensor = tf.constant([2, 2])
    stride_tensor = tf.constant([1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var_tensor = None
    name = "test9"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid - negative values in begin and end
    input_tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    begin_tensor = tf.constant([-1, -1])
    end_tensor = tf.constant([3, 3])
    stride_tensor = tf.constant([1, 1])
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    var_tensor = None
    name = "test10"
    
    input_dict = {
        "input_": input_tensor,
        "begin": begin_tensor,
        "end": end_tensor,
        "strides": stride_tensor,
        "begin_mask": begin_mask,
        "end_mask": end_mask,
        "ellipsis_mask": ellipsis_mask,
        "new_axis_mask": new_axis_mask,
        "shrink_axis_mask": shrink_axis_mask,
        "var": var_tensor,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.strided_slice"] = tf_strided_slice_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_unicode_script_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 1D array
    input_tensor = np.array([65, 66, 67], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D array
    input_tensor = np.array([[65, 66], [67, 68]], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Single element array
    input_tensor = np.array([100], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Negative values
    input_tensor = np.array([-65, -66, -67], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Mixed values
    input_tensor = np.array([123, 456, 789], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Large numbers
    input_tensor = np.array([1000, 2000, 3000], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Empty array
    input_tensor = np.array([], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 3D array
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Negative and positive mixed
    input_tensor = np.array([100, -100, 200], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Large array
    input_tensor = np.array([97, 98, 99, 100, 101], dtype=np.int32)
    input_dict = {
        "input": input_tensor,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.strings.unicode_script"] = tf_strings_unicode_script_inputs()