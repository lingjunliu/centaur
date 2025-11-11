generated_inputs = {}
import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_autodiff_forwardaccumulator_inputs():
    list_of_inputs = []
    primals1 = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    tangents1 = tf.constant(np.array([[0.5], [0.2]], dtype=np.float32))
    x1 = tf.constant(1.0, dtype=np.float32)
    input_dict1 = {'primals': primals1, 'tangents': tangents1, 'x': x1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    primals2 = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    tangents2 = tf.constant(np.array([0.1, -0.2, 0.3], dtype=np.float64))
    x2 = tf.constant(2.0, dtype=np.float64)
    input_dict2 = {'primals': primals2, 'tangents': tangents2, 'x': x2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    primals3 = tf.constant(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.complex64))
    tangents3 = tf.constant(np.array([[0.1j], [-0.2j]], dtype=np.complex64))
    x3 = tf.constant(1.0 + 1.0j, dtype=np.complex64)
    input_dict3 = {'primals': primals3, 'tangents': tangents3, 'x': x3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    primals4 = tf.constant(np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    tangents4 = tf.constant(np.array([[1.0], [-1.0]], dtype=np.float32))
    x4 = tf.constant(-1.0, dtype=np.float32)
    input_dict4 = {'primals': primals4, 'tangents': tangents4, 'x': x4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    primals5 = tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32))
    tangents5 = tf.constant(np.array([[[0.5], [0.2]], [[0.1], [0.3]]], dtype=np.float32))
    x5 = tf.constant(2.0, dtype=np.float32)
    input_dict5 = {'primals': primals5, 'tangents': tangents5, 'x': x5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    primals6 = tf.constant(np.array([1.0, 2.0], dtype=np.float32))
    tangents6 = tf.constant(np.array([0.0, 1.0], dtype=np.float32))
    x6 = tf.constant(0.0, dtype=np.float32)
    input_dict6 = {'primals': primals6, 'tangents': tangents6, 'x': x6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    primals7 = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16))
    tangents7 = tf.constant(np.array([[0.5], [0.2]], dtype=np.float16))
    x7 = tf.constant(1.0, dtype=np.float16)
    input_dict7 = {'primals': primals7, 'tangents': tangents7, 'x': x7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    primals8 = tf.constant(np.array([[-2.0, -3.0], [-4.0, -5.0]], dtype=np.float32))
    tangents8 = tf.constant(np.array([[0.1], [-0.2]], dtype=np.float32))
    x8 = tf.constant(-1.0, dtype=np.float32)
    input_dict8 = {'primals': primals8, 'tangents': tangents8, 'x': x8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    primals9 = tf.constant(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    tangents9 = tf.constant(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    x9 = tf.constant(3.0, dtype=np.float32)
    input_dict9 = {'primals': primals9, 'tangents': tangents9, 'x': x9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    primals10 = tf.constant(np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32))
    tangents10 = tf.constant(np.array([[1.0], [0.0]], dtype=np.float32))
    x10 = tf.constant(0.0, dtype=np.float32)
    input_dict10 = {'primals': primals10, 'tangents': tangents10, 'x': x10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.autodiff.ForwardAccumulator"] = tf_autodiff_forwardaccumulator_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_array_inputs():
    list_of_inputs = []

    input_1 = {
        "val": np.array([1, 2, 3]),
        "dtype": np.int32,
        "copy": True,
        "ndmin": 1
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    input_2 = {
        "val": np.array([[1, 2], [3, 4]]),
        "dtype": np.float64,
        "copy": False,
        "ndmin": 2
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    input_3 = {
        "val": np.array([[[1, 2, 3], [4, 5, 6]]]),
        "dtype": np.int16,
        "copy": True,
        "ndmin": 3
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    input_4 = {
        "val": np.array([-1, -2, -3]),
        "dtype": np.int8,
        "copy": False,
        "ndmin": 1
    }
    list_of_inputs.append(copy.deepcopy(input_4))

    input_5 = {
        "val": np.array([1.1, 2.2, 3.3]),
        "dtype": np.float32,
        "copy": True,
        "ndmin": 1
    }
    list_of_inputs.append(copy.deepcopy(input_5))

    input_6 = {
        "val": np.array([[1, 2], [3, 4], [5, 6]]),
        "dtype": np.int64,
        "copy": False,
        "ndmin": 2
    }
    list_of_inputs.append(copy.deepcopy(input_6))

    input_7 = {
        "val": np.array([1, 2, 3, 4, 5]),
        "dtype": np.uint8,
        "copy": True,
        "ndmin": 1
    }
    list_of_inputs.append(copy.deepcopy(input_7))

    input_8 = {
        "val": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        "dtype": np.float16,
        "copy": False,
        "ndmin": 3
    }
    list_of_inputs.append(copy.deepcopy(input_8))

    input_9 = {
        "val": np.array([1, 2, 3]),
        "dtype": np.int32,
        "copy": True,
        "ndmin": 0
    }
    list_of_inputs.append(copy.deepcopy(input_9))

    input_10 = {
        "val": np.array([[1, 2, 3], [4, 5, 6]]),
        "dtype": np.complex64,
        "copy": False,
        "ndmin": 2
    }
    list_of_inputs.append(copy.deepcopy(input_10))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.array"] = tf_experimental_numpy_array_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_compress_inputs():
    list_of_inputs = []

    input_dict = {
        "condition": np.array([True, False, True]),
        "a": np.array([1, 2, 3]),
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "condition": np.array([False, True, False]),
        "a": np.array([[1, 2], [3, 4], [5, 6]]),
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "condition": np.array([True, False, True, False]),
        "a": np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.compress"] = tf_experimental_numpy_compress_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_expand_dims_inputs():
    list_of_inputs = []

    a = np.array([1, 2, 3])
    axis = 0
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3]])
    axis = 1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2], [3, 4]]])
    axis = 2
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1])
    axis = -1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2], [3, 4]])
    axis = -2
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1, 2, 3, 4, 5])
    axis = 0
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3], [4, 5, 6]])
    axis = 1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 0
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 2
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1, 2])
    axis = 1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.expand_dims"] = tf_experimental_numpy_expand_dims_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_identity_inputs():
    list_of_inputs = []
    
    n1 = np.int32(3)
    dtype1 = np.float32
    input_dict1 = {"n": n1, "dtype": dtype1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    n2 = np.int32(5)
    dtype2 = np.float64
    input_dict2 = {"n": n2, "dtype": dtype2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    n3 = np.int32(1)
    dtype3 = np.float16
    input_dict3 = {"n": n3, "dtype": dtype3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    n4 = np.int32(4)
    dtype4 = np.complex64
    input_dict4 = {"n": n4, "dtype": dtype4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    n5 = np.int32(2)
    dtype5 = np.complex128
    input_dict5 = {"n": n5, "dtype": dtype5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    n6 = np.int32(6)
    dtype6 = np.int32
    input_dict6 = {"n": n6, "dtype": dtype6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    n7 = np.int32(7)
    dtype7 = np.int64
    input_dict7 = {"n": n7, "dtype": dtype7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    n8 = np.int32(8)
    dtype8 = np.uint8
    input_dict8 = {"n": n8, "dtype": dtype8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    n9 = np.int32(9)
    dtype9 = np.uint16
    input_dict9 = {"n": n9, "dtype": dtype9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    n10 = np.int32(10)
    dtype10 = np.uint32
    input_dict10 = {"n": n10, "dtype": dtype10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.identity"] = tf_experimental_numpy_identity_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_logaddexp_inputs():
    list_of_inputs = []

    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([0.0, 1.0, 2.0])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([-1.0, -2.0, -3.0])
    x2 = np.array([-4.0, -5.0, -6.0])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([1.0, 2.0])
    x2 = np.array([3.0, 4.0])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    x2 = np.array([[0.0, 1.0], [2.0, 3.0]])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
    x2 = np.array([[[0.0], [1.0]], [[2.0], [3.0]]])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([1.0, -2.0, 3.0])
    x2 = np.array([-4.0, 5.0, -6.0])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([10.0, 20.0, 30.0])
    x2 = np.array([11.0, 21.0, 31.0])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([-10.0, -20.0, -30.0])
    x2 = np.array([-11.0, -21.0, -31.0])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([1.0])
    x2 = np.array([2.0])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x2 = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["tf.experimental.numpy.logaddexp"] = tf_experimental_numpy_logaddexp_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_reciprocal_inputs():
    list_of_inputs = []
    
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-1, -2, -3, -4, -5], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1, -2], [-3, -4]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.1, 1, 2, 3], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1, 2, 3, 0.4], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.reciprocal"] = tf_experimental_numpy_reciprocal_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_rot90_inputs():
    list_of_inputs = []

    m1 = np.array([[1, 2], [3, 4]])
    k1 = 1
    axes1 = (0, 1)
    input_dict1 = {"m": m1, "k": k1, "axes": axes1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    m2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k2 = 2
    axes2 = (0, 1)
    input_dict2 = {"m": m2, "k": k2, "axes": axes2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    m3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k3 = -1
    axes3 = (0, 1)
    input_dict3 = {"m": m3, "k": k3, "axes": axes3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    m4 = np.array([[1, 2], [3, 4], [5, 6]])
    k4 = 1
    axes4 = (0, 1)
    input_dict4 = {"m": m4, "k": k4, "axes": axes4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    m5 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    k5 = 3
    axes5 = (0, 2)
    input_dict5 = {"m": m5, "k": k5, "axes": axes5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    m6 = np.array([[1, 2], [3, 4]])
    k6 = 0
    axes6 = (0, 1)
    input_dict6 = {"m": m6, "k": k6, "axes": axes6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    m7 = np.array([[[1, 2], [3, 4]]])
    k7 = 2
    axes7 = (1, 2)
    input_dict7 = {"m": m7, "k": k7, "axes": axes7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    m8 = np.array([[[1, 2, 3], [4, 5, 6]]])
    k8 = 1
    axes8 = (0, 1)
    input_dict8 = {"m": m8, "k": k8, "axes": axes8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    m9 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    k9 = 1
    axes9 = (0, 1)
    input_dict9 = {"m": m9, "k": k9, "axes": axes9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    m10 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    k10 = 3
    axes10 = (0, 1)
    input_dict10 = {"m": m10, "k": k10, "axes": axes10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.rot90"] = tf_rot90_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_zeros_like_inputs():
    list_of_inputs = []

    a = np.array([[1, 2], [3, 4]])
    dtype = np.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    dtype = np.int64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1, 2, 3])
    dtype = np.complex64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    dtype = np.float64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dtype = np.int32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1])
    dtype = np.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[-1, -2], [-3, -4]])
    dtype = np.int64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1.1, 2.2, 3.3])
    dtype = np.float16
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2], [3, 4]]])
    dtype = np.complex128
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3, 4]])
    dtype = np.uint8
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.zeros_like"] = tf_experimental_numpy_zeros_like_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_adjust_gamma_inputs():
    list_of_inputs = []

    image1 = np.random.rand(2, 2, 3).astype(np.float32)
    gamma1 = 0.5
    gain1 = 1.2
    input_dict1 = {"image": image1, "gamma": gamma1, "gain": gain1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.random.rand(4, 4, 3).astype(np.float32)
    gamma2 = 2.0
    gain2 = 0.8
    input_dict2 = {"image": image2, "gamma": gamma2, "gain": gain2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.rand(1, 1, 3).astype(np.float32)
    gamma3 = 1.0
    gain3 = 1.0
    input_dict3 = {"image": image3, "gamma": gamma3, "gain": gain3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.random.rand(2, 3).astype(np.float32)
    gamma4 = 0.3
    gain4 = 1.5
    input_dict4 = {"image": image4, "gamma": gamma4, "gain": gain4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.random.rand(3, 2, 3).astype(np.float32)
    gamma5 = 1.5
    gain5 = 0.5
    input_dict5 = {"image": image5, "gamma": gamma5, "gain": gain5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = np.random.rand(5, 5, 3).astype(np.float32)
    gamma6 = 0.0
    gain6 = 2.0
    input_dict6 = {"image": image6, "gamma": gamma6, "gain": gain6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    image7 = np.random.rand(2, 2, 3).astype(np.float32)
    gamma7 = 2.5
    gain7 = 0.2
    input_dict7 = {"image": image7, "gamma": gamma7, "gain": gain7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    image8 = np.random.rand(1, 1, 1).astype(np.float32)
    gamma8 = 0.7
    gain8 = 1.8
    input_dict8 = {"image": image8, "gamma": gamma8, "gain": gain8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    image9 = np.random.rand(4, 3, 3).astype(np.float32)
    gamma9 = 1.1
    gain9 = 0.9
    input_dict9 = {"image": image9, "gamma": gamma9, "gain": gain9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    image10 = np.random.rand(2, 4).astype(np.float32)
    gamma10 = 0.9
    gain10 = 1.1
    input_dict10 = {"image": image10, "gamma": gamma10, "gain": gain10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.image.adjust_gamma"] = tf_image_adjust_gamma_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_flip_up_down_inputs():
    list_of_inputs = []

    image1 = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    input_dict = {'image': image1}
    list_of_inputs.append(input_dict)

    image2 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {'image': image2}
    list_of_inputs.append(input_dict)

    image3 = np.array([[[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]], dtype=np.float32)
    input_dict = {'image': image3}
    list_of_inputs.append(input_dict)

    image4 = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.int32)
    input_dict = {'image': image4}
    list_of_inputs.append(input_dict)

    image5 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    input_dict = {'image': image5}
    list_of_inputs.append(input_dict)

    image6 = np.random.rand(2, 3, 4, 1).astype(np.float32)
    input_dict = {'image': image6}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.image.flip_up_down"] = tf_image_flip_up_down_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_random_flip_left_right_inputs():
    list_of_inputs = []

    image1 = np.array([[[1], [2]], [[3], [4]]])
    seed1 = 5
    input_dict1 = {"image": image1, "seed": seed1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    seed2 = 6
    input_dict2 = {"image": image2, "seed": seed2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.array([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]])
    seed3 = 7
    input_dict3 = {"image": image3, "seed": seed3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
    seed4 = 8
    input_dict4 = {"image": image4, "seed": seed4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.array([[[1, 2, 3], [4, 5, 6]]])
    seed5 = 9
    input_dict5 = {"image": image5, "seed": seed5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]])
    seed6 = 10
    input_dict6 = {"image": image6, "seed": seed6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    image7 = np.array([[[[1], [2]], [[3], [4]]]])
    seed7 = -1
    input_dict7 = {"image": image7, "seed": seed7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    image8 = np.array([[[1, 2], [3, 4], [5, 6]]])
    seed8 = 0
    input_dict8 = {"image": image8, "seed": seed8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    image9 = np.array([[[[1, 2, 3]], [[4, 5, 6]]]])
    seed9 = 11
    input_dict9 = {"image": image9, "seed": seed9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    image10 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    seed10 = 12
    input_dict10 = {"image": image10, "seed": seed10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.image.random_flip_left_right"] = tf_image_random_flip_left_right_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_random_jpeg_quality_inputs():
    list_of_inputs = []

    image1 = np.random.randint(0, 256, size=(2, 2, 3), dtype=np.uint8)
    min_jpeg_quality1 = 75
    max_jpeg_quality1 = 95
    seed1 = 42
    input_dict1 = {
        "image": image1,
        "min_jpeg_quality": min_jpeg_quality1,
        "max_jpeg_quality": max_jpeg_quality1,
        "seed": seed1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.random.randint(0, 256, size=(4, 4, 3), dtype=np.uint8)
    min_jpeg_quality2 = 20
    max_jpeg_quality2 = 80
    seed2 = 100
    input_dict2 = {
        "image": image2,
        "min_jpeg_quality": min_jpeg_quality2,
        "max_jpeg_quality": max_jpeg_quality2,
        "seed": seed2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.randint(0, 256, size=(1, 5, 3), dtype=np.uint8)
    min_jpeg_quality3 = 1
    max_jpeg_quality3 = 100
    seed3 = 0
    input_dict3 = {
        "image": image3,
        "min_jpeg_quality": min_jpeg_quality3,
        "max_jpeg_quality": max_jpeg_quality3,
        "seed": seed3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.random.randint(0, 256, size=(3, 3, 3), dtype=np.uint8)
    min_jpeg_quality4 = 50
    max_jpeg_quality4 = 70
    seed4 = 123
    input_dict4 = {
        "image": image4,
        "min_jpeg_quality": min_jpeg_quality4,
        "max_jpeg_quality": max_jpeg_quality4,
        "seed": seed4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.random.randint(0, 256, size=(2, 4, 3), dtype=np.uint8)
    min_jpeg_quality5 = 90
    max_jpeg_quality5 = 99
    seed5 = 50
    input_dict5 = {
        "image": image5,
        "min_jpeg_quality": min_jpeg_quality5,
        "max_jpeg_quality": max_jpeg_quality5,
        "seed": seed5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs

generated_inputs["tf.image.random_jpeg_quality"] = tf_image_random_jpeg_quality_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_sobel_edges_inputs():
    list_of_inputs = []

    image1 = np.random.uniform(low=0.0, high=255.0, size=[1, 28, 28, 3])
    list_of_inputs.append({"image": tf.convert_to_tensor(image1)})

    image2 = np.random.uniform(low=0.0, high=255.0, size=[1, 50, 50, 1])
    list_of_inputs.append({"image": tf.convert_to_tensor(image2)})

    image3 = np.random.uniform(low=0.0, high=255.0, size=[1, 100, 100, 3])
    list_of_inputs.append({"image": tf.convert_to_tensor(image3)})

    image4 = np.random.uniform(low=0.0, high=255.0, size=[2, 32, 32, 1])
    list_of_inputs.append({"image": tf.convert_to_tensor(image4)})

    image5 = np.random.uniform(low=0.0, high=255.0, size=[4, 64, 64, 3])
    list_of_inputs.append({"image": tf.convert_to_tensor(image5)})

    image6 = np.random.uniform(low=0.0, high=100.0, size=[1, 10, 10, 1])
    list_of_inputs.append({"image": tf.convert_to_tensor(image6)})

    image7 = np.random.uniform(low=-50.0, high=-10.0, size=[1, 20, 20, 2])
    list_of_inputs.append({"image": tf.convert_to_tensor(image7)})

    image8 = np.random.uniform(low=0.0, high=255.0, size=[1, 28, 28, 3])
    list_of_inputs.append({"image": tf.convert_to_tensor(image8)})

    image9 = np.random.uniform(low=0.0, high=100.0, size=[1, 15, 15, 1])
    list_of_inputs.append({"image": tf.convert_to_tensor(image9)})

    return list_of_inputs

generated_inputs["tf.image.sobel_edges"] = tf_image_sobel_edges_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_stateless_random_contrast_inputs():
    list_of_inputs = []

    image1 = np.random.rand(2, 2, 3).astype(np.float32)
    seed1 = np.array([1, 2], dtype=np.int32)
    input_dict1 = {
        "image": image1,
        "lower": 0.2,
        "upper": 0.5,
        "seed": seed1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.random.rand(4, 4, 3).astype(np.float32)
    seed2 = np.array([3, 4], dtype=np.int32)
    input_dict2 = {
        "image": image2,
        "lower": 0.0,
        "upper": 0.1,
        "seed": seed2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    seed3 = np.array([5, 6], dtype=np.int32)
    input_dict3 = {
        "image": image3,
        "lower": 0.0,
        "upper": 1.0,
        "seed": seed3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.random.rand(2, 2, 2, 3).astype(np.float32)
    seed4 = np.array([7, 8], dtype=np.int32)
    input_dict4 = {
        "image": image4,
        "lower": 0.5,
        "upper": 1.0,
        "seed": seed4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    image5 = np.random.rand(5, 5, 3).astype(np.float32)
    seed5 = np.array([9, 10], dtype=np.int32)
    input_dict5 = {
        "image": image5,
        "lower": 0.0,
        "upper": 0.2,
        "seed": seed5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_contrast"] = tf_image_stateless_random_contrast_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_stateless_random_hue_inputs():
    list_of_inputs = []

    image1 = np.random.rand(2, 2, 3).astype(np.float32)
    max_delta1 = 0.1
    seed1 = np.array([1, 2], dtype=np.int32)
    input_dict1 = {"image": image1, "max_delta": max_delta1, "seed": seed1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.random.rand(4, 4, 3).astype(np.float32)
    max_delta2 = 0.5
    seed2 = np.array([3, 4], dtype=np.int64)
    input_dict2 = {"image": image2, "max_delta": max_delta2, "seed": seed2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.rand(1, 1, 3).astype(np.float32)
    max_delta3 = 0.2
    seed3 = np.array([5, 6], dtype=np.int32)
    input_dict3 = {"image": image3, "max_delta": max_delta3, "seed": seed3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.random.rand(3, 3, 3).astype(np.float32)
    max_delta4 = 0.05
    seed4 = np.array([7, 8], dtype=np.int64)
    input_dict4 = {"image": image4, "max_delta": max_delta4, "seed": seed4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.random.rand(2, 2, 3).astype(np.float32)
    max_delta5 = 0.3
    seed5 = np.array([9, 10], dtype=np.int32)
    input_dict5 = {"image": image5, "max_delta": max_delta5, "seed": seed5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = np.random.rand(5, 5, 3).astype(np.float32)
    max_delta6 = 0.4
    seed6 = np.array([11, 12], dtype=np.int64)
    input_dict6 = {"image": image6, "max_delta": max_delta6, "seed": seed6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    image7 = np.random.rand(1, 5, 3).astype(np.float32)
    max_delta7 = 0.15
    seed7 = np.array([13, 14], dtype=np.int32)
    input_dict7 = {"image": image7, "max_delta": max_delta7, "seed": seed7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    image8 = np.random.rand(5, 1, 3).astype(np.float32)
    max_delta8 = 0.25
    seed8 = np.array([15, 16], dtype=np.int64)
    input_dict8 = {"image": image8, "max_delta": max_delta8, "seed": seed8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    image9 = np.random.rand(3, 4, 3).astype(np.float32)
    max_delta9 = 0.35
    seed9 = np.array([17, 18], dtype=np.int32)
    input_dict9 = {"image": image9, "max_delta": max_delta9, "seed": seed9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    image10 = np.random.rand(4, 3, 3).astype(np.float32)
    max_delta10 = 0.45
    seed10 = np.array([19, 20], dtype=np.int64)
    input_dict10 = {"image": image10, "max_delta": max_delta10, "seed": seed10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_hue"] = tf_image_stateless_random_hue_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def linear_operator_circulant2d_inputs():
    list_of_inputs = []

    spectrum1 = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.complex64)
    input_dict1 = {
        'spectrum': spectrum1,
        'input_output_dtype': tf.complex64,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'circulant_2d_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    spectrum2 = np.array([[1. + 1j, 2. - 1j], [3. + 2j, 4. - 2j]], dtype=np.complex128)
    input_dict2 = {
        'spectrum': spectrum2,
        'input_output_dtype': tf.complex128,
        'is_non_singular': False,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'circulant_2d_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    spectrum3 = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=np.float32)
    input_dict3 = {
        'spectrum': spectrum3,
        'input_output_dtype': tf.float32,
        'is_non_singular': None,
        'is_self_adjoint': True,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'circulant_2d_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    spectrum4 = np.array([[1., 0.], [0., 1.]], dtype=np.complex64)
    input_dict4 = {
        'spectrum': spectrum4,
        'input_output_dtype': tf.complex64,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'circulant_2d_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorCirculant2D"] = linear_operator_circulant2d_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_linalg_LinearOperatorLowerTriangular_inputs():
    list_of_inputs = []

    input_dict1 = {
        'tril': np.array([[1., 2.], [3., 4.]]),
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'op1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input_dict2 = {
        'tril': np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                          [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]]),
        'is_non_singular': False,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'op2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input_dict3 = {
        'tril': np.array([[-1., 0.], [0., -2.]]),
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'op3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input_dict4 = {
        'tril': np.array([[[1., 0., 0.], [2., 3., 0.], [4., 5., 6.]],
                          [[7., 0., 0.], [8., 9., 0.], [10., 11., 12.]]]),
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'op4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input_dict5 = {
        'tril': np.random.rand(3, 3),
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'op5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input_dict6 = {
        'tril': np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'op6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input_dict7 = {
        'tril': np.array([[[1., 2., 3.], [0., 4., 5.], [0., 0., 6.]]]),
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'op7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input_dict8 = {
        'tril': np.array([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]]),
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'op8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input_dict9 = {
        'tril': np.array([[[1., -2.], [-3., 4.]]]),
        'is_non_singular': True,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'op9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input_dict10 = {
        'tril': np.array([[[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]]]),
        'is_non_singular': None,
        'is_self_adjoint': None,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'op10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorLowerTriangular"] = tf_linalg_LinearOperatorLowerTriangular_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_bessel_i0_inputs():
    list_of_inputs = []

    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_dict = {"x": x, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-2.0, -1.5, -1.0], dtype=np.float64)
    input_dict = {"x": x, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    input_dict = {"x": x, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-0.1, 0.1, -0.2, 0.2], dtype=np.float32)
    input_dict = {"x": x, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64)
    input_dict = {"x": x, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1.0, 1.0], [-2.0, 2.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0], dtype=np.float64)
    input_dict = {"x": x, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    input_dict = {"x": x, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.bessel_i0"] = tf_math_bessel_i0_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_floor_inputs():
    list_of_inputs = []

    x = np.array([1.3324, -1.5, 5.555, -2.532, 0.99, float("inf")])
    input_dict = {"x": x, "name": "floor_example_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-3.14, -2.71, -1.618, 0.0, 1.0, 2.718])
    input_dict = {"x": x, "name": "floor_example_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1.0, 2.5], [3.7, -4.2]])
    input_dict = {"x": x, "name": "floor_example_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]])
    input_dict = {"x": x, "name": "floor_example_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {"x": x, "name": "floor_example_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float64)
    input_dict = {"x": x, "name": "floor_example_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([float('nan'), float('inf'), -float('inf'), 0.0])
    input_dict = {"x": x, "name": "floor_example_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.999999, 2.000001, 3.5, -0.5], dtype=np.float16)
    input_dict = {"x": x, "name": "floor_example_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float32)
    input_dict = {"x": x, "name": "floor_example_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.000001, -0.000001, 0.0], dtype=np.float64)
    input_dict = {"x": x, "name": "floor_example_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.floor"] = tf_math_floor_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_in_top_k_inputs():
    list_of_inputs = []

    targets = np.array([0, 1, 3], dtype=np.int32)
    predictions = np.array([[1.2, -0.3, 2.8, 5.2], [0.1, 0.0, 0.0, 0.0], [0.0, 0.5, 0.3, 0.3]], dtype=np.float32)
    k = 2
    name = "test1"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([2, 0, 1], dtype=np.int64)
    predictions = np.array([[0.7, 0.9, 0.2], [0.1, 0.5, 0.8], [0.4, 0.6, 0.3]], dtype=np.float32)
    k = 1
    name = "test2"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([1, 2, 0], dtype=np.int32)
    predictions = np.array([[-0.1, 0.2, 0.3], [0.4, -0.5, 0.6], [0.7, 0.8, -0.9]], dtype=np.float32)
    k = 3
    name = "test3"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([0], dtype=np.int64)
    predictions = np.array([[1.0]], dtype=np.float32)
    k = 1
    name = "test4"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([1, 1, 1], dtype=np.int32)
    predictions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    k = 2
    name = "test5"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([0, 1], dtype=np.int64)
    predictions = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    k = 2
    name = "test6"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([0, 0, 0], dtype=np.int32)
    predictions = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    k = 1
    name = "test7"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([2, 1, 0], dtype=np.int64)
    predictions = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]], dtype=np.float32)
    k = 4
    name = "test8"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([0], dtype=np.int32)
    predictions = np.array([[-1.0, 2.0, 3.0]], dtype=np.float32)
    k = 2
    name = "test9"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([1, 2, 3], dtype=np.int64)
    predictions = np.array([[0.1, 0.9, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7], [0.8, 0.2, 0.1, 0.9]], dtype=np.float32)
    k = 3
    name = "test10"
    input_dict = {
        "targets": targets,
        "predictions": predictions,
        "k": k,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.in_top_k"] = tf_math_in_top_k_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_nextafter_inputs():
    list_of_inputs = []

    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([1.1, 2.2, 2.9], dtype=np.float32)
    name = "example1"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    x2 = np.array([-0.9, -2.1, -2.9], dtype=np.float64)
    name = "example2"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([1.0], dtype=np.float32)
    x2 = np.array([1.000001], dtype=np.float32)
    name = "example3"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x2 = np.array([[1.1, 2.1], [3.1, 4.1]], dtype=np.float64)
    name = "example4"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([np.finfo(np.float32).smallest_normal], dtype=np.float32)
    x2 = np.array([np.finfo(np.float32).smallest_normal * 2], dtype=np.float32)
    name = "example5"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([np.finfo(np.float64).smallest_normal], dtype=np.float64)
    x2 = np.array([np.finfo(np.float64).smallest_normal * 2], dtype=np.float64)
    name = "example6"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([0.0], dtype=np.float32)
    x2 = np.array([1e-38], dtype=np.float32)
    name = "example7"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([1e38], dtype=np.float32)
    x2 = np.array([np.finfo(np.float32).max], dtype=np.float32)
    name = "example8"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([-1e38], dtype=np.float64)
    x2 = np.array([-np.finfo(np.float64).max], dtype=np.float64)
    name = "example9"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x2 = np.array([0.9, 1.9, 2.9], dtype=np.float64)
    name = "example10"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.nextafter"] = tf_math_nextafter_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def fresnel_cos_inputs():
    list_of_inputs = []

    x = np.array([-1.0, -0.1, 0.1, 1.0], dtype=np.float32)
    input_dict = {"x": x, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0], dtype=np.float64)
    input_dict = {"x": x, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.5, 2.0, -0.5], dtype=np.float32)
    input_dict = {"x": x, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1.0, 0.5], [2.0, -1.5]], dtype=np.float64)
    input_dict = {"x": x, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0.1, -0.2], [0.3, 0.4]], [[-0.5, 0.6], [0.7, -0.8]]], dtype=np.float32)
    input_dict = {"x": x, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10.0], dtype=np.float64)
    input_dict = {"x": x, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-10.0], dtype=np.float32)
    input_dict = {"x": x, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
    input_dict = {"x": x, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-2.5], [3.7]], dtype=np.float32)
    input_dict = {"x": x, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0.5, -1.5]], [[2.0, -2.0]]], dtype=np.float64)
    input_dict = {"x": x, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.special.fresnel_cos"] = fresnel_cos_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_unsorted_segment_max_inputs():
    list_of_inputs = []

    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], dtype=np.int32)
    segment_ids1 = np.array([0, 1, 0], dtype=np.int32)
    num_segments1 = 2
    name1 = "test1"
    input_dict1 = {
        "data": data1,
        "segment_ids": segment_ids1,
        "num_segments": num_segments1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    data2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    segment_ids2 = np.array([0, 1, 0], dtype=np.int32)
    num_segments2 = 2
    name2 = "test2"
    input_dict2 = {
        "data": data2,
        "segment_ids": segment_ids2,
        "num_segments": num_segments2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    data3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    segment_ids3 = np.array([0, 1, 0], dtype=np.int64)
    num_segments3 = 2
    name3 = "test3"
    input_dict3 = {
        "data": data3,
        "segment_ids": segment_ids3,
        "num_segments": num_segments3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    data4 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    segment_ids4 = np.array([0, 1, 0, 1, 0], dtype=np.int32)
    num_segments4 = 2
    name4 = "test4"
    input_dict4 = {
        "data": data4,
        "segment_ids": segment_ids4,
        "num_segments": num_segments4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    data5 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    segment_ids5 = np.array([0, 1], dtype=np.int32)
    num_segments5 = 2
    name5 = "test5"
    input_dict5 = {
        "data": data5,
        "segment_ids": segment_ids5,
        "num_segments": num_segments5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    data6 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    segment_ids6 = np.array([0, 1], dtype=np.int32)
    num_segments6 = 2
    name6 = "test6"
    input_dict6 = {
        "data": data6,
        "segment_ids": segment_ids6,
        "num_segments": num_segments6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["tf.math.unsorted_segment_max"] = tf_math_unsorted_segment_max_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_compute_accidental_hits_inputs():
    list_of_inputs = []

    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([2, 5, 1, 6], dtype=np.int64)
    num_true = 2
    seed = 42
    name = "test1"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    sampled_candidates = np.array([2, 4, 6, 8, 10], dtype=np.int64)
    num_true = 3
    seed = 123
    name = "test2"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1], [2]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3], dtype=np.int64)
    num_true = 1
    seed = 0
    name = "test3"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([5, 6, 7, 8], dtype=np.int64)
    num_true = 2
    seed = 99
    name = "test4"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, -2], [3, -4]], dtype=np.int64)
    sampled_candidates = np.array([-2, 5, 1, 6], dtype=np.int64)
    num_true = 2
    seed = 10
    name = "test5"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 2, 3, 4]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    num_true = 4
    seed = 50
    name = "test6"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([1, 2], dtype=np.int64)
    num_true = 2
    seed = 77
    name = "test7"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([5, 6, 7, 8], dtype=np.int64)
    num_true = 2
    seed = 1
    name = "test8"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 2, 3]], dtype=np.int64)
    sampled_candidates = np.array([3, 4, 5], dtype=np.int64)
    num_true = 3
    seed = 33
    name = "test9"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([1, 3], dtype=np.int64)
    num_true = 2
    seed = 88
    name = "test10"
    input_dict = {
        'true_classes': true_classes,
        'sampled_candidates': sampled_candidates,
        'num_true': num_true,
        'seed': seed,
        'name': name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.compute_accidental_hits"] = tf_nn_compute_accidental_hits_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_ctc_beam_search_decoder_inputs():
    list_of_inputs = []

    inputs1 = np.random.rand(10, 4, 5).astype(np.float32)
    sequence_length1 = np.array([4, 4, 4, 4], dtype=np.int32)
    beam_width1 = 10
    top_paths1 = 2

    input_dict1 = {
        "inputs": inputs1,
        "sequence_length": sequence_length1,
        "beam_width": beam_width1,
        "top_paths": top_paths1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["tf.nn.ctc_beam_search_decoder"] = tf_nn_ctc_beam_search_decoder_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_softsign_inputs():
    list_of_inputs = []

    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "softsign_test_1"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    name = "softsign_test_2"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = "softsign_test_3"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([1.0], dtype=np.float64)
    name = "softsign_test_4"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    name = "softsign_test_5"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    name = "softsign_test_6"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([1.5, -2.5, 3.5], dtype=np.float16)
    name = "softsign_test_7"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    name = "softsign_test_8"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([100.0, -100.0, 0.0], dtype=np.float32)
    name = "softsign_test_9"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    name = "softsign_test_10"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.softsign"] = tf_nn_softsign_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_quantization_fake_quant_with_min_max_vars_per_channel_gradient_inputs():
    list_of_inputs = []

    gradients = np.random.rand(4, 3, 2, 5).astype(np.float32)
    inputs = np.random.rand(4, 3, 2, 5).astype(np.float32)
    min_val = np.random.rand(5).astype(np.float32)
    max_val = np.random.rand(5).astype(np.float32)
    num_bits = 8
    narrow_range = False
    name = "fake_quant_test_1"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    gradients = np.random.rand(2, 7).astype(np.float32)
    inputs = np.random.rand(2, 7).astype(np.float32)
    min_val = np.random.rand(7).astype(np.float32)
    max_val = np.random.rand(7).astype(np.float32)
    num_bits = 4
    narrow_range = True
    name = "fake_quant_test_2"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    gradients = np.random.rand(3).astype(np.float32)
    inputs = np.random.rand(3).astype(np.float32)
    min_val = np.random.rand(3).astype(np.float32)
    max_val = np.random.rand(3).astype(np.float32)
    num_bits = 12
    narrow_range = False
    name = "fake_quant_test_3"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    gradients = np.random.rand(5, 4).astype(np.float32)
    inputs = np.random.rand(5, 4).astype(np.float32)
    min_val = np.random.rand(4).astype(np.float32)
    max_val = np.random.rand(4).astype(np.float32)
    num_bits = 2
    narrow_range = True
    name = "fake_quant_test_4"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    gradients = np.random.rand(10, 1, 1, 1).astype(np.float32)
    inputs = np.random.rand(10, 1, 1, 1).astype(np.float32)
    min_val = np.random.rand(1).astype(np.float32)
    max_val = np.random.rand(1).astype(np.float32)
    num_bits = 16
    narrow_range = False
    name = "fake_quant_test_5"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    gradients = np.random.rand(1, 2, 3, 4).astype(np.float32)
    inputs = np.random.rand(1, 2, 3, 4).astype(np.float32)
    min_val = np.random.rand(4).astype(np.float32)
    max_val = np.random.rand(4).astype(np.float32)
    num_bits = 6
    narrow_range = True
    name = "fake_quant_test_6"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    gradients = np.random.rand(6, 2).astype(np.float32)
    inputs = np.random.rand(6, 2).astype(np.float32)
    min_val = np.random.rand(2).astype(np.float32)
    max_val = np.random.rand(2).astype(np.float32)
    num_bits = 8
    narrow_range = False
    name = "fake_quant_test_7"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    gradients = np.random.rand(7, 8, 9).astype(np.float32)
    inputs = np.random.rand(7, 8, 9).astype(np.float32)
    min_val = np.random.rand(9).astype(np.float32)
    max_val = np.random.rand(9).astype(np.float32)
    num_bits = 10
    narrow_range = True
    name = "fake_quant_test_8"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    gradients = np.random.rand(2, 3).astype(np.float32)
    inputs = np.random.rand(2, 3).astype(np.float32)
    min_val = np.random.rand(3).astype(np.float32)
    max_val = np.random.rand(3).astype(np.float32)
    num_bits = 14
    narrow_range = False
    name = "fake_quant_test_9"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    gradients = np.random.rand(8).astype(np.float32)
    inputs = np.random.rand(8).astype(np.float32)
    min_val = np.random.rand(8).astype(np.float32)
    max_val = np.random.rand(8).astype(np.float32)
    num_bits = 5
    narrow_range = True
    name = "fake_quant_test_10"

    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient"] = tf_quantization_fake_quant_with_min_max_vars_per_channel_gradient_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_bucketize_inputs():
    list_of_inputs = []

    input1 = np.array([-5, 10000], dtype=np.int32)
    boundaries1 = [0, 10, 100]
    input_dict1 = {'name': 'bucketize1', 'input': input1, 'boundaries': boundaries1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([150, 10], dtype=np.int64)
    boundaries2 = [0, 10, 100, 200]
    input_dict2 = {'name': 'bucketize2', 'input': input2, 'boundaries': boundaries2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([5, 100], dtype=np.float32)
    boundaries3 = [0, 10, 100]
    input_dict3 = {'name': 'bucketize3', 'input': input3, 'boundaries': boundaries3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([-1.5, 2.7, 5.1], dtype=np.float64)
    boundaries4 = [-2, 0, 3, 6]
    input_dict4 = {'name': 'bucketize4', 'input': input4, 'boundaries': boundaries4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    boundaries5 = [0, 2, 4]
    input_dict5 = {'name': 'bucketize5', 'input': input5, 'boundaries': boundaries5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    boundaries6 = [0, 3, 6]
    input_dict6 = {'name': 'bucketize6', 'input': input6, 'boundaries': boundaries6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([10.5, 20.2, 30.8], dtype=np.float32)
    boundaries7 = [5.0, 15.0, 25.0, 35.0]
    input_dict7 = {'name': 'bucketize7', 'input': input7, 'boundaries': boundaries7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    boundaries8 = [-8, -4, 0, 4, 8]
    input_dict8 = {'name': 'bucketize8', 'input': input8, 'boundaries': boundaries8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[1.1, 2.2], [-3.3, 4.4]], dtype=np.float64)
    boundaries9 = [-2.0, 0.0, 2.0]
    input_dict9 = {'name': 'bucketize9', 'input': input9, 'boundaries': boundaries9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([100, 200, 300], dtype=np.int64)
    boundaries10 = [50, 150, 250]
    input_dict10 = {'name': 'bucketize10', 'input': input10, 'boundaries': boundaries10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.raw_ops.Bucketize"] = tf_raw_ops_bucketize_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_cross_inputs():
    list_of_inputs = []

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[2.0, 4.0, -6.0], [5.0, 7.0, 9.0]], dtype=np.float32)
    input_dict = {'name': 'cross_1', 'a': a, 'b': b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float64)
    b = np.array([[2.0, -4.0, 6.0], [-5.0, 7.0, -9.0]], dtype=np.float64)
    input_dict = {'name': 'cross_2', 'a': a, 'b': b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    b = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int32)
    input_dict = {'name': 'cross_3', 'a': a, 'b': b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    b = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int16)
    input_dict = {'name': 'cross_5', 'a': a, 'b': b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    b = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int8)
    input_dict = {'name': 'cross_6', 'a': a, 'b': b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    b = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int64)
    input_dict = {'name': 'cross_7', 'a': a, 'b': b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
    b = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.float16)
    input_dict = {'name': 'cross_8', 'a': a, 'b': b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Cross"] = tf_raw_ops_cross_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_ensure_shape_inputs():
    list_of_inputs = []

    input1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    shape1 = [2, 2]
    name1 = "ensure_shape_1"
    input_dict1 = {"name": name1, "input": input1, "shape": shape1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    shape2 = [5]
    name2 = "ensure_shape_2"
    input_dict2 = {"name": name2, "input": input2, "shape": shape2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int64)
    shape3 = [2, 2, 3]
    name3 = "ensure_shape_3"
    input_dict3 = {"name": name3, "input": input3, "shape": shape3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    shape4 = [2, 2]
    name4 = "ensure_shape_4"
    input_dict4 = {"name": name4, "input": input4, "shape": shape4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1], dtype=np.int32)
    shape5 = [1]
    name5 = "ensure_shape_5"
    input_dict5 = {"name": name5, "input": input5, "shape": shape5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1, 2, 3], dtype=np.float64)
    shape6 = [3]
    name6 = "ensure_shape_6"
    input_dict6 = {"name": name6, "input": input6, "shape": shape6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    shape7 = [2, 2, 2]
    name7 = "ensure_shape_7"
    input_dict7 = {"name": name7, "input": input7, "shape": shape7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1, 2, 3, 4], dtype=np.int16)
    shape8 = [4]
    name8 = "ensure_shape_8"
    input_dict8 = {"name": name8, "input": input8, "shape": shape8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
    shape9 = [2,3]
    name9 = "ensure_shape_9"
    input_dict9 = {"name": name9, "input": input9, "shape": shape9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([10], dtype=np.int8)
    shape10 = [1]
    name10 = "ensure_shape_10"
    input_dict10 = {"name": name10, "input": input10, "shape": shape10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.raw_ops.EnsureShape"] = tf_raw_ops_ensure_shape_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_erf_inputs():
    list_of_inputs = []

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {'name': 'erf_test_1', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {'name': 'erf_test_2', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {'name': 'erf_test_3', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    input_dict = {'name': 'erf_test_4', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0], dtype=np.float32)
    input_dict = {'name': 'erf_test_5', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.5, -2.5, 3.5], dtype=np.half)
    input_dict = {'name': 'erf_test_6', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {'name': 'erf_test_7', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float64)
    input_dict = {'name': 'erf_test_8', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-6, -1e-6], dtype=np.float32)
    input_dict = {'name': 'erf_test_9', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e6, -1e6], dtype=np.float64)
    input_dict = {'name': 'erf_test_10', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Erf"] = tf_raw_ops_erf_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_FloorMod_inputs():
    list_of_inputs = []

    x = np.array([10, 20, 30], dtype=np.int32)
    y = np.array([3, 7, 2], dtype=np.int32)
    input_dict = {'name': 'floor_mod_1', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-10, -20, -30], dtype=np.int32)
    y = np.array([3, 7, 2], dtype=np.int32)
    input_dict = {'name': 'floor_mod_2', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10.5, 20.2, 30.9], dtype=np.float32)
    y = np.array([3.1, 7.5, 2.0], dtype=np.float32)
    input_dict = {'name': 'floor_mod_3', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-10, 20], [30, -40]], dtype=np.int64)
    y = np.array([3, 7], dtype=np.int64)
    input_dict = {'name': 'floor_mod_4', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([2, 3, 4], dtype=np.int8)
    input_dict = {'name': 'floor_mod_5', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([100, 200, 300], dtype=np.float64)
    y = np.array([3.5, 7.1, 2.8], dtype=np.float64)
    input_dict = {'name': 'floor_mod_6', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 20, 30], dtype=np.int16)
    y = np.array([3, 7, 2], dtype=np.int16)
    input_dict = {'name': 'floor_mod_7', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([2.0, 3.0, 4.0], dtype=np.float16)
    input_dict = {'name': 'floor_mod_8', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FloorMod"] = tf_raw_ops_FloorMod_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_FractionalMaxPool_inputs():
    list_of_inputs = []

    input_1 = {
        'value': np.random.rand(1, 8, 8, 3).astype(np.float32),
        'pooling_ratio': [1.0, 1.414, 1.732, 1.0],
        'pseudo_random': True,
        'overlapping': False,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': 'fractional_max_pool_1'
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    input_2 = {
        'value': np.random.randint(-10, 10, size=(1, 16, 16, 2)).astype(np.int32),
        'pooling_ratio': [1.0, 1.5, 1.5, 1.0],
        'pseudo_random': False,
        'overlapping': True,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': 'fractional_max_pool_2'
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    input_3 = {
        'value': np.random.rand(2, 32, 32, 4).astype(np.float64),
        'pooling_ratio': [1.0, 2.0, 2.0, 1.0],
        'pseudo_random': True,
        'overlapping': False,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': 'fractional_max_pool_3'
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    input_4 = {
        'value': np.random.randint(0, 255, size=(1, 64, 64, 1)).astype(np.int64),
        'pooling_ratio': [1.0, 1.2, 1.2, 1.0],
        'pseudo_random': False,
        'overlapping': True,
        'deterministic': False,
        'seed': 0,
        'seed2': 0,
        'name': 'fractional_max_pool_4'
    }
    list_of_inputs.append(copy.deepcopy(input_4))

    return list_of_inputs

generated_inputs["tf.raw_ops.FractionalMaxPool"] = tf_raw_ops_FractionalMaxPool_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_imag_inputs():
    list_of_inputs = []

    input1 = np.array([-2.25 + 4.75j, 3.25 + 5.75j], dtype=np.complex64)
    input_dict1 = {'input': input1, 'Tout': np.float32, 'name': 'imag1'}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)
    input_dict2 = {'input': input2, 'Tout': np.float64, 'name': 'imag2'}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex64)
    input_dict3 = {'input': input3, 'Tout': np.float32, 'name': 'imag3'}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-1 - 1j], [0 + 0j]], dtype=np.complex128)
    input_dict4 = {'input': input4, 'Tout': np.float64, 'name': 'imag4'}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1j, 2j, 3j], dtype=np.complex64)
    input_dict5 = {'input': input5, 'Tout': np.float32, 'name': 'imag5'}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1 + 0j, 2 + 0j, 3 + 0j], dtype=np.complex128)
    input_dict6 = {'input': input6, 'Tout': np.float64, 'name': 'imag6'}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[-1.5 + 2.5j], [3.5 - 4.5j]], dtype=np.complex64)
    input_dict7 = {'input': input7, 'Tout': np.float32, 'name': 'imag7'}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[[1 + 1j, 2 + 2j]], [[3 + 3j, 4 + 4j]]], dtype=np.complex128)
    input_dict8 = {'input': input8, 'Tout': np.float64, 'name': 'imag8'}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([-1 + 1j, -2 + 2j], dtype=np.complex64)
    input_dict9 = {'input': input9, 'Tout': np.float32, 'name': 'imag9'}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([1.1 + 2.2j, 3.3 + 4.4j, 5.5 + 6.6j], dtype=np.complex128)
    input_dict10 = {'input': input10, 'Tout': np.float64, 'name': 'imag10'}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.raw_ops.Imag"] = tf_raw_ops_imag_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_LRN_inputs():
    list_of_inputs = []

    input_1 = np.random.rand(2, 4, 4, 3).astype(np.float32)
    input_dict_1 = {
        "input": input_1,
        "depth_radius": 5,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "LRN_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = np.random.rand(1, 3, 3, 2).astype(np.float32)
    input_dict_2 = {
        "input": input_2,
        "depth_radius": 3,
        "bias": 2.0,
        "alpha": 0.5,
        "beta": 1.0,
        "name": "LRN_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = np.random.rand(4, 2, 2, 1).astype(np.float32)
    input_dict_3 = {
        "input": input_3,
        "depth_radius": 1,
        "bias": 0.5,
        "alpha": 2.0,
        "beta": 0.2,
        "name": "LRN_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_4 = np.random.rand(2, 2, 2, 4).astype(np.float32)
    input_dict_4 = {
        "input": input_4,
        "depth_radius": 2,
        "bias": 1.5,
        "alpha": 0.8,
        "beta": 0.7,
        "name": "LRN_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_5 = np.random.rand(3, 3, 3, 5).astype(np.float32)
    input_dict_5 = {
        "input": input_5,
        "depth_radius": 4,
        "bias": 0.1,
        "alpha": 1.2,
        "beta": 0.3,
        "name": "LRN_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    input_6 = np.random.rand(1, 1, 1, 1).astype(np.float32)
    input_dict_6 = {
        "input": input_6,
        "depth_radius": 0,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "LRN_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    input_7 = np.random.rand(2, 2, 2, 2).astype(np.float32)
    input_dict_7 = {
        "input": input_7,
        "depth_radius": 1,
        "bias": -1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "LRN_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    input_8 = np.random.rand(4, 4, 4, 3).astype(np.float32)
    input_dict_8 = {
        "input": input_8,
        "depth_radius": 5,
        "bias": -0.5,
        "alpha": -0.2,
        "beta": 1.0,
        "name": "LRN_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    input_9 = np.random.rand(1, 2, 2, 3).astype(np.float32)
    input_dict_9 = {
        "input": input_9,
        "depth_radius": 1,
        "bias": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "name": "LRN_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    input_10 = np.random.rand(2, 3, 3, 4).astype(np.float32)
    input_dict_10 = {
        "input": input_10,
        "depth_radius": 2,
        "bias": 3.0,
        "alpha": 1.5,
        "beta": 0.8,
        "name": "LRN_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.LRN"] = tf_raw_ops_LRN_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_lgamma_inputs():
    list_of_inputs = []

    x = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {'name': 'lgamma_test_1', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -0.5, 0.0, 1.0, 2.0], dtype=np.float64)
    input_dict = {'name': 'lgamma_test_2', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {'name': 'lgamma_test_3', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict = {'name': 'lgamma_test_4', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.5, 2.5, 3.5], dtype=np.float16)
    input_dict = {'name': 'lgamma_test_5', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    input_dict = {'name': 'lgamma_test_6', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32)
    input_dict = {'name': 'lgamma_test_7', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([5.5, 6.5, 7.5], dtype=np.float64)
    input_dict = {'name': 'lgamma_test_8', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    input_dict = {'name': 'lgamma_test_9', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {'name': 'lgamma_test_10', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Lgamma"] = tf_raw_ops_lgamma_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_logsoftmax_inputs():
    list_of_inputs = []

    logits = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    name = "log_softmax_1"
    input_dict = {"name": name, "logits": logits}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]], dtype=np.float64)
    name = "log_softmax_2"
    input_dict = {"name": name, "logits": logits}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[0.5, -0.5, 0.0], [1.5, -1.5, 0.5]], dtype=np.float16)
    name = "log_softmax_3"
    input_dict = {"name": name, "logits": logits}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[10.0, -10.0, 5.0], [-5.0, 15.0, 0.0]], dtype=np.float32)
    name = "log_softmax_5"
    input_dict = {"name": name, "logits": logits}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    name = "log_softmax_6"
    input_dict = {"name": name, "logits": logits}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    name = "log_softmax_7"
    input_dict = {"name": name, "logits": logits}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float16)
    name = "log_softmax_8"
    input_dict = {"name": name, "logits": logits}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    name = "log_softmax_9"
    input_dict = {"name": name, "logits": logits}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    name = "log_softmax_10"
    input_dict = {"name": name, "logits": logits}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.LogSoftmax"] = tf_raw_ops_logsoftmax_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_random_uniform_inputs():
    list_of_inputs = []

    input_dict = {
        "shape": np.array([2, 3], dtype=np.int32),
        "dtype": np.float32,
        "seed": 10,
        "seed2": 20,
        "name": "test_uniform_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "shape": np.array([4], dtype=np.int64),
        "dtype": np.float64,
        "seed": 123,
        "seed2": 456,
        "name": "test_uniform_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.RandomUniform"] = tf_raw_ops_random_uniform_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_realdiv_inputs():
    list_of_inputs = []

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    input_dict = {'name': 'div1', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0], dtype=np.float64)
    y = np.array([10.0], dtype=np.float64)
    input_dict = {'name': 'div2', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.RealDiv"] = tf_raw_ops_realdiv_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def sparse_softmax_cross_entropy_with_logits_inputs():
    list_of_inputs = []

    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int32)
    input_dict = {'name': 'op1', 'features': features, 'labels': labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-1.0, 0.5, 1.2], [2.3, -0.8, 0.1]], dtype=np.float64)
    labels = np.array([2, 0], dtype=np.int64)
    input_dict = {'name': 'op2', 'features': features, 'labels': labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
    labels = np.array([1, 3], dtype=np.int32)
    input_dict = {'name': 'op3', 'features': features, 'labels': labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float64)
    labels = np.array([1, 2], dtype=np.int32)
    input_dict = {'name': 'op5', 'features': features, 'labels': labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    labels = np.array([0, 0], dtype=np.int64)
    input_dict = {'name': 'op6', 'features': features, 'labels': labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float32)
    labels = np.array([2, 1], dtype=np.int32)
    input_dict = {'name': 'op7', 'features': features, 'labels': labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    labels = np.array([3], dtype=np.int32)
    input_dict = {'name': 'op9', 'features': features, 'labels': labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    labels = np.array([0, 1, 0], dtype=np.int64)
    input_dict = {'name': 'op10', 'features': features, 'labels': labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits"] = sparse_softmax_cross_entropy_with_logits_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_SquaredDifference_inputs():
    list_of_inputs = []

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[2.0, 1.0], [4.0, 3.0]], dtype=np.float32)
    input_dict = {'name': 'op1', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([2, 3, 4], dtype=np.int32)
    input_dict = {'name': 'op2', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0], dtype=np.float64)
    y = np.array([-3.0, -4.0], dtype=np.float64)
    input_dict = {'name': 'op3', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    y = np.array([[2+2j, 1+1j], [4+4j, 3+3j]], dtype=np.complex64)
    input_dict = {'name': 'op4', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0], dtype=np.float32)
    y = np.array([2.0], dtype=np.float32)
    input_dict = {'name': 'op5', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {'name': 'op6', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[2.0, 1.0], [4.0, 3.0]], dtype=np.float32)
    input_dict = {'name': 'op7', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2], dtype=np.int64)
    y = np.array([3, 4], dtype=np.int64)
    input_dict = {'name': 'op8', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.5, 2.5], dtype=np.float64)
    y = np.array([3.5, 4.5], dtype=np.float64)
    input_dict = {'name': 'op9', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.SquaredDifference"] = tf_raw_ops_SquaredDifference_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sets_intersection_inputs():
    list_of_inputs = []

    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[2, 4, -6], [5, 7, 9]])
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    b = np.array([[[3, 5, 7], [8, 10, 12]], [[1, 2, 4], [6, 9, 11]]])
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1], [2]], [[3], [4]]])
    b = np.array([[[1, 5], [2, 6]], [[3, 7], [4, 8]]])
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2, 3]]])
    b = np.array([[[1, 2]]])
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[[1, -2, 3]]])
    b = np.array([[[1, 2]]])
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[[1, 2], [3, 4]]])
    b = np.array([[[1, 2], [3, 4]]])
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sets.intersection"] = tf_sets_intersection_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_eye_inputs():
    list_of_inputs = []

    input_dict = {
        "num_rows": np.int32(5),
        "num_columns": np.int32(5),
        "dtype": np.float32,
        "name": "eye_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(3),
        "num_columns": np.int32(4),
        "dtype": np.float64,
        "name": "eye_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(1),
        "num_columns": np.int32(1),
        "dtype": np.complex64,
        "name": "eye_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(0),
        "num_columns": np.int32(0),
        "dtype": np.float32,
        "name": "eye_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(2),
        "num_columns": np.int32(2),
        "dtype": np.int32,
        "name": "eye_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(6),
        "num_columns": np.int32(3),
        "dtype": np.bool_,
        "name": "eye_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(4),
        "num_columns": np.int32(4),
        "dtype": np.float16,
        "name": "eye_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(7),
        "num_columns": np.int32(7),
        "dtype": np.float32,
        "name": "eye_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(10),
        "num_columns": np.int32(5),
        "dtype": np.float64,
        "name": "eye_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.int32(2),
        "num_columns": np.int32(3),
        "dtype": np.complex128,
        "name": "eye_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.eye"] = tf_sparse_eye_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_strided_slice_inputs():
    list_of_inputs = []
    input_1 = tf.constant(np.arange(10).reshape(2, 5), dtype=tf.int32)
    begin_1 = np.array([0, 0], dtype=np.int32)
    end_1 = np.array([2, 3], dtype=np.int32)
    strides_1 = np.array([1, 1], dtype=np.int32)
    begin_mask_1 = 0
    end_mask_1 = 0
    ellipsis_mask_1 = 0
    new_axis_mask_1 = 0
    shrink_axis_mask_1 = 0
    var_1 = None
    name_1 = "slice_1"
    input_dict_1 = {
        "input_": input_1,
        "begin": begin_1,
        "end": end_1,
        "strides": strides_1,
        "begin_mask": begin_mask_1,
        "end_mask": end_mask_1,
        "ellipsis_mask": ellipsis_mask_1,
        "new_axis_mask": new_axis_mask_1,
        "shrink_axis_mask": shrink_axis_mask_1,
        "var": var_1,
        "name": name_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = tf.constant(np.arange(24).reshape(2, 3, 4), dtype=tf.int32)
    begin_2 = np.array([0, 1, 2], dtype=np.int32)
    end_2 = np.array([2, 3, 4], dtype=np.int32)
    strides_2 = np.array([1, 1, 1], dtype=np.int32)
    begin_mask_2 = 0
    end_mask_2 = 0
    ellipsis_mask_2 = 0
    new_axis_mask_2 = 0
    shrink_axis_mask_2 = 0
    var_2 = None
    name_2 = "slice_2"
    input_dict_2 = {
        "input_": input_2,
        "begin": begin_2,
        "end": end_2,
        "strides": strides_2,
        "begin_mask": begin_mask_2,
        "end_mask": end_mask_2,
        "ellipsis_mask": ellipsis_mask_2,
        "new_axis_mask": new_axis_mask_2,
        "shrink_axis_mask": shrink_axis_mask_2,
        "var": var_2,
        "name": name_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = tf.constant(np.arange(12).reshape(3, 4), dtype=tf.int32)
    begin_3 = np.array([0, 0], dtype=np.int32)
    end_3 = np.array([3, 4], dtype=np.int32)
    strides_3 = np.array([-1, 1], dtype=np.int32)
    begin_mask_3 = 0
    end_mask_3 = 0
    ellipsis_mask_3 = 0
    new_axis_mask_3 = 0
    shrink_axis_mask_3 = 0
    var_3 = None
    name_3 = "slice_3"
    input_dict_3 = {
        "input_": input_3,
        "begin": begin_3,
        "end": end_3,
        "strides": strides_3,
        "begin_mask": begin_mask_3,
        "end_mask": end_mask_3,
        "ellipsis_mask": ellipsis_mask_3,
        "new_axis_mask": new_axis_mask_3,
        "shrink_axis_mask": shrink_axis_mask_3,
        "var": var_3,
        "name": name_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_4 = tf.constant(np.arange(8).reshape(2, 2, 2), dtype=tf.int32)
    begin_4 = np.array([0, 0, 0], dtype=np.int32)
    end_4 = np.array([1, 2, 2], dtype=np.int32)
    strides_4 = np.array([1, 1, 1], dtype=np.int32)
    begin_mask_4 = 1
    end_mask_4 = 0
    ellipsis_mask_4 = 0
    new_axis_mask_4 = 0
    shrink_axis_mask_4 = 0
    var_4 = None
    name_4 = "slice_4"
    input_dict_4 = {
        "input_": input_4,
        "begin": begin_4,
        "end": end_4,
        "strides": strides_4,
        "begin_mask": begin_mask_4,
        "end_mask": end_mask_4,
        "ellipsis_mask": ellipsis_mask_4,
        "new_axis_mask": new_axis_mask_4,
        "shrink_axis_mask": shrink_axis_mask_4,
        "var": var_4,
        "name": name_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_5 = tf.constant(np.arange(16).reshape(4, 4), dtype=tf.int32)
    begin_5 = np.array([0, 0], dtype=np.int32)
    end_5 = np.array([4, 4], dtype=np.int32)
    strides_5 = np.array([1, 1], dtype=np.int32)
    begin_mask_5 = 0
    end_mask_5 = 1
    ellipsis_mask_5 = 0
    new_axis_mask_5 = 0
    shrink_axis_mask_5 = 0
    var_5 = None
    name_5 = "slice_5"
    input_dict_5 = {
        "input_": input_5,
        "begin": begin_5,
        "end": end_5,
        "strides": strides_5,
        "begin_mask": begin_mask_5,
        "end_mask": end_mask_5,
        "ellipsis_mask": ellipsis_mask_5,
        "new_axis_mask": new_axis_mask_5,
        "shrink_axis_mask": shrink_axis_mask_5,
        "var": var_5,
        "name": name_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    input_6 = tf.constant(np.arange(20).reshape(5, 4), dtype=tf.int32)
    begin_6 = np.array([0, 0], dtype=np.int32)
    end_6 = np.array([5, 4], dtype=np.int32)
    strides_6 = np.array([1, 2], dtype=np.int32)
    begin_mask_6 = 0
    end_mask_6 = 0
    ellipsis_mask_6 = 0
    new_axis_mask_6 = 0
    shrink_axis_mask_6 = 0
    var_6 = None
    name_6 = "slice_6"
    input_dict_6 = {
        "input_": input_6,
        "begin": begin_6,
        "end": end_6,
        "strides": strides_6,
        "begin_mask": begin_mask_6,
        "end_mask": end_mask_6,
        "ellipsis_mask": ellipsis_mask_6,
        "new_axis_mask": new_axis_mask_6,
        "shrink_axis_mask": shrink_axis_mask_6,
        "var": var_6,
        "name": name_6
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    input_7 = tf.constant(np.arange(10).reshape(5, 2), dtype=tf.int32)
    begin_7 = np.array([0, 0], dtype=np.int32)
    end_7 = np.array([5, 2], dtype=np.int32)
    strides_7 = np.array([2, 1], dtype=np.int32)
    begin_mask_7 = 0
    end_mask_7 = 0
    ellipsis_mask_7 = 0
    new_axis_mask_7 = 1
    shrink_axis_mask_7 = 0
    var_7 = None
    name_7 = "slice_7"
    input_dict_7 = {
        "input_": input_7,
        "begin": begin_7,
        "end": end_7,
        "strides": strides_7,
        "begin_mask": begin_mask_7,
        "end_mask": end_mask_7,
        "ellipsis_mask": ellipsis_mask_7,
        "new_axis_mask": new_axis_mask_7,
        "shrink_axis_mask": shrink_axis_mask_7,
        "var": var_7,
        "name": name_7
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    input_8 = tf.constant(np.arange(12).reshape(3, 4), dtype=tf.int32)
    begin_8 = np.array([0, 1], dtype=np.int32)
    end_8 = np.array([3, 3], dtype=np.int32)
    strides_8 = np.array([1, 1], dtype=np.int32)
    begin_mask_8 = 0
    end_mask_8 = 0
    ellipsis_mask_8 = 0
    new_axis_mask_8 = 0
    shrink_axis_mask_8 = 2
    var_8 = None
    name_8 = "slice_8"
    input_dict_8 = {
        "input_": input_8,
        "begin": begin_8,
        "end": end_8,
        "strides": strides_8,
        "begin_mask": begin_mask_8,
        "end_mask": end_mask_8,
        "ellipsis_mask": ellipsis_mask_8,
        "new_axis_mask": new_axis_mask_8,
        "shrink_axis_mask": shrink_axis_mask_8,
        "var": var_8,
        "name": name_8
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    input_9 = tf.constant(np.arange(24).reshape(2, 3, 4), dtype=tf.int32)
    begin_9 = np.array([0, 0, 0], dtype=np.int32)
    end_9 = np.array([1, 3, 4], dtype=np.int32)
    strides_9 = np.array([1, 1, 2], dtype=np.int32)
    begin_mask_9 = 0
    end_mask_9 = 0
    ellipsis_mask_9 = 0
    new_axis_mask_9 = 0
    shrink_axis_mask_9 = 0
    var_9 = None
    name_9 = "slice_9"
    input_dict_9 = {
        "input_": input_9,
        "begin": begin_9,
        "end": end_9,
        "strides": strides_9,
        "begin_mask": begin_mask_9,
        "end_mask": end_mask_9,
        "ellipsis_mask": ellipsis_mask_9,
        "new_axis_mask": new_axis_mask_9,
        "shrink_axis_mask": shrink_axis_mask_9,
        "var": var_9,
        "name": name_9
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    input_10 = tf.constant(np.arange(30).reshape(5, 3, 2), dtype=tf.int32)
    begin_10 = np.array([1, 0, 0], dtype=np.int32)
    end_10 = np.array([4, 3, 2], dtype=np.int32)
    strides_10 = np.array([1, 1, 1], dtype=np.int32)
    begin_mask_10 = 0
    end_mask_10 = 0
    ellipsis_mask_10 = 0
    new_axis_mask_10 = 0
    shrink_axis_mask_10 = 0
    var_10 = None
    name_10 = "slice_10"
    input_dict_10 = {
        "input_": input_10,
        "begin": begin_10,
        "end": end_10,
        "strides": strides_10,
        "begin_mask": begin_mask_10,
        "end_mask": end_mask_10,
        "ellipsis_mask": ellipsis_mask_10,
        "new_axis_mask": new_axis_mask_10,
        "shrink_axis_mask": shrink_axis_mask_10,
        "var": var_10,
        "name": name_10
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.strided_slice"] = tf_strided_slice_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_strings_unicode_script_inputs():
    list_of_inputs = []

    input1 = np.array([1, 31, 38], dtype=np.int32)
    input_dict1 = {"input": input1, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([65, 97, 122], dtype=np.int32)
    input_dict2 = {"input": input2, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0, 10000, 65535], dtype=np.int32)
    input_dict3 = {"input": input3, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([-1, -2, -3], dtype=np.int32)
    input_dict4 = {"input": input4, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1, 1, 1], dtype=np.int32)
    input_dict5 = {"input": input5, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict6 = {"input": input6, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict7 = {"input": input7, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([100, 200, 300], dtype=np.int32)
    input_dict8 = {"input": input8, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([66, 67, 68], dtype=np.int32)
    input_dict9 = {"input": input9, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    input_dict10 = {"input": input10, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.strings.unicode_script"] = tf_strings_unicode_script_inputs()


