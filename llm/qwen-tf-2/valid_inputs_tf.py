generated_inputs = {}
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

