generated_inputs = {}
import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_indexed_slices_inputs():
    list_of_inputs = []

    values1 = np.array([1.0, 2.0, 3.0])
    indices1 = np.array([0, 2])
    dense_shape1 = np.array([5])
    input_dict1 = {'values': values1, 'indices': indices1, 'dense_shape': dense_shape1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    values2 = np.array([[1, 2], [3, 4]])
    indices2 = np.array([1, 3])
    dense_shape2 = np.array([5, 2])
    input_dict2 = {'values': values2, 'indices': indices2, 'dense_shape': dense_shape2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    values3 = np.array([[-1, 0, 1], [2, -3, 4]])
    indices3 = np.array([0, 1, 2])
    dense_shape3 = np.array([4, 3])
    input_dict3 = {'values': values3, 'indices': indices3, 'dense_shape': dense_shape3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    values4 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    indices4 = np.array([0, 2])
    dense_shape4 = np.array([4, 2, 2])
    input_dict4 = {'values': values4, 'indices': indices4, 'dense_shape': dense_shape4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    values5 = np.array([10, 20, 30, 40])
    indices5 = np.array([1, 3])
    dense_shape5 = np.array([6])
    input_dict5 = {'values': values5, 'indices': indices5, 'dense_shape': dense_shape5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    values6 = np.array([[1,2],[3,4],[5,6]])
    indices6 = np.array([0,1,2])
    dense_shape6 = np.array([3,2])
    input_dict6 = {'values': values6, 'indices': indices6, 'dense_shape': dense_shape6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    values7 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    indices7 = np.array([0, 2, 4])
    dense_shape7 = np.array([6])
    input_dict7 = {'values': values7, 'indices': indices7, 'dense_shape': dense_shape7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    values8 = np.array([[-1.0], [2.0], [-3.0]])
    indices8 = np.array([0, 1, 2])
    dense_shape8 = np.array([3, 1])
    input_dict8 = {'values': values8, 'indices': indices8, 'dense_shape': dense_shape8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    values9 = np.array([1, 2, 3])
    indices9 = np.array([0])
    dense_shape9 = np.array([3])
    input_dict9 = {'values': values9, 'indices': indices9, 'dense_shape': dense_shape9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    values10 = np.array([[1, 2, 3], [4, 5, 6]])
    indices10 = np.array([0, 1])
    dense_shape10 = np.array([2, 3])
    input_dict10 = {'values': values10, 'indices': indices10, 'dense_shape': dense_shape10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.IndexedSlices"] = tf_indexed_slices_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_broadcast_dynamic_shape_inputs():
    list_of_inputs = []

    shape_x = np.array([1, 2, 3], dtype=np.int32)
    shape_y = np.array([5, 1, 3], dtype=np.int32)
    list_of_inputs.append({"shape_x": shape_x, "shape_y": shape_y})

    shape_x = np.array([1], dtype=np.int32)
    shape_y = np.array([5, 6, 7], dtype=np.int32)
    list_of_inputs.append({"shape_x": shape_x, "shape_y": shape_y})

    shape_x = np.array([4, 1, 1], dtype=np.int32)
    shape_y = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"shape_x": shape_x, "shape_y": shape_y})

    shape_x = np.array([1, 5], dtype=np.int32)
    shape_y = np.array([1, 5], dtype=np.int32)
    list_of_inputs.append({"shape_x": shape_x, "shape_y": shape_y})

    shape_x = np.array([3, 4, 5], dtype=np.int32)
    shape_y = np.array([1, 1, 5], dtype=np.int32)
    list_of_inputs.append({"shape_x": shape_x, "shape_y": shape_y})

    return list_of_inputs

generated_inputs["tf.broadcast_dynamic_shape"] = tf_broadcast_dynamic_shape_inputs()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def forward_compatible_inputs():
    list_of_inputs = []

    input_dict = {"year": 2024, "month": 1, "day": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"year": 2023, "month": 12, "day": 31}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"year": 2025, "month": 6, "day": 15}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"year": 2024, "month": 2, "day": 29}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"year": 2023, "month": 3, "day": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"year": 2026, "month": 1, "day": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"year": 2022, "month": 11, "day": 20}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"year": 2024, "month": 12, "day": 31}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"year": 2023, "month": 6, "day": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"year": 2025, "month": 3, "day": 15}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"year": 2027, "month": 1, "day": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.compat.forward_compatible"] = forward_compatible_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_convert_to_tensor_inputs():
    list_of_inputs = []

    value1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    dtype1 = None
    dtype_hint1 = tf.float32
    name1 = "tensor1"
    input_dict1 = {
        "value": value1,
        "dtype": dtype1,
        "dtype_hint": dtype_hint1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    value2 = np.array([1, 2, 3], dtype=np.int32)
    dtype2 = None
    dtype_hint2 = tf.int64
    name2 = "tensor2"
    input_dict2 = {
        "value": value2,
        "dtype": dtype2,
        "dtype_hint": dtype_hint2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    value3 = np.array([[-1, 2], [3, -4]], dtype=np.float64)
    dtype3 = None
    dtype_hint3 = tf.float32
    name3 = "tensor3"
    input_dict3 = {
        "value": value3,
        "dtype": dtype3,
        "dtype_hint": dtype_hint3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    value4 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int64)
    dtype4 = None
    dtype_hint4 = tf.int64
    name4 = "tensor4"
    input_dict4 = {
        "value": value4,
        "dtype": dtype4,
        "dtype_hint": dtype_hint4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    value5 = np.array([], dtype=np.float32)
    dtype5 = tf.float32
    dtype_hint5 = None
    name5 = "tensor5"
    input_dict5 = {
        "value": value5,
        "dtype": dtype5,
        "dtype_hint": dtype_hint5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    value6 = np.array([1.0], dtype=np.float64)
    dtype6 = None
    dtype_hint6 = tf.float64
    name6 = "tensor6"
    input_dict6 = {
        "value": value6,
        "dtype": dtype6,
        "dtype_hint": dtype_hint6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["tf.convert_to_tensor"] = tf_convert_to_tensor_inputs()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_Counter_inputs():
    list_of_inputs = []

    input_dict = {
        "start": np.int64(0),
        "step": np.int64(1),
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(5),
        "step": np.int64(2),
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(-3),
        "step": np.int64(1),
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(10),
        "step": np.int64(-2),
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(100),
        "step": np.int64(5),
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(-10),
        "step": np.int64(-1),
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(0),
        "step": np.int64(-1),
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(7),
        "step": np.int64(3),
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(-5),
        "step": np.int64(2),
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "start": np.int64(15),
        "step": np.int64(-3),
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.data.experimental.Counter"] = tf_data_experimental_Counter_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_data_experimental_from_variant_inputs():
    list_of_inputs = []
    
    variant1 = tf.constant([1, 2, 3])
    structure1 = [tf.TensorSpec(shape=(), dtype=np.int32)]
    list_of_inputs.append({"variant": variant1, "structure": structure1})
    
    variant2 = tf.constant([[1, 2], [3, 4]])
    structure2 = [tf.TensorSpec(shape=(2,), dtype=np.int32)]
    list_of_inputs.append({"variant": variant2, "structure": structure2})
    
    variant3 = tf.constant([1.0, 2.0, 3.0])
    structure3 = [tf.TensorSpec(shape=(), dtype=np.float32)]
    list_of_inputs.append({"variant": variant3, "structure": structure3})
    
    variant4 = tf.constant([[-1, 2], [3, -4]])
    structure4 = [tf.TensorSpec(shape=(2,), dtype=np.int32)]
    list_of_inputs.append({"variant": variant4, "structure": structure4})
    
    variant5 = tf.constant([True, False, True])
    structure5 = [tf.TensorSpec(shape=(), dtype=np.bool_)]
    list_of_inputs.append({"variant": variant5, "structure": structure5})

    variant6 = tf.constant([1, 2, 3, 4, 5])
    structure6 = [tf.TensorSpec(shape=(), dtype=np.int64)]
    list_of_inputs.append({"variant": variant6, "structure": structure6})

    variant7 = tf.constant([0.1, 0.2, 0.3])
    structure7 = [tf.TensorSpec(shape=(), dtype=np.float64)]
    list_of_inputs.append({"variant": variant7, "structure": structure7})
    
    return list_of_inputs

generated_inputs["tf.data.experimental.from_variant"] = tf_data_experimental_from_variant_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_append_inputs():
    list_of_inputs = []

    arr1 = np.array([1, 2, 3])
    values1 = np.array([4, 5, 6])
    axis1 = None
    input_dict1 = {"arr": arr1, "values": values1, "axis": axis1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    arr2 = np.array([[1, 2], [3, 4]])
    values2 = np.array([[5, 6]])
    axis2 = 0
    input_dict2 = {"arr": arr2, "values": values2, "axis": axis2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    arr3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    values3 = np.array([[[9, 10], [11, 12]]])
    axis3 = 0
    input_dict3 = {"arr": arr3, "values": values3, "axis": axis3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    arr4 = np.array([1, 2, 3])
    values4 = np.array([4])
    axis4 = None
    input_dict4 = {"arr": arr4, "values": values4, "axis": axis4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    arr5 = np.array([[1, 2], [3, 4]])
    values5 = np.array([[5], [6]])
    axis5 = 1
    input_dict5 = {"arr": arr5, "values": values5, "axis": axis5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    arr6 = np.array([[[1, 2], [3, 4]]])
    values6 = np.array([[[5, 6], [7, 8]]])
    axis6 = 0
    input_dict6 = {"arr": arr6, "values": values6, "axis": axis6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    arr7 = np.array([1, 2, 3, 4])
    values7 = np.array([5, 6])
    axis7 = None
    input_dict7 = {"arr": arr7, "values": values7, "axis": axis7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    arr8 = np.array([[1, 2, 3], [4, 5, 6]])
    values8 = np.array([[7, 8, 9]])
    axis8 = 0
    input_dict8 = {"arr": arr8, "values": values8, "axis": axis8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    arr9 = np.array([1, 2, 3])
    values9 = np.array([4, 5, 6])
    axis9 = -1
    input_dict9 = {"arr": arr9, "values": values9, "axis": axis9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    arr10 = np.array([[[1, 2]]])
    values10 = np.array([[[3, 4]]])
    axis10 = 0
    input_dict10 = {"arr": arr10, "values": values10, "axis": axis10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.append"] = tf_experimental_numpy_append_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_argmin_inputs():
    list_of_inputs = []

    a = np.array([1, 2, 3, 4, 5])
    axis = None
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3], [4, 5, 6]])
    axis = 0
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3], [4, 5, 6]])
    axis = 1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[-1, 2, -3], [4, -5, 6]])
    axis = None
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 0
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 2
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1.0, 2.5, 3.7, 4.2, 5.9])
    axis = None
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    axis = 0
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[10, 2], [3, -4]], dtype=np.int8)
    axis = 1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.argmin"] = tf_experimental_numpy_argmin_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_bitwise_not_inputs():
    list_of_inputs = []
    
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1, -2, -3], dtype=np.int64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 0], [0, 1]], [[1, 1], [0, 0]]], dtype=np.bool_)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 20, 30], dtype=np.int16)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.bitwise_not"] = tf_bitwise_not_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_conj_inputs():
    list_of_inputs = []

    x = np.array([1+1j, 2+2j, 3+3j])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1-1j, -2-2j], [-3-3j, -4-4j]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0+1.0j, 2.0+2.0j, 3.0+3.0j])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0+0j])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1+0j, 2+0j, 3+0j])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0+1j, 0+2j, 0+3j])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1-1j, 2-2j, 3-3j])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[1+1j], [2+2j]], [[3+3j], [4+4j]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.conj"] = tf_experimental_numpy_conj_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_cumsum_inputs():
    list_of_inputs = []

    a = np.array([1, 2, 3])
    axis = None
    dtype = np.int32
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3], [4, 5, 6]])
    axis = 0
    dtype = np.int64
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3], [4, 5, 6]])
    axis = 1
    dtype = np.float32
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = 2
    dtype = np.int16
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([-1, -2, -3])
    axis = None
    dtype = np.int32
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1.5, 2.5, 3.5])
    axis = None
    dtype = np.float64
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2], [3, 4]])
    axis = 0
    dtype = np.float32
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    axis = 1
    dtype = np.int64
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1, 2, 3, 4, 5])
    axis = None
    dtype = np.int8
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axis = None
    dtype = np.int32
    input_dict = {"a": a, "axis": axis, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.cumsum"] = tf_experimental_numpy_cumsum_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_diag_inputs():
    list_of_inputs = []
    
    v1 = np.array([[1, 2], [3, 4]])
    k1 = 0
    input_dict1 = {"v": v1, "k": k1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    v2 = np.array([1, 2, 3])
    k2 = 0
    input_dict2 = {"v": v2, "k": k2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    v3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k3 = 1
    input_dict3 = {"v": v3, "k": k3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    v4 = np.array([[1, 2], [3, 4]])
    k4 = -1
    input_dict4 = {"v": v4, "k": k4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    v5 = np.array([1, 2])
    k5 = 0
    input_dict5 = {"v": v5, "k": k5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    v6 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    k6 = 2
    input_dict6 = {"v": v6, "k": k6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    v7 = np.array([[1, 2, 3], [4, 5, 6]])
    k7 = -1
    input_dict7 = {"v": v7, "k": k7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    v8 = np.array([1, 2, 3, 4, 5])
    k8 = 1
    input_dict8 = {"v": v8, "k": k8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    v9 = np.array([[1, 2], [3, 4]])
    k9 = 0
    input_dict9 = {"v": v9, "k": k9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    v10 = np.array([[1, 2, 3], [4, 5, 6]])
    k10 = -2
    input_dict10 = {"v": v10, "k": k10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.diag"] = tf_experimental_numpy_diag_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_fix_inputs():
    list_of_inputs = []
    
    x = np.array([1.2, 2.5, 3.7, -4.1])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.2, -2.5, -3.7, 4.1])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, 0.5, -0.5, 1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.2, 2.5], [3.7, -4.1]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.2, 2.5], [3.7, -4.1]], [[5.0, 6.0], [-7.0, 8.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.99999, 2.00001, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-1.99999, -2.00001, -3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.23456789, 2.98765432, -3.14159265])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.00000001, -0.00000001, 1e-10])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.2], [2.5], [3.7]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.fix"] = tf_experimental_numpy_fix_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_floor_divide_inputs():
    list_of_inputs = []

    x1 = np.array([10, 20, 30])
    x2 = np.array([3, 4, 5])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([-10, -20, -30])
    x2 = np.array([3, 4, 5])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([10, 20, 30])
    x2 = np.array([-3, -4, -5])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([[-10, -20], [-30, -40]])
    x2 = np.array([3, 4])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([10, 20, 30])
    x2 = np.array([2])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([1.5, 2.5, 3.5])
    x2 = np.array([1, 2, 3])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([10, 20, 30])
    x2 = np.array([1, 1, 1])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([10, 20, 30])
    x2 = np.array([2, 2, 2])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([10, 20, 30])
    x2 = np.array([3, 4, 6])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([[1, 2], [3, 4]])
    x2 = np.array([[2, 3], [4, 5]])
    list_of_inputs.append({"x1": x1, "x2": x2})

    x1 = np.array([10])
    x2 = np.array([3])
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["tf.experimental.numpy.floor_divide"] = tf_floor_divide_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_isfinite_inputs():
    list_of_inputs = []

    x = np.array([1.0, 2.0, np.inf, np.nan, 5.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0, np.inf, np.nan, -5.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0], [np.inf, np.nan]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0, np.nan], [2.0, np.inf]], [[np.nan, 5.0], [6.0, 7.0]]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.inf, np.nan, np.inf])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0], [2.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, np.nan, 2.0, np.inf, -3.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1.0, np.inf], [np.nan, -2.0]])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([0.0, -0.0, 1.0, -1.0])
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.isfinite"] = tf_experimental_numpy_isfinite_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_isinf_inputs():
    list_of_inputs = []

    x = np.array([1.0, np.inf, -np.inf, 2.0])
    list_of_inputs.append({"x": x})

    x = np.array([[-np.inf, 0.0, np.inf], [2.0, -1.0, np.nan]])
    list_of_inputs.append({"x": x})

    x = np.array([1.0, 2.0, 3.0])
    list_of_inputs.append({"x": x})

    return list_of_inputs

generated_inputs["tf.experimental.numpy.isinf"] = tf_experimental_numpy_isinf_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_isnan_inputs():
    list_of_inputs = []

    x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    list_of_inputs.append({"x": x})

    x = np.array([[-1.0, 2.0], [np.nan, 4.0]])
    list_of_inputs.append({"x": x})

    x = np.array([[[1.0, np.nan], [3.0, 4.0]], [[np.nan, 6.0], [7.0, 8.0]]])
    list_of_inputs.append({"x": x})
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.isnan"] = tf_experimental_numpy_isnan_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_isposinf_inputs():
    list_of_inputs = []

    x = np.array([np.inf], dtype=np.float32)
    list_of_inputs.append({"x": x})

    return list_of_inputs

generated_inputs["tf.experimental.numpy.isposinf"] = tf_experimental_numpy_isposinf_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_kron_inputs():
    list_of_inputs = []

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([[1, 0], [0, 1]])
    b = np.array([[2, 3], [4, 5]])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7, 8], [9, 10]])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([[-1, 2], [3, -4]])
    b = np.array([[5, -6], [-7, 8]])
    list_of_inputs.append({"a": a, "b": b})
    
    a = np.array([[1]])
    b = np.array([[2]])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([[1, 2], [3, 4]])
    b = np.array([1])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([1])
    b = np.array([[5, 6], [7, 8]])
    list_of_inputs.append({"a": a, "b": b})

    return list_of_inputs

generated_inputs["tf.experimental.numpy.kron"] = tf_experimental_numpy_kron_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_moveaxis_inputs():
    list_of_inputs = []

    a = np.random.rand(3, 4, 5)
    source = 0
    destination = 2
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.random.rand(2, 3)
    source = 1
    destination = 0
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.random.rand(1, 2, 3, 4)
    source = -1
    destination = 0
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.random.rand(5, 4, 3, 2, 1)
    source = 2
    destination = 4
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.random.rand(6, 7, 8)
    source = -2
    destination = 1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.random.rand(4, 5, 6, 7)
    source = 0
    destination = 3
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.random.rand(2, 2, 2, 2)
    source = 1
    destination = 2
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.random.rand(7, 6)
    source = 0
    destination = 1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.random.rand(10, 9, 8, 7)
    source = -3
    destination = 0
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.random.rand(3, 3, 3)
    source = 0
    destination = 1
    input_dict = {
        "a": a,
        "source": source,
        "destination": destination
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.moveaxis"] = tf_experimental_numpy_moveaxis_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_nanprod_inputs():
    list_of_inputs = []

    a = np.array([1.0, 2.0, 3.0])
    axis = None
    dtype = np.float32
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
    axis = 0
    dtype = np.float64
    keepdims = True
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]])
    axis = 1
    dtype = np.float32
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    axis = 2
    dtype = np.float64
    keepdims = True
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1.0, 2.0, 3.0, np.nan])
    axis = None
    dtype = np.float16
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
    axis = 0
    dtype = np.float32
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [7.0, np.nan]]])
    axis = 1
    dtype = np.float64
    keepdims = True
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1.0, 2.0, 3.0])
    axis = None
    dtype = np.float32
    keepdims = True
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    axis = 0
    dtype = np.float64
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    axis = None
    dtype = np.float16
    keepdims = False
    input_dict = {
        "a": a,
        "axis": axis,
        "dtype": dtype,
        "keepdims": keepdims
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.nanprod"] = tf_experimental_numpy_nanprod_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_promote_types_inputs():
    list_of_inputs = []

    input_dict = {
        'type1': np.dtype('int32'),
        'type2': np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('float64'),
        'type2': np.dtype('complex128')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('int8'),
        'type2': np.dtype('int16')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('uint8'),
        'type2': np.dtype('uint16')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('complex64'),
        'type2': np.dtype('complex128')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('float16'),
        'type2': np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('int64'),
        'type2': np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('bool'),
        'type2': np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('object'),
        'type2': np.dtype('str_')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'type1': np.dtype('float32'),
        'type2': np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.promote_types"] = tf_promote_types_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_ravel_inputs():
    list_of_inputs = []

    a = np.array([[1, 2, 3], [4, 5, 6]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1, 2, 3, 4, 5])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[-1, -2], [3, 4]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([0, 0, 0])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1.0, 2.0, 3.0])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2]]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2], [3, 4], [5, 6]])
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.ravel"] = tf_experimental_numpy_ravel_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_signbit_inputs():
    list_of_inputs = []
    
    x = tf.constant([-1.0, -2.0, -3.0], dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.signbit"] = tf_experimental_numpy_signbit_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_tril_inputs():
    list_of_inputs = []

    m1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k1 = 0
    input_dict1 = {"m": m1, "k": k1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    m2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k2 = -1
    input_dict2 = {"m": m2, "k": k2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    m3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k3 = 1
    input_dict3 = {"m": m3, "k": k3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    m4 = np.array([[1, 2], [3, 4]])
    k4 = -1
    input_dict4 = {"m": m4, "k": k4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    m5 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    k5 = 2
    input_dict5 = {"m": m5, "k": k5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    m6 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    k6 = 0
    input_dict6 = {"m": m6, "k": k6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    m7 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    k7 = -1
    input_dict7 = {"m": m7, "k": k7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    m8 = np.array([[1.0, 2.0], [3.0, 4.0]])
    k8 = 0
    input_dict8 = {"m": m8, "k": k8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    m9 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k9 = 3
    input_dict9 = {"m": m9, "k": k9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    m10 = np.array([[1, 2, 3, 4]])
    k10 = -2
    input_dict10 = {"m": m10, "k": k10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.tril"] = tf_experimental_numpy_tril_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_vdot_inputs():
    list_of_inputs = []

    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([1+1j, 2-1j])
    b = np.array([3+2j, 4-3j])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([1, -2, 3])
    b = np.array([-4, 5, -6])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7, 8])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.4, 0.5, 0.6])
    list_of_inputs.append({"a": a, "b": b})

    a = np.array([-1, -2, -3])
    b = np.array([-4, -5, -6])
    list_of_inputs.append({"a": a, "b": b})

    return list_of_inputs

generated_inputs["tf.experimental.numpy.vdot"] = tf_experimental_numpy_vdot_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_central_crop_inputs():
    list_of_inputs = []

    image1 = np.random.rand(100, 100, 3).astype(np.float32)
    central_fraction1 = 0.5
    input_dict1 = {"image": image1, "central_fraction": central_fraction1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.random.rand(50, 50, 1).astype(np.float32)
    central_fraction2 = 0.8
    input_dict2 = {"image": image2, "central_fraction": central_fraction2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.rand(256, 256, 3).astype(np.float32)
    central_fraction3 = 0.2
    input_dict3 = {"image": image3, "central_fraction": central_fraction3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.random.rand(32, 32, 1).astype(np.float32)
    central_fraction4 = 1.0
    input_dict4 = {"image": image4, "central_fraction": central_fraction4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.random.rand(64, 64, 3).astype(np.float32)
    central_fraction5 = 0.1
    input_dict5 = {"image": image5, "central_fraction": central_fraction5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = np.random.rand(128, 128, 1).astype(np.float32)
    central_fraction6 = 0.9
    input_dict6 = {"image": image6, "central_fraction": central_fraction6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    image7 = np.random.rand(200, 200, 3).astype(np.float32)
    central_fraction7 = 0.6
    input_dict7 = {"image": image7, "central_fraction": central_fraction7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    image8 = np.random.rand(10, 10, 1).astype(np.float32)
    central_fraction8 = 0.7
    input_dict8 = {"image": image8, "central_fraction": central_fraction8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    image9 = np.random.rand(16, 16, 3).astype(np.float32)
    central_fraction9 = 0.3
    input_dict9 = {"image": image9, "central_fraction": central_fraction9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    image10 = np.random.rand(300, 300, 1).astype(np.float32)
    central_fraction10 = 0.4
    input_dict10 = {"image": image10, "central_fraction": central_fraction10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.image.central_crop"] = tf_image_central_crop_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_crop_to_bounding_box_inputs():
    list_of_inputs = []

    image = np.arange(1, 28, dtype=np.float32).reshape([3, 3, 3])
    offset_height = 0
    offset_width = 0
    target_height = 2
    target_width = 2
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.rand(4, 5, 6, 3).astype(np.float32)
    offset_height = 1
    offset_width = 2
    target_height = 3
    target_width = 4
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.rand(5, 5, 3).astype(np.float32)
    offset_height = 2
    offset_width = 1
    target_height = 1
    target_width = 1
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.rand(2, 4, 4, 1).astype(np.float32)
    offset_height = 0
    offset_width = 0
    target_height = 2
    target_width = 4
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.rand(6, 6, 3).astype(np.float32)
    offset_height = 3
    offset_width = 3
    target_height = 2
    target_width = 2
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.rand(4, 4, 1).astype(np.float32)
    offset_height = 1
    offset_width = 1
    target_height = 2
    target_width = 2
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(5, 5, 3).astype(np.float32)
    offset_height = 0
    offset_width = 0
    target_height = 5
    target_width = 5
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.rand(3, 3, 3).astype(np.float32)
    offset_height = 1
    offset_width = 1
    target_height = 1
    target_width = 1
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.rand(4, 5, 6, 2).astype(np.float32)
    offset_height = 2
    offset_width = 3
    target_height = 1
    target_width = 2
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(2, 2, 2).astype(np.float32)
    offset_height = 0
    offset_width = 0
    target_height = 2
    target_width = 2
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs["tf.image.crop_to_bounding_box"] = tf_image_crop_to_bounding_box_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_grayscale_to_rgb_inputs():
    list_of_inputs = []

    images1 = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    name1 = "example_1"
    input_dict1 = {"images": images1, "name": name1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    images2 = np.array([[[[0.5]], [[0.8]]]], dtype=np.float32)
    name2 = "example_2"
    input_dict2 = {"images": images2, "name": name2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    images3 = np.array([[[[-1.0], [0.0]]]], dtype=np.float32)
    name3 = "example_3"
    input_dict3 = {"images": images3, "name": name3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    images4 = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
    name4 = "example_4"
    input_dict4 = {"images": images4, "name": name4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    images5 = np.array([[[[0.1], [0.2]], [[0.3], [0.4]]], [[ [0.5], [0.6]], [[0.7], [0.8]]]], dtype=np.float32)
    name5 = "example_5"
    input_dict5 = {"images": images5, "name": name5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    images6 = np.array([[[[1.0]]]], dtype=np.float64)
    name6 = "example_6"
    input_dict6 = {"images": images6, "name": name6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    images7 = np.array([[[[255.0]]]], dtype=np.uint8)
    name7 = "example_7"
    input_dict7 = {"images": images7, "name": name7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    images8 = np.array([[[[0.0]]]], dtype=np.float16)
    name8 = "example_8"
    input_dict8 = {"images": images8, "name": name8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    images9 = np.array([[[[1.0], [-1.0]]]], dtype=np.float32)
    name9 = "example_9"
    input_dict9 = {"images": images9, "name": name9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    images10 = np.array([[[[0.5]]]], dtype=np.complex64)
    name10 = "example_10"
    input_dict10 = {"images": images10, "name": name10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.image.grayscale_to_rgb"] = tf_image_grayscale_to_rgb_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_random_contrast_inputs():
    list_of_inputs = []

    image1 = np.random.rand(2, 2, 3).astype(np.float32)
    lower1 = 0.2
    upper1 = 0.5
    seed1 = 42
    input_dict1 = {
        "image": image1,
        "lower": lower1,
        "upper": upper1,
        "seed": seed1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.random.rand(4, 4, 3).astype(np.float32)
    lower2 = 0.1
    upper2 = 0.8
    seed2 = 100
    input_dict2 = {
        "image": image2,
        "lower": lower2,
        "upper": upper2,
        "seed": seed2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    lower3 = 0.8
    upper3 = 1.2
    seed3 = 0
    input_dict3 = {
        "image": image3,
        "lower": lower3,
        "upper": upper3,
        "seed": seed3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.random.rand(3, 3, 3).astype(np.float32)
    lower4 = 0.0
    upper4 = 0.3
    seed4 = 123
    input_dict4 = {
        "image": image4,
        "lower": lower4,
        "upper": upper4,
        "seed": seed4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.random.rand(2, 3, 4).astype(np.float32)
    lower5 = 0.1
    upper5 = 0.6
    seed5 = 789
    input_dict5 = {
        "image": image5,
        "lower": lower5,
        "upper": upper5,
        "seed": seed5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = np.random.rand(5, 5, 3).astype(np.float32)
    lower6 = 0.0
    upper6 = 0.2
    seed6 = 555
    input_dict6 = {
        "image": image6,
        "lower": lower6,
        "upper": upper6,
        "seed": seed6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    image7 = np.random.rand(1, 1, 1, 3).astype(np.float32)
    lower7 = 0.5
    upper7 = 0.7
    seed7 = 999
    input_dict7 = {
        "image": image7,
        "lower": lower7,
        "upper": upper7,
        "seed": seed7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    image8 = np.random.rand(2, 2, 2).astype(np.float32)
    lower8 = 0.2
    upper8 = 0.4
    seed8 = 111
    input_dict8 = {
        "image": image8,
        "lower": lower8,
        "upper": upper8,
        "seed": seed8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["tf.image.random_contrast"] = tf_image_random_contrast_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_random_hue_inputs():
    list_of_inputs = []

    image1 = np.random.rand(2, 2, 3).astype(np.float32)
    max_delta1 = 0.1
    seed1 = 10
    input_dict1 = {"image": image1, "max_delta": max_delta1, "seed": seed1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.random.rand(4, 4, 3).astype(np.float32)
    max_delta2 = 0.2
    seed2 = 20
    input_dict2 = {"image": image2, "max_delta": max_delta2, "seed": seed2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.rand(1, 1, 3).astype(np.float32)
    max_delta3 = 0.05
    seed3 = 30
    input_dict3 = {"image": image3, "max_delta": max_delta3, "seed": seed3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.random.rand(2, 3, 3).astype(np.float32)
    max_delta4 = 0.3
    seed4 = 40
    input_dict4 = {"image": image4, "max_delta": max_delta4, "seed": seed4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.random.rand(3, 2, 3).astype(np.float32)
    max_delta5 = 0.4
    seed5 = 50
    input_dict5 = {"image": image5, "max_delta": max_delta5, "seed": seed5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = np.random.rand(5, 5, 3).astype(np.float32)
    max_delta6 = 0.01
    seed6 = 60
    input_dict6 = {"image": image6, "max_delta": max_delta6, "seed": seed6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    image7 = np.random.rand(10, 10, 3).astype(np.float32)
    max_delta7 = 0.25
    seed7 = 70
    input_dict7 = {"image": image7, "max_delta": max_delta7, "seed": seed7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    image8 = np.random.rand(1, 5, 3).astype(np.float32)
    max_delta8 = 0.15
    seed8 = 80
    input_dict8 = {"image": image8, "max_delta": max_delta8, "seed": seed8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    image9 = np.random.rand(6, 3, 3).astype(np.float32)
    max_delta9 = 0.35
    seed9 = 90
    input_dict9 = {"image": image9, "max_delta": max_delta9, "seed": seed9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    image10 = np.random.rand(2, 2, 3).astype(np.float32)
    max_delta10 = 0.5
    seed10 = 100
    input_dict10 = {"image": image10, "max_delta": max_delta10, "seed": seed10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.image.random_hue"] = tf_image_random_hue_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_resize_with_crop_or_pad_inputs():
    list_of_inputs = []

    image1 = np.arange(75).reshape(5, 5, 3).astype(np.int64)
    target_height1 = 3
    target_width1 = 3
    input_dict1 = {"image": image1, "target_height": target_height1, "target_width": target_width1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.arange(1, 28).reshape(3, 3, 3).astype(np.int64)
    target_height2 = 5
    target_width2 = 5
    input_dict2 = {"image": image2, "target_height": target_height2, "target_width": target_width2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.rand(2, 2, 2, 3).astype(np.float32)
    target_height3 = 4
    target_width3 = 4
    input_dict3 = {"image": image3, "target_height": target_height3, "target_width": target_width3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.arange(12).reshape(2, 3, 2).astype(np.int32)
    target_height4 = 1
    target_width4 = 1
    input_dict4 = {"image": image4, "target_height": target_height4, "target_width": target_width4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.zeros((4, 4, 1)).astype(np.float64)
    target_height5 = 8
    target_width5 = 8
    input_dict5 = {"image": image5, "target_height": target_height5, "target_width": target_width5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = np.random.randint(0, 256, size=(6, 6, 3), dtype=np.uint8)
    target_height6 = 3
    target_width6 = 3
    input_dict6 = {"image": image6, "target_height": target_height6, "target_width": target_width6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    image7 = np.arange(16).reshape(4, 4, 1).astype(np.int64)
    target_height7 = 2
    target_width7 = 2
    input_dict7 = {"image": image7, "target_height": target_height7, "target_width": target_width7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["tf.image.resize_with_crop_or_pad"] = tf_image_resize_with_crop_or_pad_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_rgb_to_hsv_inputs():
    list_of_inputs = []

    images1 = np.random.rand(5, 5, 3).astype(np.float32)
    input_dict1 = {"images": tf.convert_to_tensor(images1), "name": "rgb_to_hsv_1"}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    images2 = np.random.rand(2, 3, 3).astype(np.float64)
    input_dict2 = {"images": tf.convert_to_tensor(images2), "name": "rgb_to_hsv_2"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    images3 = np.random.rand(10, 10, 3).astype(np.float16)
    input_dict3 = {"images": tf.convert_to_tensor(images3), "name": "rgb_to_hsv_3"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    images4 = np.random.rand(3, 3, 3).astype(np.float32)
    input_dict4 = {"images": tf.convert_to_tensor(images4), "name": "rgb_to_hsv_4"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    images5 = np.random.rand(4, 4, 3).astype(np.float32)
    input_dict5 = {"images": tf.convert_to_tensor(images5), "name": "rgb_to_hsv_5"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    images6 = np.random.rand(1, 1, 3).astype(np.float32)
    input_dict6 = {"images": tf.convert_to_tensor(images6), "name": "rgb_to_hsv_6"}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    images7 = np.random.rand(6, 6, 3).astype(np.float32)
    input_dict7 = {"images": tf.convert_to_tensor(images7), "name": "rgb_to_hsv_7"}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    images8 = np.random.rand(7, 7, 3).astype(np.float32)
    input_dict8 = {"images": tf.convert_to_tensor(images8), "name": "rgb_to_hsv_8"}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    images9 = np.random.rand(2, 2, 3).astype(np.float32)
    input_dict9 = {"images": tf.convert_to_tensor(images9), "name": "rgb_to_hsv_9"}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    images10 = np.random.rand(8, 8, 3).astype(np.float32)
    input_dict10 = {"images": tf.convert_to_tensor(images10), "name": "rgb_to_hsv_10"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.image.rgb_to_hsv"] = tf_image_rgb_to_hsv_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_stateless_random_flip_left_right_inputs():
    list_of_inputs = []

    image1 = np.array([[[1], [2]], [[3], [4]]], dtype=np.int32)
    seed1 = np.array([2, 3], dtype=np.int32)
    input_dict1 = {"image": image1, "seed": seed1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.random.randint(0, 256, size=(2, 3, 4, 1), dtype=np.int32)
    seed2 = np.array([5, 7], dtype=np.int32)
    input_dict2 = {"image": image2, "seed": seed2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.randint(-100, 100, size=(1, 5, 6, 3), dtype=np.int32)
    seed3 = np.array([10, 20], dtype=np.int32)
    input_dict3 = {"image": image3, "seed": seed3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.int64)
    seed4 = np.array([1, 1], dtype=np.int64)
    input_dict4 = {"image": image4, "seed": seed4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.random.randint(0, 256, size=(4, 4, 3), dtype=np.int32)
    seed5 = np.array([123, 456], dtype=np.int32)
    input_dict5 = {"image": image5, "seed": seed5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = np.random.randint(0, 256, size=(1, 1, 1, 1), dtype=np.int32)
    seed6 = np.array([789, 101], dtype=np.int32)
    input_dict6 = {"image": image6, "seed": seed6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    image7 = np.random.randint(-50, 50, size=(2, 2, 2, 2), dtype=np.int32)
    seed7 = np.array([2, 2], dtype=np.int32)
    input_dict7 = {"image": image7, "seed": seed7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    image8 = np.random.randint(0, 256, size=(3, 3, 3), dtype=np.int32)
    seed8 = np.array([99, 100], dtype=np.int32)
    input_dict8 = {"image": image8, "seed": seed8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    image9 = np.random.randint(0, 256, size=(5, 5, 1, 4), dtype=np.int32)
    seed9 = np.array([11, 12], dtype=np.int32)
    input_dict9 = {"image": image9, "seed": seed9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    image10 = np.random.randint(-200, 200, size=(1, 6, 7, 2), dtype=np.int32)
    seed10 = np.array([13, 14], dtype=np.int32)
    input_dict10 = {"image": image10, "seed": seed10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    

    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_left_right"] = tf_image_stateless_random_flip_left_right_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_stateless_random_flip_up_down_inputs():
    list_of_inputs = []

    image1 = np.array([[[1], [2]], [[3], [4]]])
    seed1 = np.array([2, 3], dtype=np.int32)
    input_dict1 = {"image": image1, "seed": seed1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    seed2 = np.array([5, 7], dtype=np.int64)
    input_dict2 = {"image": image2, "seed": seed2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = np.random.rand(1, 100, 100, 3).astype(np.float32)
    seed3 = np.array([10, 20], dtype=np.int32)
    input_dict3 = {"image": image3, "seed": seed3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = np.random.randint(0, 256, size=(2, 50, 50, 1), dtype=np.uint8)
    seed4 = np.array([15, 25], dtype=np.int32)
    input_dict4 = {"image": image4, "seed": seed4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = np.array([[[1, 2, 3]]])
    seed5 = np.array([8, 9], dtype=np.int32)
    input_dict5 = {"image": image5, "seed": seed5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = np.random.rand(4, 64, 64, 3).astype(np.float32)
    seed6 = np.array([1, 1], dtype=np.int32)
    input_dict6 = {"image": image6, "seed": seed6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    image7 = np.random.randint(0, 100, size=(1, 32, 32, 3), dtype=np.int32)
    seed7 = np.array([100, 200], dtype=np.int32)
    input_dict7 = {"image": image7, "seed": seed7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    image8 = np.array([[[[1, 2], [3, 4]]]])
    seed8 = np.array([123, 456], dtype=np.int32)
    input_dict8 = {"image": image8, "seed": seed8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    image9 = np.random.rand(2, 128, 128, 3).astype(np.float32)
    seed9 = np.array([7, 11], dtype=np.int32)
    input_dict9 = {"image": image9, "seed": seed9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    image10 = np.random.randint(0, 256, size=(3, 64, 64, 1), dtype=np.uint8)
    seed10 = np.array([111, 222], dtype=np.int32)
    input_dict10 = {"image": image10, "seed": seed10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_up_down"] = tf_image_stateless_random_flip_up_down_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_transpose_inputs():
    list_of_inputs = []

    image_1 = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    input_dict_1 = {"image": tf.constant(image_1), "name": "transpose_1"}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    image_2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.int32)
    input_dict_2 = {"image": tf.constant(image_2), "name": "transpose_2"}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    image_3 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict_3 = {"image": tf.constant(image_3), "name": "transpose_3"}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    image_4 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_dict_4 = {"image": tf.constant(image_4), "name": "transpose_4"}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    image_5 = np.random.randint(-10, 10, size=(1, 5, 5, 3)).astype(np.int32)
    input_dict_5 = {"image": tf.constant(image_5), "name": "transpose_5"}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    image_6 = np.array([[[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]], dtype=np.float32)
    input_dict_6 = {"image": tf.constant(image_6), "name": "transpose_6"}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    image_7 = np.random.rand(3, 4, 5).astype(np.float32)
    input_dict_7 = {"image": tf.constant(image_7), "name": "transpose_7"}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    image_8 = np.random.randint(0, 256, size=(2, 2, 3)).astype(np.uint8)
    input_dict_8 = {"image": tf.constant(image_8, dtype=tf.float32), "name": "transpose_8"}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    image_9 = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    input_dict_9 = {"image": tf.constant(image_9), "name": "transpose_9"}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    image_10 = np.random.rand(4, 3, 2, 1).astype(np.float32)
    input_dict_10 = {"image": tf.constant(image_10), "name": "transpose_10"}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.image.transpose"] = tf_image_transpose_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_linalg_LinearOperatorHouseholder_inputs():
    list_of_inputs = []

    input_dict1 = {
        'reflection_axis': np.array([1.0, 1.0], dtype=np.float32),
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'householder_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input_dict2 = {
        'reflection_axis': np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'householder_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input_dict3 = {
        'reflection_axis': np.array([0.5, -0.5, 0.7, -0.7], dtype=np.float32),
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'householder_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input_dict4 = {
        'reflection_axis': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'householder_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorHouseholder"] = tf_linalg_LinearOperatorHouseholder_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def linear_operator_zeros_inputs():
    list_of_inputs = []

    input_dict = {
        'num_rows': 2,
        'num_columns': 3,
        'batch_shape': [],
        'dtype': np.float32,
        'is_non_singular': False,
        'is_self_adjoint': False,
        'is_positive_definite': False,
        'is_square': False,
        'assert_proper_shapes': False,
        'name': 'LinearOperatorZeros_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'num_rows': 5,
        'num_columns': 5,
        'batch_shape': [2, 3],
        'dtype': np.complex64,
        'is_non_singular': False,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'assert_proper_shapes': True,
        'name': 'LinearOperatorZeros_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorZeros"] = linear_operator_zeros_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_linalg_eigh_inputs():
    list_of_inputs = []

    input_dict = {
        "tensor": np.array([[[1.0, 2.0], [2.0, 1.0]]], dtype=np.float64),
        "name": "eigh_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "tensor": np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        "name": "eigh_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.eigh"] = tf_linalg_eigh_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_linalg_lstsq_inputs():
    list_of_inputs = []

    matrix1 = np.random.rand(2, 3).astype(np.float64)
    rhs1 = np.random.rand(2, 1).astype(np.float64)
    input_dict1 = {
        "matrix": matrix1,
        "rhs": rhs1,
        "l2_regularizer": 0.0,
        "fast": True,
        "name": "lstsq_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    matrix2 = np.random.rand(3, 2).astype(np.float64)
    rhs2 = np.random.rand(3, 1).astype(np.float64)
    input_dict2 = {
        "matrix": matrix2,
        "rhs": rhs2,
        "l2_regularizer": 1e-6,
        "fast": False,
        "name": "lstsq_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    matrix3 = np.random.rand(4, 4).astype(np.float64)
    rhs3 = np.random.rand(4, 2).astype(np.float64)
    input_dict3 = {
        "matrix": matrix3,
        "rhs": rhs3,
        "l2_regularizer": 0.1,
        "fast": True,
        "name": "lstsq_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    matrix4 = np.random.rand(5, 3).astype(np.float64)
    rhs4 = np.random.rand(5, 3).astype(np.float64)
    input_dict4 = {
        "matrix": matrix4,
        "rhs": rhs4,
        "l2_regularizer": 0.0,
        "fast": False,
        "name": "lstsq_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    matrix5 = np.random.rand(2, 2).astype(np.float64)
    rhs5 = np.random.rand(2, 1).astype(np.float64)
    input_dict5 = {
        "matrix": matrix5,
        "rhs": rhs5,
        "l2_regularizer": 1.0,
        "fast": True,
        "name": "lstsq_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    matrix6 = np.random.rand(3, 4).astype(np.float64)
    rhs6 = np.random.rand(3, 2).astype(np.float64)
    input_dict6 = {
        "matrix": matrix6,
        "rhs": rhs6,
        "l2_regularizer": 0.01,
        "fast": False,
        "name": "lstsq_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    matrix7 = np.random.rand(4, 3).astype(np.float64)
    rhs7 = np.random.rand(4, 1).astype(np.float64)
    input_dict7 = {
        "matrix": matrix7,
        "rhs": rhs7,
        "l2_regularizer": 0.5,
        "fast": True,
        "name": "lstsq_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    matrix8 = np.random.rand(5, 5).astype(np.float64)
    rhs8 = np.random.rand(5, 3).astype(np.float64)
    input_dict8 = {
        "matrix": matrix8,
        "rhs": rhs8,
        "l2_regularizer": 0.0,
        "fast": False,
        "name": "lstsq_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    matrix9 = np.random.rand(2, 4).astype(np.float64)
    rhs9 = np.random.rand(2, 2).astype(np.float64)
    input_dict9 = {
        "matrix": matrix9,
        "rhs": rhs9,
        "l2_regularizer": 1e-8,
        "fast": True,
        "name": "lstsq_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    matrix10 = np.random.rand(6, 2).astype(np.float64)
    rhs10 = np.random.rand(6, 3).astype(np.float64)
    input_dict10 = {
        "matrix": matrix10,
        "rhs": rhs10,
        "l2_regularizer": 0.2,
        "fast": False,
        "name": "lstsq_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.linalg.lstsq"] = tf_linalg_lstsq_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_linalg_matrix_transpose_inputs():
    list_of_inputs = []

    a = np.array([[1, 2, 3], [4, 5, 6]])
    name = "transpose_1"
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
    name = "transpose_2"
    conjugate = True
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    name = "transpose_3"
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    name = "transpose_4"
    conjugate = True
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2], [3, 4], [5, 6]])
    name = "transpose_5"
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[-1, -2, -3], [-4, -5, -6]])
    name = "transpose_6"
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[0, 0, 0], [0, 0, 0]])
    name = "transpose_7"
    conjugate = True
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])
    name = "transpose_8"
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    name = "transpose_9"
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    name = "transpose_10"
    conjugate = True
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.matrix_transpose"] = tf_linalg_matrix_transpose_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_linalg_solve_inputs():
    list_of_inputs = []

    matrix1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rhs1 = np.array([[5.0], [11.0]], dtype=np.float32)
    adjoint1 = False
    name1 = "solve_example_1"

    input_dict1 = {
        "matrix": matrix1,
        "rhs": rhs1,
        "adjoint": adjoint1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    matrix2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    rhs2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    adjoint2 = False
    name2 = "solve_example_2"

    input_dict2 = {
        "matrix": matrix2,
        "rhs": rhs2,
        "adjoint": adjoint2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    matrix3 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    rhs3 = np.array([[[9.0], [10.0]], [[11.0], [12.0]]], dtype=np.float32)
    adjoint3 = False
    name3 = "solve_example_3"

    input_dict3 = {
        "matrix": matrix3,
        "rhs": rhs3,
        "adjoint": adjoint3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    matrix4 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    rhs4 = np.array([[1.0], [2.0]], dtype=np.float64)
    adjoint4 = True
    name4 = "solve_example_4"

    input_dict4 = {
        "matrix": matrix4,
        "rhs": rhs4,
        "adjoint": adjoint4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    matrix5 = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    rhs5 = np.array([[5.0], [8.0]], dtype=np.float32)
    adjoint5 = False
    name5 = "solve_example_5"

    input_dict5 = {
        "matrix": matrix5,
        "rhs": rhs5,
        "adjoint": adjoint5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["tf.linalg.solve"] = tf_linalg_solve_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_linalg_trace_inputs():
    list_of_inputs = []

    x = np.array([[1, 2], [3, 4]])
    input_dict = {'x': x, 'name': 'trace_1'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    input_dict = {'x': x, 'name': 'trace_2'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]])
    input_dict = {'x': x, 'name': 'trace_3'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {'x': x, 'name': 'trace_4'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]])
    input_dict = {'x': x, 'name': 'trace_5'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, -2], [-3, 4]])
    input_dict = {'x': x, 'name': 'trace_6'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])
    input_dict = {'x': x, 'name': 'trace_7'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]])
    input_dict = {'x': x, 'name': 'trace_8'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    input_dict = {'x': x, 'name': 'trace_9'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict = {'x': x, 'name': 'trace_10'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.trace"] = tf_linalg_trace_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_atan2_inputs():
    list_of_inputs = []

    y = np.array([1.0, -1.0])
    x = np.array([1.0, 1.0])
    name = "atan2_example_1"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([0.0, 0.0])
    x = np.array([1.0, -1.0])
    name = "atan2_example_2"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([-1.0, 1.0])
    x = np.array([-1.0, -1.0])
    name = "atan2_example_3"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([1.0, 2.0, 3.0])
    x = np.array([4.0, 5.0, 6.0])
    name = "atan2_example_4"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([[1.0, -1.0], [2.0, -2.0]])
    x = np.array([[1.0, 1.0], [2.0, 2.0]])
    name = "atan2_example_5"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    x = np.array([1.0, 1.0, -1.0], dtype=np.float32)
    name = "atan2_example_6"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    x = np.array([1.0, 1.0, -1.0], dtype=np.float64)
    name = "atan2_example_7"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([1.0, -1.0, 0.0], dtype=np.float16)
    x = np.array([1.0, 1.0, -1.0], dtype=np.float16)
    name = "atan2_example_8"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([[[1.0, -1.0]], [[2.0, -2.0]]])
    x = np.array([[[1.0, 1.0]], [[2.0, 2.0]]])
    name = "atan2_example_9"
    input_dict = {"y": y, "x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.atan2"] = tf_math_atan2_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_bessel_i1e_inputs():
    list_of_inputs = []

    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_dict = {"x": x, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-2.0, -1.5, 0.0, 1.5, 2.0], dtype=np.float64)
    input_dict = {"x": x, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict = {"x": x, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    input_dict = {"x": x, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-0.1, -0.2, -0.3, -0.4, -0.5], dtype=np.float64)
    input_dict = {"x": x, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float64)
    input_dict = {"x": x, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1], dtype=np.float64)
    input_dict = {"x": x, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.bessel_i1e"] = tf_math_bessel_i1e_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_cos_inputs():
    list_of_inputs = []

    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")])
    input_dict = {"x": x, "name": "cos_inf"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    input_dict = {"x": x, "name": "cos_special"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0])
    input_dict = {"x": x, "name": "cos_positive"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0, -3.0])
    input_dict = {"x": x, "name": "cos_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"x": x, "name": "cos_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    input_dict = {"x": x, "name": "cos_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0 + 1j, 2.0 + 2j, 3.0 + 3j])
    input_dict = {"x": x, "name": "cos_complex"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, 1.0, 2.0], dtype=np.float16)
    input_dict = {"x": x, "name": "cos_float16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    input_dict = {"x": x, "name": "cos_float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, 1.0, 2.0], dtype=np.complex64)
    input_dict = {"x": x, "name": "cos_complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.cos"] = tf_math_cos_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_cumprod_inputs():
    list_of_inputs = []

    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
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
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.5, 3.2, 4.1], dtype=np.float64)
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
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    axis = 1
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
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1, 2], [-3, 4]], dtype=np.int32)
    axis = 0
    exclusive = True
    reverse = False
    name = "cumprod_4"
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.uint8)
    axis = 0
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
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1j, 2j, 3j], dtype=np.complex64)
    axis = 0
    exclusive = True
    reverse = True
    name = "cumprod_6"
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis = 2
    exclusive = False
    reverse = False
    name = "cumprod_7"
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3, 4], dtype=np.int16)
    axis = 0
    exclusive = False
    reverse = True
    name = "cumprod_8"
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    axis = 0
    exclusive = True
    reverse = False
    name = "cumprod_9"
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.int8)
    axis = 0
    exclusive = False
    reverse = False
    name = "cumprod_10"
    input_dict = {
        "x": x,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.cumprod"] = tf_math_cumprod_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_erf_inputs():
    list_of_inputs = []

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "name": "erf_test_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, -2.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "erf_test_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1.5, 0.0, 1.5]], dtype=np.float32)
    input_dict = {"x": x, "name": "erf_test_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0.5], [1.0]], [[1.5], [2.0]]], dtype=np.float64)
    input_dict = {"x": x, "name": "erf_test_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "name": "erf_test_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float16)
    input_dict = {"x": x, "name": "erf_test_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1], dtype=np.float32)
    input_dict = {"x": x, "name": "erf_test_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e5, 1e6, 1e7, 1e8, 1e9], dtype=np.float64)
    input_dict = {"x": x, "name": "erf_test_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[np.nan, 2.0], [3.0, np.inf]], dtype=np.float32)
    input_dict = {"x": x, "name": "erf_test_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-np.inf, -2.0], [3.0, np.nan]], dtype=np.float64)
    input_dict = {"x": x, "name": "erf_test_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.erf"] = tf_math_erf_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_floormod_inputs():
    list_of_inputs = []

    x = np.array([10, 20, 30], dtype=np.int32)
    y = np.array([3, 7, 2], dtype=np.int32)
    input_dict = {'x': x, 'y': y, 'name': "floormod_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-10, -20, -30], dtype=np.int32)
    y = np.array([3, 7, 2], dtype=np.int32)
    input_dict = {'x': x, 'y': y, 'name': "floormod_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10.5, 20.5, 30.5], dtype=np.float32)
    y = np.array([3.2, 7.1, 2.5], dtype=np.float32)
    input_dict = {'x': x, 'y': y, 'name': "floormod_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[10, 20], [30, 40]], dtype=np.int64)
    y = np.array([3, 7], dtype=np.int64)
    input_dict = {'x': x, 'y': y, 'name': "floormod_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 20, 30], dtype=np.uint8)
    y = np.array([3, 7, 2], dtype=np.uint8)
    input_dict = {'x': x, 'y': y, 'name': "floormod_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 20, 30], dtype=np.float64)
    y = np.array([3.0, 7.0, 2.0], dtype=np.float64)
    input_dict = {'x': x, 'y': y, 'name': "floormod_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]], dtype=np.int32)
    y = np.array([3, 7], dtype=np.int32)
    input_dict = {'x': x, 'y': y, 'name': "floormod_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, -20, 30], dtype=np.int16)
    y = np.array([3, 7, 2], dtype=np.int16)
    input_dict = {'x': x, 'y': y, 'name': "floormod_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 20, 30], dtype=np.float16)
    y = np.array([3, 7, 2], dtype=np.float16)
    input_dict = {'x': x, 'y': y, 'name': "floormod_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.floormod"] = tf_math_floormod_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_invert_permutation_inputs():
    list_of_inputs = []

    x = np.array([3, 4, 0, 2, 1], dtype=np.int32)
    input_dict = {'x': x, 'name': 'example_1'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    input_dict = {'x': x, 'name': 'example_2'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([4, 3, 2, 1, 0], dtype=np.int32)
    input_dict = {'x': x, 'name': 'example_3'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 0], dtype=np.int64)
    input_dict = {'x': x, 'name': 'example_4'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0], dtype=np.int32)
    input_dict = {'x': x, 'name': 'example_5'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([5, 0, 2, 1, 4, 3], dtype=np.int64)
    input_dict = {'x': x, 'name': 'example_6'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([2, 0, 1], dtype=np.int32)
    input_dict = {'x': x, 'name': 'example_7'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 5, 0, 2, 8, 1, 7, 3, 6, 4, 9], dtype=np.int64)
    input_dict = {'x': x, 'name': 'example_8'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0, 2, 4, 1, 3], dtype=np.int32)
    input_dict = {'x': x, 'name': 'example_9'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int64)
    input_dict = {'x': x, 'name': 'example_10'}
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

    x = np.array([1.0, np.inf, 3.0, np.inf], dtype=np.float32)
    input_dict = {"x": x, "name": "test1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-np.inf, 2.5, -np.inf, 7.0], dtype=np.float64)
    input_dict = {"x": x, "name": "test2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, np.inf], [np.inf, 3.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "test3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0, np.inf], [np.inf, 3.0]], [[4.0, 5.0], [6.0, np.inf]]], dtype=np.float64)
    input_dict = {"x": x, "name": "test4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.inf], dtype=np.float16)
    input_dict = {"x": x, "name": "test5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "name": "test6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    input_dict = {"x": x, "name": "test7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    input_dict = {"x": x, "name": "test8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.5, -np.inf, 2.7, np.inf], dtype=np.float32)
    input_dict = {"x": x, "name": "test9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[-np.inf, 0], [0, np.inf]], dtype=np.float64)
    input_dict = {"x": x, "name": "test10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.is_inf"] = tf_math_is_inf_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_lgamma_inputs():
    list_of_inputs = []

    x = np.array([0, 0.5, 1, 4.5, -4, -5.6], dtype=np.float32)
    input_dict = {"x": x, "name": "example1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"x": x, "name": "example2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {"x": x, "name": "example3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([5.0], dtype=np.float16)
    input_dict = {"x": x, "name": "example4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "name": "example5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict = {"x": x, "name": "example6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.1, 1.1, 2.1, 3.1], dtype=np.float32)
    input_dict = {"x": x, "name": "example7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-0.1, -1.1, -2.1, -3.1], dtype=np.float64)
    input_dict = {"x": x, "name": "example8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    input_dict = {"x": x, "name": "example9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float16)
    input_dict = {"x": x, "name": "example10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.lgamma"] = tf_lgamma_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_log1p_inputs():
    list_of_inputs = []

    x = np.array([0.0], dtype=np.float32)
    input_dict = {"x": x, "name": "log1p_0"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "name": "log1p_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-0.5, -1.0, -2.0], dtype=np.float32)
    input_dict = {"x": x, "name": "log1p_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.0, 0.5], [1.0, 2.0]], dtype=np.float64)
    input_dict = {"x": x, "name": "log1p_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0.0, 0.5], [1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]]], dtype=np.float32)
    input_dict = {"x": x, "name": "log1p_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-5, 1e-2, 1e-1], dtype=np.float32)
    input_dict = {"x": x, "name": "log1p_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e5, 1e6, 1e7], dtype=np.float64)
    input_dict = {"x": x, "name": "log1p_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0 + 1j, 1 + 1j, 2 + 1j], dtype=np.complex64)
    input_dict = {"x": x, "name": "log1p_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, 0.5, 1.0], dtype=np.float16)
    input_dict = {"x": x, "name": "log1p_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0, -3.0], dtype=np.complex128)
    input_dict = {"x": x, "name": "log1p_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.log1p"] = tf_math_log1p_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_logical_not_inputs():
    list_of_inputs = []

    x = np.array([True, False])
    input_dict = {"x": tf.constant(x), "name": "test_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([False, True, False])
    input_dict = {"x": tf.constant(x), "name": "test_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[True, False], [False, True]])
    input_dict = {"x": tf.constant(x), "name": "test_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    input_dict = {"x": tf.constant(x), "name": "test_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([True])
    input_dict = {"x": tf.constant(x), "name": "test_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([False])
    input_dict = {"x": tf.constant(x), "name": "test_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[True]])
    input_dict = {"x": tf.constant(x), "name": "test_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[False]])
    input_dict = {"x": tf.constant(x), "name": "test_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([True, True, True])
    input_dict = {"x": tf.constant(x), "name": "test_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([False, False, False])
    input_dict = {"x": tf.constant(x), "name": "test_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[True, True], [False, False]])
    input_dict = {"x": tf.constant(x), "name": "test_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.logical_not"] = tf_math_logical_not_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_maximum_inputs():
    list_of_inputs = []

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([0.0, 2.0, 1.0, 5.0], dtype=np.float32)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_1'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_2'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([3, 2, 1], dtype=np.int32)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_3'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[0, 1], [2, 3]], dtype=np.float32)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_4'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0], dtype=np.float64)
    y = np.array([0.0, 3.0], dtype=np.float64)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_5'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([0, 1, 2], dtype=np.uint8)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_6'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-5, 0, 5], dtype=np.int16)
    y = np.array([-2, 1, 3], dtype=np.int16)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_7'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float16)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_8'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_9'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    y = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    input_dict = {'x': x, 'y': y, 'name': 'maximum_10'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.maximum"] = tf_math_maximum_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_polygamma_inputs():
    list_of_inputs = []

    a = np.float32(1.0)
    x = np.float32(2.0)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.float64(2.0)
    x = np.float64(3.0)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.float32(0.0)
    x = np.float32(1.0)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.float64(3.0)
    x = np.float64(-1.0)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.float32(1.0)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1.0, 2.0], dtype=np.float64)
    x = np.float64(4.0)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.float32(2.0)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x = np.float64(5.0)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.float32(5.0)
    x = np.float32(0.0)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.float64(1.5)
    x = np.float64(2.5)
    input_dict = {"a": tf.convert_to_tensor(a), "x": tf.convert_to_tensor(x), "name": "polygamma_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.polygamma"] = tf_math_polygamma_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_pow_inputs():
    list_of_inputs = []

    x = np.array([[2, 2], [3, 3]], dtype=np.float32)
    y = np.array([[8, 16], [2, 3]], dtype=np.float32)
    name = "pow_example_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([2, 3, 4], dtype=np.int32)
    name = "pow_example_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1, -2, -3], dtype=np.float64)
    y = np.array([2, 3, 4], dtype=np.float64)
    name = "pow_example_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([0.5, 1.0, 1.5], dtype=np.float16)
    name = "pow_example_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    y = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.int64)
    name = "pow_example_7"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0, 1, 2], dtype=np.int32)
    y = np.array([0, 1, 2], dtype=np.int32)
    name = "pow_example_8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([2.5, 3.5, 4.5], dtype=np.float32)
    y = np.array([-1, -2, -3], dtype=np.float32)
    name = "pow_example_9"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2], dtype=np.int64)
    y = np.array([100, 200], dtype=np.int64)
    name = "pow_example_10"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.pow"] = tf_math_pow_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_real_inputs():
    list_of_inputs = []

    input1 = np.array([-2.25 + 4.75j, 3.25 + 5.75j])
    input_dict1 = {"input": tf.constant(input1), "name": "real_part1"}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([1 + 0j, 2 + 0j, 3 + 0j])
    input_dict2 = {"input": tf.constant(input2), "name": "real_part2"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
    input_dict3 = {"input": tf.constant(input3), "name": "real_part3"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-1.5 - 2.5j], [3.5 - 4.5j]])
    input_dict4 = {"input": tf.constant(input4), "name": "real_part4"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([5.0 + 0.0j])
    input_dict5 = {"input": tf.constant(input5), "name": "real_part5"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[-1 + 2j, 3 - 4j], [5 + 6j, -7 - 8j]])
    input_dict6 = {"input": tf.constant(input6), "name": "real_part6"}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1j, 2j, 3j])
    input_dict7 = {"input": tf.constant(input7), "name": "real_part7"}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[[1 + 1j], [2 + 2j]], [[3 + 3j], [4 + 4j]]])
    input_dict8 = {"input": tf.constant(input8), "name": "real_part8"}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([-1.0 - 1.0j, -2.0 - 2.0j])
    input_dict9 = {"input": tf.constant(input9), "name": "real_part9"}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([0 + 1j, 0 - 1j])
    input_dict10 = {"input": tf.constant(input10), "name": "real_part10"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.math.real"] = tf_math_real_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_bessel_j1_inputs():
    list_of_inputs = []

    x = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    input_dict = {"x": x, "name": "bessel_j1_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-0.5, -1.0, -2.0, -4.0], dtype=np.float32)
    input_dict = {"x": x, "name": "bessel_j1_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float64)
    input_dict = {"x": x, "name": "bessel_j1_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0.5, 1.0], [2.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"x": x, "name": "bessel_j1_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0.5], [1.0]], [[2.0], [4.0]]], dtype=np.float64)
    input_dict = {"x": x, "name": "bessel_j1_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10.0], dtype=np.float32)
    input_dict = {"x": x, "name": "bessel_j1_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0], dtype=np.float64)
    input_dict = {"x": x, "name": "bessel_j1_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    input_dict = {"x": x, "name": "bessel_j1_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.5, -2.5, -3.5, -4.5], dtype=np.float64)
    input_dict = {"x": x, "name": "bessel_j1_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "name": "bessel_j1_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.special.bessel_j1"] = tf_bessel_j1_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_squared_difference_inputs():
    list_of_inputs = []
    
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    y = np.array([5, 6, 7, 8], dtype=np.float32)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_1'})

    x = np.array([-1, -2, -3], dtype=np.int32)
    y = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_2'})

    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([[5, 6], [7, 8]], dtype=np.float64)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_3'})

    x = np.array([1j, 2j, 3j], dtype=np.complex64)
    y = np.array([4j, 5j, 6j], dtype=np.complex64)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_4'})

    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_5'})
    
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([4, 5, 6], dtype=np.int64)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_6'})

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    y = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.float32)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_7'})

    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    y = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_8'})
    
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_9'})

    x = np.array([1, 2], dtype=np.int32)
    y = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append({'x': tf.constant(x), 'y': tf.constant(y), 'name': 'squared_diff_10'})
    
    return list_of_inputs

generated_inputs["tf.math.squared_difference"] = tf_math_squared_difference_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_zero_fraction_inputs():
    list_of_inputs = []

    input_dict = {
        "value": np.array([0, 1, 0, 2, 0], dtype=np.float32),
        "name": "example_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": np.array([[0, 1], [2, 0]], dtype=np.int32),
        "name": "example_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": np.array([[-1, 0, 1], [0, -2, 0]], dtype=np.float64),
        "name": "example_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": np.array([], dtype=np.float32),
        "name": "example_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": np.array([1, 2, 3, 4, 5], dtype=np.int64),
        "name": "example_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": np.array([[[0, 0], [1, 0]], [[0, 2], [0, 0]]], dtype=np.float32),
        "name": "example_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": np.array([0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float64),
        "name": "example_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32),
        "name": "example_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "value": np.array([0, 0, 0, 0, 0], dtype=np.float32),
        "name": "example_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "value": np.array([1, 2, 3, 4, 5], dtype=np.float16),
        "name": "example_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.zero_fraction"] = tf_math_zero_fraction_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nest_assert_same_structure_inputs():
    list_of_inputs = []
    input_dict1 = {
        "nest1": np.array([1, 2, 3]),
        "nest2": np.array([4, 5, 6]),
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input_dict2 = {
        "nest1": np.array([[1, 2], [3, 4]]),
        "nest2": np.array([[5, 6], [7, 8]]),
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input_dict3 = {
        "nest1": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        "nest2": np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]),
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input_dict4 = {
        "nest1": np.array([1, 2, 3]),
        "nest2": np.array([[4, 5], [6, 7]]),
        "check_types": False,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input_dict5 = {
        "nest1": np.array([-1, -2, -3]),
        "nest2": np.array([1, 2, 3]),
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input_dict6 = {
        "nest1": np.array([1.0, 2.0, 3.0]),
        "nest2": np.array([4.0, 5.0, 6.0]),
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input_dict7 = {
        "nest1": np.array([1, 2]),
        "nest2": np.array([[3, 4], [5, 6]]),
        "check_types": False,
        "expand_composites": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input_dict8 = {
        "nest1": np.array([]),
        "nest2": np.array([]),
        "check_types": True,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input_dict9 = {
        "nest1": np.array([1]),
        "nest2": np.array([[2]]),
        "check_types": False,
        "expand_composites": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input_dict10 = {
        "nest1": np.array([1, 2, 3]),
        "nest2": np.array([1, 2, 3]),
        "check_types": True,
        "expand_composites": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.nest.assert_same_structure"] = tf_nest_assert_same_structure_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_crelu_inputs():
    list_of_inputs = []

    features = np.array([-1.0, 0.0, 1.0, -2.0, 2.0], dtype=np.float32)
    axis = -1
    name = "crelu_1"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    axis = 0
    name = "crelu_2"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[1, -2], [-3, 4]], [[-5, 6], [7, -8]]], dtype=np.int32)
    axis = 1
    name = "crelu_3"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([10, -20, 30, -40], dtype=np.int64)
    axis = 0
    name = "crelu_4"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[0.5, -0.5], [1.0, -1.0]], dtype=np.float32)
    axis = -1
    name = "crelu_5"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-100, 200], [300, -400]], dtype=np.int16)
    axis = -1
    name = "crelu_6"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([1, -2, 3, -4, 5], dtype=np.int8)
    axis = 0
    name = "crelu_7"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-1.1, 2.2, -3.3], [4.4, -5.5, 6.6]], dtype=np.float32)
    axis = 1
    name = "crelu_8"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]], dtype=np.float64)
    axis = -1
    name = "crelu_9"
    input_dict = {"features": features, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.crelu"] = tf_nn_crelu_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_isotonic_regression_inputs():
    list_of_inputs = []

    inputs1 = np.array([[3, 1, 2], [1, 3, 4]], dtype=np.float32)
    decreasing1 = True
    axis1 = 1
    input_dict1 = {"inputs": inputs1, "decreasing": decreasing1, "axis": axis1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    inputs2 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    decreasing2 = False
    axis2 = 0
    input_dict2 = {"inputs": inputs2, "decreasing": decreasing2, "axis": axis2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    inputs3 = np.array([[-1, -2, -3], [0, 1, 2]], dtype=np.float32)
    decreasing3 = True
    axis3 = 1
    input_dict3 = {"inputs": inputs3, "decreasing": decreasing3, "axis": axis3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    inputs4 = np.array([[5, 4, 3, 2, 1]], dtype=np.float32)
    decreasing4 = True
    axis4 = 0
    input_dict4 = {"inputs": inputs4, "decreasing": decreasing4, "axis": axis4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    inputs5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    decreasing5 = False
    axis5 = 2
    input_dict5 = {"inputs": inputs5, "decreasing": decreasing5, "axis": axis5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    inputs6 = np.array([[0.5, 0.2, 0.8, 0.1]], dtype=np.float32)
    decreasing6 = True
    axis6 = 0
    input_dict6 = {"inputs": inputs6, "decreasing": decreasing6, "axis": axis6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    inputs7 = np.array([[10, 5, 15, 2]], dtype=np.float32)
    decreasing7 = False
    axis7 = 1
    input_dict7 = {"inputs": inputs7, "decreasing": decreasing7, "axis": axis7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    inputs8 = np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=np.float32)
    decreasing8 = True
    axis8 = 1
    input_dict8 = {"inputs": inputs8, "decreasing": decreasing8, "axis": axis8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    inputs9 = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    decreasing9 = False
    axis9 = 0
    input_dict9 = {"inputs": inputs9, "decreasing": decreasing9, "axis": axis9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    inputs10 = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    decreasing10 = True
    axis10 = 0
    input_dict10 = {"inputs": inputs10, "decreasing": decreasing10, "axis": axis10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.nn.isotonic_regression"] = tf_nn_isotonic_regression_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def log_poisson_loss_inputs():
    list_of_inputs = []

    targets = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    log_input = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    compute_full_loss = False
    name = "test1"
    input_dict = {
        "targets": targets,
        "log_input": log_input,
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([1.0, 2.0], dtype=np.float32)
    log_input = np.array([0.1, 0.2], dtype=np.float32)
    compute_full_loss = True
    name = "test2"
    input_dict = {
        "targets": targets,
        "log_input": log_input,
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([1.0], dtype=np.float32)
    log_input = np.array([0.5], dtype=np.float32)
    compute_full_loss = True
    name = "test3"
    input_dict = {
        "targets": targets,
        "log_input": log_input,
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    targets = np.array([1.0, 1.0], dtype=np.float32)
    log_input = np.array([0.5, 0.5], dtype=np.float32)
    compute_full_loss = False
    name = "test4"
    input_dict = {
        "targets": targets,
        "log_input": log_input,
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.log_poisson_loss"] = log_poisson_loss_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_softmax_inputs():
    list_of_inputs = []

    logits = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    axis = -1
    name = "softmax_1"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = 1
    name = "softmax_2"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[-1.0, -2.0, -3.0]], dtype=np.float32)
    axis = 0
    name = "softmax_3"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]], dtype=np.float64)
    axis = -1
    name = "softmax_4"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    axis = 0
    name = "softmax_5"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[-1.0], [-2.0], [-3.0]], dtype=np.float32)
    axis = 0
    name = "softmax_6"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    axis = 2
    name = "softmax_7"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)
    axis = -1
    name = "softmax_8"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    axis = -1
    name = "softmax_9"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    logits = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    axis = 1
    name = "softmax_10"
    input_dict = {"logits": logits, "axis": axis, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.softmax"] = tf_nn_softmax_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_random_stateless_gamma_inputs():
    list_of_inputs = []

    input_dict = {
        'shape': np.array([10, 2], dtype=np.int32),
        'seed': np.array([12, 34], dtype=np.int32),
        'alpha': np.array([0.5, 1.5], dtype=np.float32),
        'beta': np.array([2.0, 3.0], dtype=np.float32),
        'dtype': np.float32,
        'name': 'gamma_example_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'shape': np.array([7, 5, 2], dtype=np.int32),
        'seed': np.array([12, 34], dtype=np.int32),
        'alpha': np.array([0.5, 1.5], dtype=np.float32),
        'beta': np.array([2.0, 3.0], dtype=np.float32),
        'dtype': np.float32,
        'name': 'gamma_example_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'shape': np.array([1, 1], dtype=np.int32),
        'seed': np.array([12, 34], dtype=np.int32),
        'alpha': np.array([1.0], dtype=np.float32),
        'beta': np.array([1.0], dtype=np.float32),
        'dtype': np.float32,
        'name': 'gamma_example_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_gamma"] = tf_random_stateless_gamma_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_random_stateless_parameterized_truncated_normal_inputs():
    list_of_inputs = []

    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([7, 17], dtype=np.int32)
    means = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    stddevs = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    minvals = np.array([-2.0, -1.0, -3.0], dtype=np.float32)
    maxvals = np.array([2.0, 3.0, 1.0], dtype=np.float32)
    name = "test1"
    input_dict = {
        "shape": shape,
        "seed": seed,
        "means": means,
        "stddevs": stddevs,
        "minvals": minvals,
        "maxvals": maxvals,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_parameterized_truncated_normal"] = tf_random_stateless_parameterized_truncated_normal_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_BatchMatMulV3_inputs():
    list_of_inputs = []

    x = np.random.rand(2, 3).astype(np.float32)
    y = np.random.rand(3, 4).astype(np.float32)
    Tout = np.float32
    adj_x = False
    adj_y = False
    grad_x = False
    grad_y = False
    name = "batch_matmul_1"

    input_dict = {
        'Tout': Tout,
        'adj_x': adj_x,
        'adj_y': adj_y,
        'grad_x': grad_x,
        'grad_y': grad_y,
        'name': name,
        'x': x,
        'y': y
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.BatchMatMulV3"] = tf_raw_ops_BatchMatMulV3_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_cosh_inputs():
    list_of_inputs = []

    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")], dtype=np.float32)
    input_dict = {'x': x, 'name': 'cosh_1'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1, 0, 1], dtype=np.float64)
    input_dict = {'x': x, 'name': 'cosh_2'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1+1j, 2-2j, 3+0j], dtype=np.complex64)
    input_dict = {'x': x, 'name': 'cosh_3'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict = {'x': x, 'name': 'cosh_4'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-2.5, 0.0, 2.5], dtype=np.float16)
    input_dict = {'x': x, 'name': 'cosh_5'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-3.14, 0.0, 3.14], dtype=np.half)
    input_dict = {'x': x, 'name': 'cosh_6'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.complex128)
    input_dict = {'x': x, 'name': 'cosh_7'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-100, 0, 100], dtype=np.float32)
    input_dict = {'x': x, 'name': 'cosh_8'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0], dtype=np.float64)
    input_dict = {'x': x, 'name': 'cosh_9'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    input_dict = {'x': x, 'name': 'cosh_10'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Cosh"] = tf_raw_ops_cosh_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_Div_inputs():
    list_of_inputs = []

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    input_dict = {'name': 'div_1', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([2, 4, 6], dtype=np.int32)
    input_dict = {'name': 'div_2', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    y = np.array([2.0, -1.0], dtype=np.float32)
    input_dict = {'name': 'div_3', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)
    y = np.array([1j, 1 + 1j], dtype=np.complex64)
    input_dict = {'name': 'div_4', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([2, 4, 6], dtype=np.uint8)
    input_dict = {'name': 'div_5', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 20, 30], dtype=np.int64)
    y = np.array([5, 10, 15], dtype=np.int64)
    input_dict = {'name': 'div_6', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0], dtype=np.float64)
    y = np.array([3.0, 4.0], dtype=np.float64)
    input_dict = {'name': 'div_7', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    y = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.int32)
    input_dict = {'name': 'div_8', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([100, 200, 300], dtype=np.int32)
    y = np.array([2, 4, 6], dtype=np.int32)
    input_dict = {'name': 'div_9', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    y = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    input_dict = {'name': 'div_10', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Div"] = tf_raw_ops_Div_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_DrawBoundingBoxes_inputs():
    list_of_inputs = []

    images1 = np.random.rand(2, 100, 200, 3).astype(np.float32)
    boxes1 = np.random.rand(2, 5, 4).astype(np.float32)
    name1 = "draw_boxes_1"
    input_dict1 = {'name': name1, 'images': images1, 'boxes': boxes1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    images2 = np.random.rand(1, 50, 100, 1).astype(np.float32)
    boxes2 = np.random.rand(1, 3, 4).astype(np.float32)
    name2 = "draw_boxes_2"
    input_dict2 = {'name': name2, 'images': images2, 'boxes': boxes2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    images3 = np.random.rand(4, 256, 256, 3).astype(np.float16)
    boxes3 = np.random.rand(4, 2, 4).astype(np.float32)
    name3 = "draw_boxes_3"
    input_dict3 = {'name': name3, 'images': images3, 'boxes': boxes3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    images4 = np.random.rand(1, 128, 128, 3).astype(np.float32)
    boxes4 = np.random.rand(1, 10, 4).astype(np.float32)
    name4 = "draw_boxes_4"
    input_dict4 = {'name': name4, 'images': images4, 'boxes': boxes4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    images5 = np.random.rand(3, 64, 64, 3).astype(np.float32)
    boxes5 = np.random.rand(3, 1, 4).astype(np.float32)
    name5 = "draw_boxes_5"
    input_dict5 = {'name': name5, 'images': images5, 'boxes': boxes5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    images6 = np.random.rand(2, 200, 300, 3).astype(np.float32)
    boxes6 = np.random.rand(2, 7, 4).astype(np.float32)
    name6 = "draw_boxes_6"
    input_dict6 = {'name': name6, 'images': images6, 'boxes': boxes6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    images7 = np.random.rand(1, 32, 32, 1).astype(np.float32)
    boxes7 = np.random.rand(1, 4, 4).astype(np.float32)
    name7 = "draw_boxes_7"
    input_dict7 = {'name': name7, 'images': images7, 'boxes': boxes7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    images8 = np.random.rand(5, 100, 100, 3).astype(np.float32)
    boxes8 = np.random.rand(5, 6, 4).astype(np.float32)
    name8 = "draw_boxes_8"
    input_dict8 = {'name': name8, 'images': images8, 'boxes': boxes8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    images9 = np.random.rand(1, 224, 224, 3).astype(np.float32)
    boxes9 = np.random.rand(1, 8, 4).astype(np.float32)
    name9 = "draw_boxes_9"
    input_dict9 = {'name': name9, 'images': images9, 'boxes': boxes9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    images10 = np.random.rand(2, 150, 150, 3).astype(np.float32)
    boxes10 = np.random.rand(2, 3, 4).astype(np.float32)
    name10 = "draw_boxes_10"
    input_dict10 = {'name': name10, 'images': images10, 'boxes': boxes10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.raw_ops.DrawBoundingBoxes"] = tf_raw_ops_DrawBoundingBoxes_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_elu_inputs():
    list_of_inputs = []

    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    name = "elu_1"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    name = "elu_2"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name = "elu_3"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([1.0], dtype=np.float64)
    name = "elu_4"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    name = "elu_6"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[1.0], [-1.0]], [[2.0], [-2.0]]], dtype=np.float32)
    name = "elu_7"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([1000.0, -1000.0], dtype=np.float32)
    name = "elu_8"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([0.5, -0.5, 1.5, -1.5], dtype=np.float32)
    name = "elu_9"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([2.71828, -2.71828], dtype=np.float32)
    name = "elu_10"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    name = "elu_11"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Elu"] = tf_raw_ops_elu_inputs()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_Fact_inputs():
    list_of_inputs = []

    name1 = "fact_op_1"
    input_dict1 = {"name": name1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    name2 = "fact_op_2"
    input_dict2 = {"name": name2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    name3 = "fact_op_3"
    input_dict3 = {"name": name3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    name4 = "fact_op_4"
    input_dict4 = {"name": name4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    name5 = "fact_op_5"
    input_dict5 = {"name": name5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    name6 = "fact_op_6"
    input_dict6 = {"name": name6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    name7 = "fact_op_7"
    input_dict7 = {"name": name7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    name8 = "fact_op_8"
    input_dict8 = {"name": name8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    name9 = "fact_op_9"
    input_dict9 = {"name": name9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    name10 = "fact_op_10"
    input_dict10 = {"name": name10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Fact"] = tf_raw_ops_Fact_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_gather_inputs():
    list_of_inputs = []

    params1 = np.array([[1, 2], [3, 4], [5, 6]])
    indices1 = np.array([0, 1, 2], dtype=np.int32)
    validate_indices1 = True
    name1 = "gather_test_1"

    input_dict1 = {
        "params": params1,
        "indices": indices1,
        "validate_indices": validate_indices1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    params2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    indices2 = np.array([[0, 1], [1, 0]], dtype=np.int64)
    validate_indices2 = False
    name2 = "gather_test_2"

    input_dict2 = {
        "params": params2,
        "indices": indices2,
        "validate_indices": validate_indices2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    params3 = np.array([10, 20, 30, 40])
    indices3 = np.array([2, 0, 1], dtype=np.int32)
    validate_indices3 = True
    name3 = "gather_test_3"

    input_dict3 = {
        "params": params3,
        "indices": indices3,
        "validate_indices": validate_indices3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    params4 = np.array([[1, 2, 3], [4, 5, 6]])
    indices4 = np.array([0, 0, 1], dtype=np.int64)
    validate_indices4 = False
    name4 = "gather_test_4"

    input_dict4 = {
        "params": params4,
        "indices": indices4,
        "validate_indices": validate_indices4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    params5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    indices5 = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]], dtype=np.int64)
    validate_indices5 = True
    name5 = "gather_test_5"

    input_dict5 = {
        "params": params5,
        "indices": indices5,
        "validate_indices": validate_indices5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    params6 = np.array([[1, 2], [3, 4]])
    indices6 = np.array([0, 1], dtype=np.int32)
    validate_indices6 = False
    name6 = "gather_test_6"

    input_dict6 = {
        "params": params6,
        "indices": indices6,
        "validate_indices": validate_indices6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    params7 = np.array([1, 2, 3])
    indices7 = np.array([0], dtype=np.int64)
    validate_indices7 = True
    name7 = "gather_test_7"

    input_dict7 = {
        "params": params7,
        "indices": indices7,
        "validate_indices": validate_indices7,
        "name": name7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    params8 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indices8 = np.array([1, 1, 1], dtype=np.int32)
    validate_indices8 = False
    name8 = "gather_test_8"

    input_dict8 = {
        "params": params8,
        "indices": indices8,
        "validate_indices": validate_indices8,
        "name": name8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["tf.raw_ops.Gather"] = tf_raw_ops_gather_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_Greater_inputs():
    list_of_inputs = []

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([3, 2, 1], dtype=np.int32)
    input_dict = {'name': 'greater_1', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([5.0, 4.0, 6.0], dtype=np.float32)
    y = np.array([5.0, 2.0, 5.0], dtype=np.float32)
    input_dict = {'name': 'greater_2', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([5, 4, 6], dtype=np.int64)
    y = np.array([5], dtype=np.int64)
    input_dict = {'name': 'greater_3', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([[2, 1], [4, 3]], dtype=np.float64)
    input_dict = {'name': 'greater_4', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, -2, 3], dtype=np.int32)
    y = np.array([0, -3, 2], dtype=np.int32)
    input_dict = {'name': 'greater_5', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    y = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input_dict = {'name': 'greater_6', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([0, 1, 2], dtype=np.uint8)
    input_dict = {'name': 'greater_7', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 20, 30], dtype=np.int16)
    y = np.array([5, 15, 25], dtype=np.int16)
    input_dict = {'name': 'greater_8', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([0, 1, 2], dtype=np.float16)
    input_dict = {'name': 'greater_9', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Greater"] = tf_raw_ops_Greater_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_L2Loss_inputs():
    list_of_inputs = []

    input_dict = {
        "name": "L2Loss_1",
        "t": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "L2Loss_2",
        "t": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "L2Loss_3",
        "t": np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "L2Loss_4",
        "t": np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "L2Loss_5",
        "t": np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "L2Loss_6",
        "t": np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "L2Loss_7",
        "t": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "L2Loss_8",
        "t": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "name": "L2Loss_9",
        "t": np.array([0.0, 0.0, 0.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.L2Loss"] = tf_raw_ops_L2Loss_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_log_inputs():
    list_of_inputs = []

    x = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {'x': x, 'name': 'log1'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    input_dict = {'x': x, 'name': 'log2'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict = {'x': x, 'name': 'log3'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    input_dict = {'x': x, 'name': 'log4'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 20, 30], dtype=np.float16)
    input_dict = {'x': x, 'name': 'log5'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.001, 0.01, 0.1], dtype=np.float32)
    input_dict = {'x': x, 'name': 'log6'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e9, 1e10, 1e11], dtype=np.float64)
    input_dict = {'x': x, 'name': 'log7'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.5, 2.5, 3.5], dtype=np.half)
    input_dict = {'x': x, 'name': 'log8'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1+0j, 2+0j, 3+0j], dtype=np.complex128)
    input_dict = {'x': x, 'name': 'log9'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Log"] = tf_raw_ops_log_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_maximum_inputs():
    list_of_inputs = []
    
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([0.0, 2.0, 1.0, 5.0], dtype=np.float32)
    input_dict = {'name': 'maximum_1', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {'name': 'maximum_2', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([3, 2, 1], dtype=np.int32)
    input_dict = {'name': 'maximum_3', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([[0, 1], [2, 3]], dtype=np.float64)
    input_dict = {'name': 'maximum_4', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([3.0], dtype=np.float32)
    input_dict = {'name': 'maximum_5', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([2, 1, 4], dtype=np.uint8)
    input_dict = {'name': 'maximum_6', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([10, 20, 30], dtype=np.int16)
    y = np.array([5, 15, 25], dtype=np.int16)
    input_dict = {'name': 'maximum_7', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float16)
    input_dict = {'name': 'maximum_8', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([4, 5, 6], dtype=np.int64)
    input_dict = {'name': 'maximum_9', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Maximum"] = tf_raw_ops_maximum_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_mean_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    axis1 = np.array([0], dtype=np.int32)
    keep_dims1 = True
    name1 = "mean_1"
    
    input_dict1 = {
        "input": input1,
        "axis": axis1,
        "keep_dims": keep_dims1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    axis2 = np.array([1], dtype=np.int64)
    keep_dims2 = False
    name2 = "mean_2"
    
    input_dict2 = {
        "input": input2,
        "axis": axis2,
        "keep_dims": keep_dims2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    axis3 = np.array([0, 1], dtype=np.int32)
    keep_dims3 = True
    name3 = "mean_3"
    
    input_dict3 = {
        "input": input3,
        "axis": axis3,
        "keep_dims": keep_dims3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
    axis4 = np.array([0, 2], dtype=np.int64)
    keep_dims4 = False
    name4 = "mean_4"
    
    input_dict4 = {
        "input": input4,
        "axis": axis4,
        "keep_dims": keep_dims4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    axis5 = np.array([0], dtype=np.int32)
    keep_dims5 = False
    name5 = "mean_5"
    
    input_dict5 = {
        "input": input5,
        "axis": axis5,
        "keep_dims": keep_dims5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int16)
    axis6 = np.array([0], dtype=np.int64)
    keep_dims6 = True
    name6 = "mean_6"
    
    input_dict6 = {
        "input": input6,
        "axis": axis6,
        "keep_dims": keep_dims6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis7 = np.array([1], dtype=np.int32)
    keep_dims7 = False
    name7 = "mean_7"

    input_dict7 = {
        "input": input7,
        "axis": axis7,
        "keep_dims": keep_dims7,
        "name": name7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    axis8 = np.array([0], dtype=np.int64)
    keep_dims8 = True
    name8 = "mean_8"

    input_dict8 = {
        "input": input8,
        "axis": axis8,
        "keep_dims": keep_dims8,
        "name": name8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([[1, 2], [3, 4]], dtype=np.float16)
    axis9 = np.array([0], dtype=np.int32)
    keep_dims9 = False
    name9 = "mean_9"
    
    input_dict9 = {
        "input": input9,
        "axis": axis9,
        "keep_dims": keep_dims9,
        "name": name9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float16)
    axis10 = np.array([1], dtype=np.int64)
    keep_dims10 = True
    name10 = "mean_10"

    input_dict10 = {
        "input": input10,
        "axis": axis10,
        "keep_dims": keep_dims10,
        "name": name10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    

    return list_of_inputs

generated_inputs["tf.raw_ops.Mean"] = tf_raw_ops_mean_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_pow_inputs():
    list_of_inputs = []

    x = np.array([[2, 2], [3, 3]], dtype=np.float32)
    y = np.array([[8, 16], [2, 3]], dtype=np.float32)
    input_dict = {'name': 'pow_1', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([2, 3, 4], dtype=np.int32)
    input_dict = {'name': 'pow_2', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1, 2], dtype=np.float64)
    y = np.array([3, 2], dtype=np.float64)
    input_dict = {'name': 'pow_3', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    y = np.array([[2, 3], [4, 5]], dtype=np.complex64)
    input_dict = {'name': 'pow_4', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.5, 1.5], dtype=np.float32)
    y = np.array([2, 0.5], dtype=np.float32)
    input_dict = {'name': 'pow_5', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10], dtype=np.int8)
    y = np.array([2], dtype=np.int8)
    input_dict = {'name': 'pow_6', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([2.0], dtype=np.float64)
    y = np.array([-1.0], dtype=np.float64)
    input_dict = {'name': 'pow_7', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3], dtype=np.int16)
    y = np.array([1, 1, 1], dtype=np.int16)
    input_dict = {'name': 'pow_8', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0], dtype=np.float32)
    y = np.array([0], dtype=np.float32)
    input_dict = {'name': 'pow_9', 'x': x, 'y': y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Pow"] = tf_raw_ops_pow_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_prod_inputs():
    list_of_inputs = []

    input1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis1 = np.array([0], dtype=np.int32)
    keep_dims1 = True
    name1 = "prod_1"
    input_dict1 = {
        'keep_dims': keep_dims1,
        'name': name1,
        'input': input1,
        'axis': axis1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([1, 2, 3, 4], dtype=np.int32)
    axis2 = np.array([0], dtype=np.int64)
    keep_dims2 = False
    name2 = "prod_2"
    input_dict2 = {
        'keep_dims': keep_dims2,
        'name': name2,
        'input': input2,
        'axis': axis2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    axis3 = np.array([1, 2], dtype=np.int32)
    keep_dims3 = True
    name3 = "prod_3"
    input_dict3 = {
        'keep_dims': keep_dims3,
        'name': name3,
        'input': input3,
        'axis': axis3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.int32)
    axis4 = np.array([0], dtype=np.int64)
    keep_dims4 = False
    name4 = "prod_4"
    input_dict4 = {
        'keep_dims': keep_dims4,
        'name': name4,
        'input': input4,
        'axis': axis4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1, 2, 3], dtype=np.uint8)
    axis5 = np.array([0], dtype=np.int32)
    keep_dims5 = False
    name5 = "prod_5"
    input_dict5 = {
        'keep_dims': keep_dims5,
        'name': name5,
        'input': input5,
        'axis': axis5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1, 2], [3, 4]], dtype=np.complex64)
    axis6 = np.array([1], dtype=np.int32)
    keep_dims6 = True
    name6 = "prod_6"
    input_dict6 = {
        'keep_dims': keep_dims6,
        'name': name6,
        'input': input6,
        'axis': axis6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    axis7 = np.array([0], dtype=np.int64)
    keep_dims7 = False
    name7 = "prod_7"
    input_dict7 = {
        'keep_dims': keep_dims7,
        'name': name7,
        'input': input7,
        'axis': axis7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
    axis8 = np.array([1], dtype=np.int32)
    keep_dims8 = False
    name8 = "prod_8"
    input_dict8 = {
        'keep_dims': keep_dims8,
        'name': name8,
        'input': input8,
        'axis': axis8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis9 = np.array([0, 1], dtype=np.int32)
    keep_dims9 = True
    name9 = "prod_9"
    input_dict9 = {
        'keep_dims': keep_dims9,
        'name': name9,
        'input': input9,
        'axis': axis9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    axis10 = np.array([0], dtype=np.int32)
    keep_dims10 = True
    name10 = "prod_10"
    input_dict10 = {
        'keep_dims': keep_dims10,
        'name': name10,
        'input': input10,
        'axis': axis10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.raw_ops.Prod"] = tf_raw_ops_prod_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_randomshuffle_inputs():
    list_of_inputs = []

    input_dict = {
        'value': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        'seed': 42,
        'seed2': 100,
        'name': 'shuffle_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'value': np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32),
        'seed': 42,
        'seed2': 0,
        'name': 'shuffle_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'value': np.array([1, 2, 3, 4, 5], dtype=np.int64),
        'seed': 42,
        'seed2': 5,
        'name': 'shuffle_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.RandomShuffle"] = tf_raw_ops_randomshuffle_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_real_inputs():
    list_of_inputs = []

    input1 = np.array([-2.25 + 4.75j, 3.25 + 5.75j], dtype=np.complex64)
    input_dict1 = {'input': input1, 'Tout': np.float32, 'name': 'real_part_1'}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([1.0 - 1.0j, 2.0 + 2.0j, -3.0 - 3.0j], dtype=np.complex128)
    input_dict2 = {'input': input2, 'Tout': np.float64, 'name': 'real_part_2'}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1.0 + 2.0j], [3.0 - 4.0j]], dtype=np.complex64)
    input_dict3 = {'input': input3, 'Tout': np.float32, 'name': 'real_part_3'}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[[0.5 + 0.5j], [1.5 - 1.5j]], [[2.5 + 2.5j], [-3.5 - 3.5j]]], dtype=np.complex128)
    input_dict4 = {'input': input4, 'Tout': np.float64, 'name': 'real_part_4'}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([-10.0 + 5.0j], dtype=np.complex64)
    input_dict5 = {'input': input5, 'Tout': np.float32, 'name': 'real_part_5'}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([100.0 - 50.0j, -200.0 + 100.0j], dtype=np.complex128)
    input_dict6 = {'input': input6, 'Tout': np.float64, 'name': 'real_part_6'}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j], dtype=np.complex64)
    input_dict7 = {'input': input7, 'Tout': np.float32, 'name': 'real_part_7'}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[-1.0 - 1.0j, 2.0 + 2.0j], [-3.0 - 3.0j, 4.0 + 4.0j]], dtype=np.complex128)
    input_dict8 = {'input': input8, 'Tout': np.float64, 'name': 'real_part_8'}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([0.0 + 1.0j], dtype=np.complex64)
    input_dict9 = {'input': input9, 'Tout': np.float32, 'name': 'real_part_9'}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([[-5.5 + 2.2j, 1.1 - 3.3j], [4.4 + 0.5j, -2.2 - 1.1j]], dtype=np.complex128)
    input_dict10 = {'input': input10, 'Tout': np.float64, 'name': 'real_part_10'}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.raw_ops.Real"] = tf_raw_ops_real_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_round_inputs():
    list_of_inputs = []
    
    x = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    input_dict = {'x': x, 'name': 'round_test_1'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-1.5, -2.5, -3.5, -4.5], dtype=np.float64)
    input_dict = {'x': x, 'name': 'round_test_2'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1.2, 2.7], [3.1, 4.9]], dtype=np.float32)
    input_dict = {'x': x, 'name': 'round_test_3'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]], dtype=np.float64)
    input_dict = {'x': x, 'name': 'round_test_4'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.int32)
    input_dict = {'x': x, 'name': 'round_test_5'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float16)
    input_dict = {'x': x, 'name': 'round_test_7'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.half)
    input_dict = {'x': x, 'name': 'round_test_8'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Round"] = tf_raw_ops_round_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_Selu_inputs():
    list_of_inputs = []

    features = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    name = "selu_test_1"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    name = "selu_test_2"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[1.0, -2.0], [3.0, 4.0]], [[-5.0, 6.0], [7.0, -8.0]]], dtype=np.float32)
    name = "selu_test_3"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([1.5, -2.5, 3.5], dtype=np.float32)
    name = "selu_test_4"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-0.5], [1.5]], dtype=np.float64)
    name = "selu_test_5"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([0.0, 0.0, 0.0], dtype=np.half)
    name = "selu_test_6"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-10.0, 5.0, -2.0], [8.0, -1.0, 3.0]], dtype=np.float32)
    name = "selu_test_7"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    name = "selu_test_8"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-1.1, 0.0, 1.1, -2.2, 3.3]], dtype=np.float32)
    name = "selu_test_9"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-1.0, 0.5, 2.0], [3.0, -1.5, 0.0]], dtype=np.float32)
    name = "selu_test_10"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Selu"] = tf_raw_ops_Selu_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_Sinh_inputs():
    list_of_inputs = []

    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")], dtype=np.float32)
    input_dict = {'x': x, 'name': 'sinh_test_1'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.5, 0, 2.7, -3.1], dtype=np.float64)
    input_dict = {'x': x, 'name': 'sinh_test_2'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1+1j, 2-2j, -1-1j], dtype=np.complex64)
    input_dict = {'x': x, 'name': 'sinh_test_3'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    input_dict = {'x': x, 'name': 'sinh_test_4'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, 2.0, 3.0], dtype=np.half)
    input_dict = {'x': x, 'name': 'sinh_test_6'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
    input_dict = {'x': x, 'name': 'sinh_test_7'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    input_dict = {'x': x, 'name': 'sinh_test_8'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-100, 0, 100], dtype=np.float64)
    input_dict = {'x': x, 'name': 'sinh_test_9'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Sinh"] = tf_raw_ops_Sinh_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_softplus_inputs():
    list_of_inputs = []

    features = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    name = "softplus_1"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    name = "softplus_2"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]], dtype=np.float32)
    name = "softplus_3"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([-1.0, -2.0, -3.0], dtype=np.float16)
    name = "softplus_4"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([1.5, 2.5, 3.5], dtype=np.half)
    name = "softplus_5"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    name = "softplus_6"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-1.0], [0.0], [1.0]], dtype=np.float64)
    name = "softplus_7"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([0.0], dtype=np.float16)
    name = "softplus_8"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([1.0, 2.0, 3.0], dtype=np.half)
    name = "softplus_9"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-2.0, 1.0], [3.0, -1.0]], dtype=np.float32)
    name = "softplus_10"
    input_dict = {"features": features, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Softplus"] = tf_raw_ops_softplus_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_SparseSegmentMean_inputs():
    list_of_inputs = []

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    sparse_gradient = False
    name = "test1"
    input_dict = {
        'sparse_gradient': sparse_gradient,
        'name': name,
        'data': data,
        'indices': indices,
        'segment_ids': segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float64)
    indices = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    sparse_gradient = True
    name = "test2"
    input_dict = {
        'sparse_gradient': sparse_gradient,
        'name': name,
        'data': data,
        'indices': indices,
        'segment_ids': segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float16)
    indices = np.array([0, 1, 0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    sparse_gradient = False
    name = "test3"
    input_dict = {
        'sparse_gradient': sparse_gradient,
        'name': name,
        'data': data,
        'indices': indices,
        'segment_ids': segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
    sparse_gradient = True
    name = "test4"
    input_dict = {
        'sparse_gradient': sparse_gradient,
        'name': name,
        'data': data,
        'indices': indices,
        'segment_ids': segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    sparse_gradient = False
    name = "test5"
    input_dict = {
        'sparse_gradient': sparse_gradient,
        'name': name,
        'data': data,
        'indices': indices,
        'segment_ids': segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSegmentMean"] = tf_raw_ops_SparseSegmentMean_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_SparseSegmentSum_inputs():
    list_of_inputs = []

    data1 = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=np.float32)
    indices1 = np.array([0, 1], dtype=np.int32)
    segment_ids1 = np.array([0, 0], dtype=np.int32)
    sparse_gradient1 = False
    name1 = "test1"

    input_dict1 = {
        "data": data1,
        "indices": indices1,
        "segment_ids": segment_ids1,
        "sparse_gradient": sparse_gradient1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    data2 = np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=np.int32)
    indices2 = np.array([0, 1, 2], dtype=np.int32)
    segment_ids2 = np.array([0, 0, 1], dtype=np.int32)
    sparse_gradient2 = True
    name2 = "test2"

    input_dict2 = {
        "data": data2,
        "indices": indices2,
        "segment_ids": segment_ids2,
        "sparse_gradient": sparse_gradient2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    data3 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    indices3 = np.array([0, 1], dtype=np.int32)
    segment_ids3 = np.array([0, 1], dtype=np.int32)
    sparse_gradient3 = False
    name3 = "test3"

    input_dict3 = {
        "data": data3,
        "indices": indices3,
        "segment_ids": segment_ids3,
        "sparse_gradient": sparse_gradient3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    data4 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    indices4 = np.array([0, 1], dtype=np.int32)
    segment_ids4 = np.array([0, 1], dtype=np.int32)
    sparse_gradient4 = False
    name4 = "test4"

    input_dict4 = {
        "data": data4,
        "indices": indices4,
        "segment_ids": segment_ids4,
        "sparse_gradient": sparse_gradient4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    data5 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    indices5 = np.array([0, 1], dtype=np.int32)
    segment_ids5 = np.array([0, 1], dtype=np.int32)
    sparse_gradient5 = True
    name5 = "test5"

    input_dict5 = {
        "data": data5,
        "indices": indices5,
        "segment_ids": segment_ids5,
        "sparse_gradient": sparse_gradient5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    data6 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices6 = np.array([0, 1], dtype=np.int32)
    segment_ids6 = np.array([0, 1], dtype=np.int32)
    sparse_gradient6 = False
    name6 = "test6"

    input_dict6 = {
        "data": data6,
        "indices": indices6,
        "segment_ids": segment_ids6,
        "sparse_gradient": sparse_gradient6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSegmentSum"] = tf_raw_ops_SparseSegmentSum_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_tan_inputs():
    list_of_inputs = []

    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")], dtype=np.float32)
    input_dict = {'name': 'tan_1', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], dtype=np.float64)
    input_dict = {'name': 'tan_2', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1, 0, 1], [-2, -3, -4]], dtype=np.float32)
    input_dict = {'name': 'tan_3', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1+1j, 2-2j, 3+0j], dtype=np.complex64)
    input_dict = {'name': 'tan_4', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1.5, 2.7], [3.1, -4.8]], dtype=np.float64)
    input_dict = {'name': 'tan_5', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float16)
    input_dict = {'name': 'tan_6', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input_dict = {'name': 'tan_7', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0, np.pi/2], [np.pi, 3*np.pi/2]]], dtype=np.float32)
    input_dict = {'name': 'tan_8', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-np.pi, -np.pi/2, -np.pi/4], dtype=np.float64)
    input_dict = {'name': 'tan_9', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.complex128)
    input_dict = {'name': 'tan_10', 'x': x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Tan"] = tf_raw_ops_tan_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_UnicodeScript_inputs():
    list_of_inputs = []

    input1 = np.array([1, 31, 38], dtype=np.int32)
    input_dict1 = {'name': 'test1', 'input': input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([65, 97, 122], dtype=np.int32)
    input_dict2 = {'name': 'test2', 'input': input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0, 65535, 10000], dtype=np.int32)
    input_dict3 = {'name': 'test3', 'input': input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([-1, 256, 512], dtype=np.int32)
    input_dict4 = {'name': 'test4', 'input': input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict5 = {'name': 'test5', 'input': input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict6 = {'name': 'test6', 'input': input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    input_dict7 = {'name': 'test7', 'input': input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1000, 2000, 3000], dtype=np.int32)
    input_dict8 = {'name': 'test8', 'input': input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([66, 67, 68, 69], dtype=np.int32)
    input_dict9 = {'name': 'test9', 'input': input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict10 = {'name': 'test10', 'input': input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.raw_ops.UnicodeScript"] = tf_raw_ops_UnicodeScript_inputs()


import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_Where_inputs():
    list_of_inputs = []

    condition = np.array([[True, False], [True, False]], dtype=np.bool_)
    input_dict = {'name': 'where_test_1', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([[[True, False], [True, False]], [[False, True], [False, True]]], dtype=np.bool_)
    input_dict = {'name': 'where_test_2', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([[[1.5, 0.0], [-0.5, 0.0]], [[0.0, 0.25], [0.0, 0.75]]], dtype=np.float32)
    input_dict = {'name': 'where_test_3', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([[[1.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.5j, 0.0 + 0.0j]], [[0.0 + 0.0j, 0.25 + 1.5j], [0.0 + 0.0j, 0.75 + 0.0j]]], dtype=np.complex64)
    input_dict = {'name': 'where_test_4', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([[1, 2, 3], [4, 0, 6]], dtype=np.int32)
    input_dict = {'name': 'where_test_5', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([[-1, -2, -3], [-4, 0, -6]], dtype=np.int32)
    input_dict = {'name': 'where_test_6', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float64)
    input_dict = {'name': 'where_test_7', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([[[True, True], [False, False]], [[True, False], [False, True]]], dtype=np.bool_)
    input_dict = {'name': 'where_test_8', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([1, 0, 1, 0, 1], dtype=np.int64)
    input_dict = {'name': 'where_test_9', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([[[True]], [[False]]], dtype=np.bool_)
    input_dict = {'name': 'where_test_10', 'condition': condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Where"] = tf_raw_ops_Where_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_WriteFile_inputs():
    list_of_inputs = []

    input_dict = {
        'name': 'string',
        'filename': np.array("test1.txt", dtype=np.str_),
        'contents': np.array("Hello, world!", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("/tmp/test2.txt", dtype=np.str_),
        'contents': np.array("Another test!", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("test3.txt", dtype=np.str_),
        'contents': np.array("This is a longer string to test file writing.", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("test4.txt", dtype=np.str_),
        'contents': np.array("1234567890", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("test5.txt", dtype=np.str_),
        'contents': np.array("", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("test6.txt", dtype=np.str_),
        'contents': np.array("Special characters: !@#$%^&*()_+=-`~[]\{}|;':\",./<>?", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("test7.txt", dtype=np.str_),
        'contents': np.array("Unicode test: こんにちは世界", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("test8.txt", dtype=np.str_),
        'contents': np.array("Newline test:\nThis is a new line.", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("test9.txt", dtype=np.str_),
        'contents': np.array("Tab test:\tThis is a tab.", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'name': 'string',
        'filename': np.array("test10.txt", dtype=np.str_),
        'contents': np.array("Number test: 123.456", dtype=np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.WriteFile"] = tf_raw_ops_WriteFile_inputs()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_reverse_inputs():
    list_of_inputs = []

    tensor1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis1 = np.array([0], dtype=np.int32)
    name1 = "reverse_1"
    input_dict1 = {"tensor": tensor1, "axis": axis1, "name": name1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    tensor2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis2 = np.array([1], dtype=np.int32)
    name2 = "reverse_2"
    input_dict2 = {"tensor": tensor2, "axis": axis2, "name": name2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    tensor3 = np.array([1, 2, 3, 4], dtype=np.int32)
    axis3 = np.array([0], dtype=np.int32)
    name3 = "reverse_3"
    input_dict3 = {"tensor": tensor3, "axis": axis3, "name": name3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    tensor5 = np.array([[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [12, 13, 14, 15]]], dtype=np.int32)
    axis5 = np.array([-1], dtype=np.int32)
    name5 = "reverse_5"
    input_dict5 = {"tensor": tensor5, "axis": axis5, "name": name5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    tensor6 = np.array([[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [12, 13, 14, 15]]], dtype=np.int32)
    axis6 = np.array([0, 2], dtype=np.int32)
    name6 = "reverse_6"
    input_dict6 = {"tensor": tensor6, "axis": axis6, "name": name6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    tensor7 = np.array([1, 2, 3], dtype=np.float32)
    axis7 = np.array([0], dtype=np.int32)
    name7 = "reverse_7"
    input_dict7 = {"tensor": tensor7, "axis": axis7, "name": name7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    tensor8 = np.array([[True, False], [False, True]], dtype=np.bool_)
    axis8 = np.array([1], dtype=np.int32)
    name8 = "reverse_8"
    input_dict8 = {"tensor": tensor8, "axis": axis8, "name": name8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    tensor9 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    axis9 = np.array([0, 1], dtype=np.int64)
    name9 = "reverse_9"
    input_dict9 = {"tensor": tensor9, "axis": axis9, "name": name9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs["tf.reverse"] = tf_reverse_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sets_union_inputs():
    list_of_inputs = []

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[2, 3], [4, 5]])
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[-1, -2], [-3, -4]])
    b = np.array([[-2, -3], [-4, -5]])
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    b = np.array([[[3, 4, 5], [6, 7, 8]], [[9, 10, 11], [12, 13, 14]]])
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2, 3]])
    b = np.array([[4, 5, 6]])
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[1, 2], [3, 4]])
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    b = np.array([[3, 4, 5, 6], [7, 8, 9, 10]])
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1], [2]], [[3], [4]]])
    b = np.array([[[2], [3]], [[4], [5]]])
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    validate_indices = False
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    b = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    validate_indices = True
    input_dict = {"a": a, "b": b, "validate_indices": validate_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sets.union"] = tf_sets_union_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_tensor_inputs():
    list_of_inputs = []

    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    dense_shape = np.array([3, 4], dtype=np.int64)
    input_dict = {
        "indices": tf.constant(indices),
        "values": tf.constant(values),
        "dense_shape": tf.constant(dense_shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    values = np.array([3.14, 2.71], dtype=np.float32)
    dense_shape = np.array([2, 2, 2], dtype=np.int64)
    input_dict = {
        "indices": tf.constant(indices),
        "values": tf.constant(values),
        "dense_shape": tf.constant(dense_shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0], [2], [4]], dtype=np.int64)
    values = np.array([-1, -2, -3], dtype=np.int32)
    dense_shape = np.array([5], dtype=np.int64)
    input_dict = {
        "indices": tf.constant(indices),
        "values": tf.constant(values),
        "dense_shape": tf.constant(dense_shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    values = np.array([1, 2, 3, 4], dtype=np.int64)
    dense_shape = np.array([2, 2], dtype=np.int64)
    input_dict = {
        "indices": tf.constant(indices),
        "values": tf.constant(values),
        "dense_shape": tf.constant(dense_shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    dense_shape = np.array([2, 2, 2], dtype=np.int64)
    input_dict = {
        "indices": tf.constant(indices),
        "values": tf.constant(values),
        "dense_shape": tf.constant(dense_shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    values = np.array([True, False], dtype=bool)
    dense_shape = np.array([2, 2], dtype=np.int64)
    input_dict = {
        "indices": tf.constant(indices),
        "values": tf.constant(values),
        "dense_shape": tf.constant(dense_shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 0, 0]], dtype=np.int64)
    values = np.array([12345], dtype=np.int32)
    dense_shape = np.array([1, 1, 1], dtype=np.int64)
    input_dict = {
        "indices": tf.constant(indices),
        "values": tf.constant(values),
        "dense_shape": tf.constant(dense_shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    values = np.array([1, 2, 3], dtype=np.int64)
    dense_shape = np.array([2, 3], dtype=np.int64)
    input_dict = {
        "indices": tf.constant(indices),
        "values": tf.constant(values),
        "dense_shape": tf.constant(dense_shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 0, 0], [1, 1, 0]], dtype=np.int64)
    values = np.array([1.1, 2.2], dtype=np.float32)
    dense_shape = np.array([2, 2, 2], dtype=np.int64)
    input_dict = {
        "indices": tf.constant(indices),
        "values": tf.constant(values),
        "dense_shape": tf.constant(dense_shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 0], [1, 0]], dtype=np.int64)
    values = np.array([10, 20], dtype=np.int64)
    dense_shape = np.array([3, 2], dtype=np.int64)
    input_dict = {
        "indices": tf.constant(indices),
        "values": tf.constant(values),
        "dense_shape": tf.constant(dense_shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.SparseTensor"] = tf_sparse_tensor_inputs()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_train_coordinator_inputs():
    list_of_inputs = []
    input_dict_1 = {"clean_stop_exception_types": [Exception]}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    return list_of_inputs

generated_inputs["tf.train.Coordinator"] = tf_train_coordinator_inputs()

