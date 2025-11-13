generated_inputs = {}
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

