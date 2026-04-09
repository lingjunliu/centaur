generated_inputs = {}import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_ToBool_inputs():
    list_of_inputs = []

    # Input 1: 0D tensor, False
    input_tensor = np.array(0, dtype=np.int32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0D tensor, True
    input_tensor = np.array(1, dtype=np.int32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: >0D tensor, empty, False
    input_tensor = np.array([], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: >0D tensor, non-empty, True
    input_tensor = np.array([1, 2, 3], dtype=np.float32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D tensor, False, string - Removed since tf doesn't like bytes_
    # input_tensor = np.array(b"", dtype=np.bytes_)
    # input_dict = {"input": input_tensor, "name": None}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D tensor, True, string - Removed since tf doesn't like bytes_
    # input_tensor = np.array(b"test", dtype=np.bytes_)
    # input_dict = {"input": input_tensor, "name": None}
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: >0D tensor, multi-dimensional, non-empty, True
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: >0D tensor, multi-dimensional, empty, False
    input_tensor = np.array([[]], dtype=np.int64)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 0D tensor, negative value, True
    input_tensor = np.array(-1, dtype=np.int32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: >0D tensor, non-empty, with zeros, True
    input_tensor = np.array([0, 1, 2], dtype=np.int32)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.raw_ops.ToBool"] = tf_raw_ops_ToBool_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sigmoid_grad_inputs():
    list_of_inputs = []

    # Input 1: Simple float32
    y = np.array([0.5], dtype=np.float32)
    dy = np.array([1.0], dtype=np.float32)
    input_dict = {"y": y, "dy": dy, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 with negative values
    y = np.array([-0.2, 0.8, -0.5], dtype=np.float64)
    dy = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    input_dict = {"y": y, "dy": dy, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: bfloat16 (needs conversion)
    y = np.array([0.3, 0.7], dtype=np.float32).astype(np.float16)
    dy = np.array([0.4, 0.6], dtype=np.float32).astype(np.float16)
    input_dict = {"y": y, "dy": dy, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: half (needs conversion)
    y = np.array([0.1, 0.9], dtype=np.float32).astype(np.float16)
    dy = np.array([0.6, -0.4], dtype=np.float32).astype(np.float16)
    input_dict = {"y": y, "dy": dy, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64
    y = np.array([0.2 + 0.1j, 0.8 - 0.3j], dtype=np.complex64)
    dy = np.array([0.3 - 0.2j, -0.1 + 0.4j], dtype=np.complex64)
    input_dict = {"y": y, "dy": dy, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128
    y = np.array([0.4 + 0.2j, 0.6 - 0.5j], dtype=np.complex128)
    dy = np.array([-0.2 + 0.3j, 0.5 - 0.1j], dtype=np.complex128)
    input_dict = {"y": y, "dy": dy, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32
    y = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    dy = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    input_dict = {"y": y, "dy": dy, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32
    y = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    dy = np.array([[[0.9, 1.0], [1.1, 1.2]], [[1.3, 1.4], [1.5, 1.6]]], dtype=np.float32)
    input_dict = {"y": y, "dy": dy, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, name provided
    y = np.array([0.9, 0.1], dtype=np.float32)
    dy = np.array([0.2, 0.8], dtype=np.float32)
    input_dict = {"y": y, "dy": dy, "name": "my_sigmoid_grad"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 with different values
    y = np.array([0.6, 0.4, 0.7, 0.3], dtype=np.float32)
    dy = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
    input_dict = {"y": y, "dy": dy, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.raw_ops.SigmoidGrad"] = tf_raw_ops_sigmoid_grad_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_conv_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D convolution, VALID padding
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3, 2).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "CHANNELS_LAST"
    dilations = [1, 1, 1, 1]
    batch_dims = 1
    groups = 1
    name = None

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "batch_dims": batch_dims,
        "groups": groups,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D convolution, SAME padding
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3, 2).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "SAME"
    explicit_paddings = []
    data_format = "CHANNELS_LAST"
    dilations = [1, 1, 1, 1]
    batch_dims = 1
    groups = 1
    name = None

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "batch_dims": batch_dims,
        "groups": groups,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D convolution, EXPLICIT padding
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3, 2).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "EXPLICIT"
    explicit_paddings = [0, 0, 1, 1, 1, 1, 0, 0]
    data_format = "CHANNELS_LAST"
    dilations = [1, 1, 1, 1]
    batch_dims = 1
    groups = 1
    name = None

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "batch_dims": batch_dims,
        "groups": groups,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Batch dims > 1
    input_tensor = np.random.rand(2, 5, 5, 3).astype(np.float32)
    filter_tensor = np.random.rand(3, 3, 3, 2).astype(np.float32)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "CHANNELS_LAST"
    dilations = [1, 1, 1, 1]
    batch_dims = 1
    groups = 1
    name = None

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "batch_dims": batch_dims,
        "groups": groups,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: half type
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float16)
    filter_tensor = np.random.rand(3, 3, 3, 2).astype(np.float16)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "CHANNELS_LAST"
    dilations = [1, 1, 1, 1]
    batch_dims = 1
    groups = 1
    name = None

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "batch_dims": batch_dims,
        "groups": groups,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13: int32 type
    input_tensor = np.random.randint(0, 10, size=(1, 5, 5, 3)).astype(np.int32)
    filter_tensor = np.random.randint(0, 10, size=(3, 3, 3, 2)).astype(np.int32)
    strides = [1, 1, 1, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "CHANNELS_LAST"
    dilations = [1, 1, 1, 1]
    batch_dims = 1
    groups = 1
    name = None

    input_dict = {
        "input": input_tensor,
        "filter": filter_tensor,
        "strides": strides,
        "padding": padding,
        "explicit_paddings": explicit_paddings,
        "data_format": data_format,
        "dilations": dilations,
        "batch_dims": batch_dims,
        "groups": groups,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.raw_ops.Conv"] = tf_raw_ops_conv_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_RandomGammaGrad_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    alpha = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sample = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    input_dict = {"alpha": alpha, "sample": sample, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic float64
    alpha = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    sample = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    input_dict = {"alpha": alpha, "sample": sample, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional float32
    alpha = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    sample = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    input_dict = {"alpha": alpha, "sample": sample, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional float64
    alpha = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    sample = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    input_dict = {"alpha": alpha, "sample": sample, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero values float32
    alpha = np.array([0.1, 0.2, 0.3], dtype=np.float32) #Avoid zero to prevent NaN values in Gamma
    sample = np.array([0.1, 0.2, 0.3], dtype=np.float32) #Avoid zero to prevent NaN values in Gamma
    input_dict = {"alpha": alpha, "sample": sample, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero values float64
    alpha = np.array([0.1, 0.2, 0.3], dtype=np.float64)  #Avoid zero to prevent NaN values in Gamma
    sample = np.array([0.1, 0.2, 0.3], dtype=np.float64) #Avoid zero to prevent NaN values in Gamma
    input_dict = {"alpha": alpha, "sample": sample, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger values float32
    alpha = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    sample = np.array([50.0, 150.0, 250.0], dtype=np.float32)
    input_dict = {"alpha": alpha, "sample": sample, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger values float64
    alpha = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    sample = np.array([50.0, 150.0, 250.0], dtype=np.float64)
    input_dict = {"alpha": alpha, "sample": sample, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Different shape float32
    alpha = np.array([1.0, 2.0], dtype=np.float32)
    sample = np.array([0.5, 1.5], dtype=np.float32)
    input_dict = {"alpha": alpha, "sample": sample, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different shape float64
    alpha = np.array([1.0, 2.0], dtype=np.float64)
    sample = np.array([0.5, 1.5], dtype=np.float64)
    input_dict = {"alpha": alpha, "sample": sample, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: named operation
    alpha = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sample = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    input_dict = {"alpha": alpha, "sample": sample, "name": "my_gamma_grad"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs


generated_inputs["tf.raw_ops.RandomGammaGrad"] = tf_raw_ops_RandomGammaGrad_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_approx_max_k_inputs():
    list_of_inputs = []

    # Input 1
    operand = np.random.rand(10, 20).astype(np.float32).tolist()
    k = 5
    reduction_dimension = -1
    recall_target = 0.95
    reduction_input_size_override = -1
    aggregate_to_topk = True
    name = None

    input_dict = {
        "operand": operand,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    operand = np.random.rand(5, 15).astype(np.float32).tolist()
    k = 10
    reduction_dimension = -1
    recall_target = 0.9
    reduction_input_size_override = -1
    aggregate_to_topk = False
    name = "approx_max"

    input_dict = {
        "operand": operand,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    operand = np.random.rand(20).astype(np.float32).tolist()
    k = 3
    reduction_dimension = -1
    recall_target = 0.85
    reduction_input_size_override = 30
    aggregate_to_topk = True
    name = None

    input_dict = {
        "operand": operand,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    operand = np.random.rand(4, 8).astype(np.float32).tolist()
    k = 7
    reduction_dimension = -1
    recall_target = 0.99
    reduction_input_size_override = -1
    aggregate_to_topk = False
    name = "another_max"

    input_dict = {
        "operand": operand,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    operand = np.random.rand(100).astype(np.float32).tolist()
    k = 25
    reduction_dimension = -1
    recall_target = 0.75
    reduction_input_size_override = -1
    aggregate_to_topk = True
    name = None

    input_dict = {
        "operand": operand,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    operand = np.random.rand(3, 5).astype(np.float32).tolist()
    k = 2
    reduction_dimension = -1
    recall_target = 0.92
    reduction_input_size_override = 7
    aggregate_to_topk = False
    name = "third_max"

    input_dict = {
        "operand": operand,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    operand = np.random.rand(7, 9).astype(np.float32).tolist()
    k = 4
    reduction_dimension = -1
    recall_target = 0.88
    reduction_input_size_override = -1
    aggregate_to_topk = True
    name = None

    input_dict = {
        "operand": operand,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    operand = np.random.rand(2, 4).astype(np.float32).tolist()
    k = 3
    reduction_dimension = -1
    recall_target = 0.97
    reduction_input_size_override = 10
    aggregate_to_topk = False
    name = "fourth_max"

    input_dict = {
        "operand": operand,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    operand = np.random.rand(50).astype(np.float32).tolist()
    k = 12
    reduction_dimension = -1
    recall_target = 0.82
    reduction_input_size_override = -1
    aggregate_to_topk = True
    name = None

    input_dict = {
        "operand": operand,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    operand = np.random.rand(4,6).astype(np.float32).tolist()
    k = 3
    reduction_dimension = -1
    recall_target = 0.91
    reduction_input_size_override = 8
    aggregate_to_topk = False
    name = "fifth_max"

    input_dict = {
        "operand": operand,
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
        "reduction_input_size_override": reduction_input_size_override,
        "aggregate_to_topk": aggregate_to_topk,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["tf.math.approx_max_k"] = tf_math_approx_max_k_inputs()

