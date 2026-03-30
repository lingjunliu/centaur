
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_concat_inputs():
    list_of_inputs = []

    # Input 1: Basic concatenation along axis 0
    indices1 = np.array([[0, 0], [1, 2]])
    values1 = np.array([1, 2])
    shape1 = np.array([2, 3])
    st1 = tf.SparseTensor(indices1, values1, shape1)
    indices2 = np.array([[0, 1], [1, 0]])
    values2 = np.array([3, 4])
    shape2 = np.array([2, 3])
    st2 = tf.SparseTensor(indices2, values2, shape2)
    sp_inputs = [st1, st2]
    input_dict = {'axis': 0, 'sp_inputs': sp_inputs, 'expand_nonconcat_dims': False, 'name': 'concat_0'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Concatenation along axis 1
    indices1 = np.array([[0, 0], [1, 1]])
    values1 = np.array([5, 6])
    shape1 = np.array([2, 2])
    st1 = tf.SparseTensor(indices1, values1, shape1)
    indices2 = np.array([[0, 0], [1, 1]])
    values2 = np.array([7, 8])
    shape2 = np.array([2, 2])
    st2 = tf.SparseTensor(indices2, values2, shape2)
    sp_inputs = [st1, st2]
    input_dict = {'axis': 1, 'sp_inputs': sp_inputs, 'expand_nonconcat_dims': False, 'name': 'concat_1'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: expand_nonconcat_dims = True
    indices1 = np.array([[0, 0], [1, 1]])
    values1 = np.array([9, 10])
    shape1 = np.array([2, 2])
    st1 = tf.SparseTensor(indices1, values1, shape1)
    indices2 = np.array([[0, 0]])
    values2 = np.array([11])
    shape2 = np.array([1, 3])
    st2 = tf.SparseTensor(indices2, values2, shape2)
    sp_inputs = [st1, st2]
    input_dict = {'axis': 0, 'sp_inputs': sp_inputs, 'expand_nonconcat_dims': True, 'name': 'concat_expand'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative axis
    indices1 = np.array([[0, 0], [1, 1]])
    values1 = np.array([12, 13])
    shape1 = np.array([2, 2])
    st1 = tf.SparseTensor(indices1, values1, shape1)
    indices2 = np.array([[0, 0], [1, 1]])
    values2 = np.array([14, 15])
    shape2 = np.array([2, 2])
    st2 = tf.SparseTensor(indices2, values2, shape2)
    sp_inputs = [st1, st2]
    input_dict = {'axis': -1, 'sp_inputs': sp_inputs, 'expand_nonconcat_dims': False, 'name': 'concat_neg'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D SparseTensors
    indices1 = np.array([[0, 0, 0], [1, 1, 1]])
    values1 = np.array([16, 17])
    shape1 = np.array([2, 2, 2])
    st1 = tf.SparseTensor(indices1, values1, shape1)
    indices2 = np.array([[0, 0, 0], [1, 1, 1]])
    values2 = np.array([18, 19])
    shape2 = np.array([2, 2, 2])
    st2 = tf.SparseTensor(indices2, values2, shape2)
    sp_inputs = [st1, st2]
    input_dict = {'axis': 1, 'sp_inputs': sp_inputs, 'expand_nonconcat_dims': False, 'name': 'concat_3d'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multiple SparseTensors
    indices1 = np.array([[0, 0]])
    values1 = np.array([21])
    shape1 = np.array([1, 1])
    st1 = tf.SparseTensor(indices1, values1, shape1)
    indices2 = np.array([[0, 0]])
    values2 = np.array([22])
    shape2 = np.array([1, 1])
    st2 = tf.SparseTensor(indices2, values2, shape2)
    indices3 = np.array([[0, 0]])
    values3 = np.array([23])
    shape3 = np.array([1, 1])
    st3 = tf.SparseTensor(indices3, values3, shape3)
    sp_inputs = [st1, st2, st3]
    input_dict = {'axis': 1, 'sp_inputs': sp_inputs, 'expand_nonconcat_dims': False, 'name': 'concat_multi'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Integer values
    indices1 = np.array([[0, 0], [1, 1]])
    values1 = np.array([24, 25], dtype=np.int32)
    shape1 = np.array([2, 2])
    st1 = tf.SparseTensor(indices1, values1, shape1)
    indices2 = np.array([[0, 0], [1, 1]])
    values2 = np.array([26, 27], dtype=np.int32)
    shape2 = np.array([2, 2])
    st2 = tf.SparseTensor(indices2, values2, shape2)
    sp_inputs = [st1, st2]
    input_dict = {'axis': 1, 'sp_inputs': sp_inputs, 'expand_nonconcat_dims': False, 'name': 'concat_int'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: expand_nonconcat_dims with axis = 0
    indices1 = np.array([[0, 0]])
    values1 = np.array([28])
    shape1 = np.array([1, 2])
    st1 = tf.SparseTensor(indices1, values1, shape1)
    indices2 = np.array([[0, 0]])
    values2 = np.array([29])
    shape2 = np.array([1, 3])
    st2 = tf.SparseTensor(indices2, values2, shape2)
    sp_inputs = [st1, st2]
    input_dict = {'axis': 0, 'sp_inputs': sp_inputs, 'expand_nonconcat_dims': True, 'name': 'concat_expand_0'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: different dtypes of values
    indices1 = np.array([[0, 0], [1, 1]])
    values1 = np.array([30.0, 31.0], dtype=np.float32)
    shape1 = np.array([2, 2])
    st1 = tf.SparseTensor(indices1, values1, shape1)
    indices2 = np.array([[0, 0], [1, 1]])
    values2 = np.array([32.0, 33.0], dtype=np.float32)
    shape2 = np.array([2, 2])
    st2 = tf.SparseTensor(indices2, values2, shape2)
    sp_inputs = [st1, st2]
    input_dict = {'axis': 1, 'sp_inputs': sp_inputs, 'expand_nonconcat_dims': False, 'name': 'concat_dtype'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.sparse.concat"] = tf_sparse_concat_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.concat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.concat'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sparse.concat', generated_inputs['tf.sparse.concat'], lib="tf", suffix=0)
