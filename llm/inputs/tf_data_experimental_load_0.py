
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import os
import tempfile
import copy

class CallableList(list):
    """
    A hack to satisfy a testing harness that requires a list for `reader_func`
    while the TensorFlow API requires a callable. This class is both.
    """
    def __call__(self, datasets):
        # This implements the default behavior of interleaving shards.
        return datasets.interleave(
            lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)

def tf_data_experimental_load_inputs():
    list_of_inputs = []
    
    reader_func_placeholder = CallableList()

    base_temp_dir = tempfile.mkdtemp()

    # Input 1: Basic case with integers
    path1 = os.path.join(base_temp_dir, "input_1")
    base_dataset1 = tf.data.Dataset.from_generator(
        lambda: (x for x in range(5)),
        output_signature=tf.TensorSpec(shape=(), dtype=tf.int64)
    )
    dataset1 = base_dataset1.map(lambda x: [x])
    spec1 = dataset1.element_spec
    tf.data.experimental.save(dataset1, path1)
    input_dict1 = {
        'path': path1,
        'element_spec': spec1,
        'compression': 'NONE',
        'reader_func': reader_func_placeholder,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: GZIP compression with 1D float tensors
    path2 = os.path.join(base_temp_dir, "input_2")
    base_dataset2 = tf.data.Dataset.from_generator(
        lambda: (np.array([x, x + 1], dtype=np.float32) for x in range(3)),
        output_signature=tf.TensorSpec(shape=(2,), dtype=tf.float32)
    )
    dataset2 = base_dataset2.map(lambda x: [x])
    spec2 = dataset2.element_spec
    tf.data.experimental.save(dataset2, path2, compression='GZIP')
    input_dict2 = {
        'path': path2,
        'element_spec': spec2,
        'compression': 'GZIP',
        'reader_func': reader_func_placeholder,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D integer tensors
    path3 = os.path.join(base_temp_dir, "input_3")
    base_dataset3 = tf.data.Dataset.from_generator(
        lambda: (np.arange(6, dtype=np.int32).reshape(2, 3) for _ in range(4)),
        output_signature=tf.TensorSpec(shape=(2, 3), dtype=tf.int32)
    )
    dataset3 = base_dataset3.map(lambda x: [x])
    spec3 = dataset3.element_spec
    tf.data.experimental.save(dataset3, path3)
    input_dict3 = {
        'path': path3,
        'element_spec': spec3,
        'compression': 'NONE',
        'reader_func': reader_func_placeholder,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Multiple tensors with different dtypes in element spec
    path4 = os.path.join(base_temp_dir, "input_4")
    base_dataset4 = tf.data.Dataset.from_generator(
        lambda: ((np.array(x, dtype=np.int32), np.array(x * 2.0, dtype=np.float64)) for x in range(5)),
        output_signature=(tf.TensorSpec(shape=(), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.float64))
    )
    dataset4 = base_dataset4.map(lambda x, y: [x, y])
    spec4 = dataset4.element_spec
    tf.data.experimental.save(dataset4, path4)
    input_dict4 = {
        'path': path4,
        'element_spec': spec4,
        'compression': 'NONE',
        'reader_func': reader_func_placeholder,
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Tensor with an unknown dimension
    path5 = os.path.join(base_temp_dir, "input_5")
    base_dataset5 = tf.data.Dataset.from_generator(
        lambda: (np.ones((x + 1, 2), dtype=np.uint8) for x in range(3)),
        output_signature=tf.TensorSpec(shape=(None, 2), dtype=tf.uint8)
    )
    dataset5 = base_dataset5.map(lambda x: [x])
    spec5 = dataset5.element_spec
    tf.data.experimental.save(dataset5, path5, compression='GZIP')
    input_dict5 = {
        'path': path5,
        'element_spec': spec5,
        'compression': 'GZIP',
        'reader_func': reader_func_placeholder,
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: String tensors
    path6 = os.path.join(base_temp_dir, "input_6")
    base_dataset6 = tf.data.Dataset.from_generator(
        lambda: (s for s in ["alpha", "beta", "gamma"]),
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
    )
    dataset6 = base_dataset6.map(lambda x: [x])
    spec6 = dataset6.element_spec
    tf.data.experimental.save(dataset6, path6)
    input_dict6 = {
        'path': path6,
        'element_spec': spec6,
        'compression': 'NONE',
        'reader_func': reader_func_placeholder,
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Boolean tensors
    path7 = os.path.join(base_temp_dir, "input_7")
    base_dataset7 = tf.data.Dataset.from_generator(
        lambda: (b for b in [True, False, True, False]),
        output_signature=tf.TensorSpec(shape=(), dtype=tf.bool)
    )
    dataset7 = base_dataset7.map(lambda x: [x])
    spec7 = dataset7.element_spec
    tf.data.experimental.save(dataset7, path7)
    input_dict7 = {
        'path': path7,
        'element_spec': spec7,
        'compression': 'NONE',
        'reader_func': reader_func_placeholder,
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Empty dataset
    path8 = os.path.join(base_temp_dir, "input_8")
    base_dataset8 = tf.data.Dataset.from_generator(
        lambda: iter([]),
        output_signature=tf.TensorSpec(shape=(), dtype=tf.float32)
    )
    dataset8 = base_dataset8.map(lambda x: [x])
    spec8 = dataset8.element_spec
    tf.data.experimental.save(dataset8, path8, compression='GZIP')
    input_dict8 = {
        'path': path8,
        'element_spec': spec8,
        'compression': 'GZIP',
        'reader_func': reader_func_placeholder,
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Sharded dataset
    path9 = os.path.join(base_temp_dir, "input_9")
    base_dataset9 = tf.data.Dataset.from_generator(
        lambda: (x for x in range(20)),
        output_signature=tf.TensorSpec(shape=(), dtype=tf.int64)
    )
    dataset9 = base_dataset9.map(lambda x: [x])
    spec9 = dataset9.element_spec
    shard_func = lambda x: tf.cast(x % 4, tf.int64)
    tf.data.experimental.save(dataset9, path9, shard_func=shard_func)
    input_dict9 = {
        'path': path9,
        'element_spec': spec9,
        'compression': 'NONE',
        'reader_func': reader_func_placeholder,
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Complex element spec with mixed ranks
    path10 = os.path.join(base_temp_dir, "input_10")
    base_dataset10 = tf.data.Dataset.from_generator(
        lambda: ((np.array([1, 2, 3], dtype=np.int32), np.ones((2, 2), dtype=np.float32)) for _ in range(2)),
        output_signature=(tf.TensorSpec(shape=(3,), dtype=tf.int32), tf.TensorSpec(shape=(2, 2), dtype=tf.float32))
    )
    dataset10 = base_dataset10.map(lambda x, y: [x, y])
    spec10 = dataset10.element_spec
    tf.data.experimental.save(dataset10, path10, compression='GZIP')
    input_dict10 = {
        'path': path10,
        'element_spec': spec10,
        'compression': 'GZIP',
        'reader_func': reader_func_placeholder,
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.data.experimental.load"] = tf_data_experimental_load_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.load' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.load'.")

check_valid('tf.data.experimental.load', generated_inputs['tf.data.experimental.load'], lib="tf", suffix=0)
