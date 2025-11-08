generated_inputs = {}

import tensorflow as tf
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
import copy
import numpy as np

def tf_compat_path_to_str_inputs():
    list_of_inputs = []

    path = r"C:\XYZ\tensorflow\./.././tensorflow"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = r"\\Server\Share\Folder\file.txt"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = r"D:\path with spaces\sub dir\file name.txt"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "./.././Corpus"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "/var/log/../tmp//./app/"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "."
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = ".."
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "~/.cache/pip"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "relative/path/with//double///slashes"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "archive.tar.gz"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = ""
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = ".env"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "C:/Windows/System32/drivers/etc/hosts"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "/home/用户/项目/数据集"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    path = "../../..//folder/./subfolder/../file"
    list_of_inputs.append(copy.deepcopy({"path": path}))

    return list_of_inputs

generated_inputs["tf.compat.path_to_str"] = tf_compat_path_to_str_inputs()

def tf_experimental_numpy_isfinite_inputs():
    list_of_inputs = []

    x = np.array([1.0, -2.5, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([[np.inf, -np.inf], [np.nan, 3.14]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array(42.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([np.nan, np.inf, -1.23], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([[[1.0, -0.0], [np.inf, -np.inf]], [[np.nan, 2.0], [3.5, -4.5]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    arr = np.linspace(-10, 10, 20, dtype=np.float64).reshape(4, 5)
    arr[2, 4] = -np.inf
    x = arr[:, ::2]
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.asfortranarray(np.array([[1.0, np.nan], [np.inf, -3.0]], dtype=np.float64))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.full((2, 3, 4, 5), fill_value=np.nan, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([-0.0, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([1e308, -1e308, 1e-308, np.inf, -np.inf, np.nan], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    finfo16 = np.finfo(np.float16)
    x = np.array([finfo16.max, finfo16.tiny, -finfo16.max, np.nan, np.inf, -np.inf], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.isfinite"] = tf_experimental_numpy_isfinite_inputs()



def tf_feature_column_categorical_column_with_identity_inputs():
    list_of_inputs = []

    key = "user_id"
    num_buckets = np.int32(10)
    default_value = np.int32(0)
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "video_id"
    num_buckets = 1000000
    default_value = 0
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "session_index"
    num_buckets = np.int64(3)
    default_value = np.int64(2)
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "item"
    num_buckets = 255
    default_value = 1
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "country_code"
    num_buckets = 5
    default_value = 1
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "feature/segment"
    num_buckets = np.int16(2)
    default_value = np.int16(0)
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "edge_case_zero"
    num_buckets = 1
    default_value = 0
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "product_id"
    num_buckets = np.int64(1024)
    default_value = np.int64(123)
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "bucketed_age"
    num_buckets = np.int32(100)
    default_value = np.int32(99)
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "ad_slot"
    num_buckets = 7
    default_value = 3
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "city_hash"
    num_buckets = 2048
    default_value = 1024
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    key = "experiment_group"
    num_buckets = np.int32(4)
    default_value = np.int32(0)
    list_of_inputs.append(copy.deepcopy({"key": key, "num_buckets": num_buckets, "default_value": default_value}))

    return list_of_inputs

generated_inputs["tf.feature_column.categorical_column_with_identity"] = tf_feature_column_categorical_column_with_identity_inputs()



def tf_feature_column_sequence_categorical_column_with_hash_bucket_inputs():
    list_of_inputs = []

    input_dict = {
        "key": "tokens",
        "hash_bucket_size": 1000,
        "dtype": np.dtype("U10")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "words",
        "hash_bucket_size": np.int32(2),
        "dtype": np.dtype("S8")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "ids",
        "hash_bucket_size": 17,
        "dtype": np.dtype("int32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "categories_en",
        "hash_bucket_size": np.int64(4096),
        "dtype": np.dtype("int64")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "序列",
        "hash_bucket_size": 257,
        "dtype": np.dtype(np.str_)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "byte_tokens",
        "hash_bucket_size": 65535,
        "dtype": np.dtype("S1")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "click_ids",
        "hash_bucket_size": 100,
        "dtype": np.dtype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "product_ids",
        "hash_bucket_size": np.int32(8192),
        "dtype": np.dtype(np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "tags",
        "hash_bucket_size": 3,
        "dtype": np.dtype("U4")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "features",
        "hash_bucket_size": 50,
        "dtype": np.dtype("U1")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "labels",
        "hash_bucket_size": 1024,
        "dtype": np.dtype("S16")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "key": "session_tokens",
        "hash_bucket_size": 200,
        "dtype": np.dtype("U32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.feature_column.sequence_categorical_column_with_hash_bucket"] = tf_feature_column_sequence_categorical_column_with_hash_bucket_inputs()



def tf_get_static_value_inputs():
    list_of_inputs = []

    tensor = tf.constant(np.int32(10))
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.float32(-3.5))
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.array([1, 2, 3], dtype=np.int64))
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.array([[1.0, -2.5], [3.1, 4.2]], dtype=np.float64))
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.array([True, False, True], dtype=np.bool_))
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.array([1+2j, -3+0.5j], dtype=np.complex64))
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    b = tf.constant(np.array([-1, 0, 1], dtype=np.int32))
    tensor = tf.add(a, b)
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = tf.constant(np.arange(6, dtype=np.int32))
    shape = tf.constant(np.array([2, 3], dtype=np.int32))
    tensor = tf.reshape(base, shape)
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    mat = tf.constant(np.arange(12, dtype=np.float32).reshape(3, 4))
    tensor = tf.transpose(mat)
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t1 = tf.constant(np.array([[1, 2]], dtype=np.int32))
    t2 = tf.constant(np.array([[3, 4]], dtype=np.int32))
    tensor = tf.concat([t1, t2], axis=0)
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.arange(24, dtype=np.int16).reshape(2, 3, 4))
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.array([np.nan, np.inf, -np.inf], dtype=np.float32))
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.array([0, 255], dtype=np.uint8))
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base_int = tf.constant(np.array([1, 0, 1], dtype=np.int32))
    tensor = tf.cast(base_int, tf.bool)
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    mat = tf.constant(np.arange(6, dtype=np.int32).reshape(2, 3))
    axis = tf.constant(np.int32(1))
    tensor = tf.reduce_sum(mat, axis=axis)
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.add(tf.constant(np.int32(3)), tf.Variable(np.int32(4)))
    partial = True
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = tf.constant(np.array([], dtype=np.float32))
    partial = False
    input_dict = {"tensor": tensor, "partial": partial}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.get_static_value"] = tf_get_static_value_inputs()



def tf_identity_inputs():
    list_of_inputs = []

    input_arr = np.array([0.78], dtype=np.float32)
    input_dict = {"input": input_arr, "name": "float32_vector"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array(5, dtype=np.int32)
    input_dict = {"input": input_arr, "name": "int32_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[-1, 0, 2], [3, -4, 5]], dtype=np.int64)
    input_dict = {"input": input_arr, "name": "int64_matrix_with_negatives"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[np.nan, np.inf], [-np.inf, -1.5]]], dtype=np.float64)
    input_dict = {"input": input_arr, "name": "float64_3d_with_nan_inf"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([], dtype=np.int32)
    input_dict = {"input": input_arr, "name": "empty_int32_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[True, False, True], [False, False, True]], dtype=bool)
    input_dict = {"input": input_arr, "name": "bool_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    input_dict = {"input": input_arr, "name": "complex64_vector"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.arange(2 * 3 * 1 * 4, dtype=np.float16).reshape(2, 3, 1, 4)
    input_dict = {"input": input_arr, "name": "float16_4d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array(True, dtype=bool)
    input_dict = {"input": input_arr, "name": "bool_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.empty((2, 0, 3), dtype=np.float32)
    input_dict = {"input": input_arr, "name": "empty_axis_float32_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    input_dict = {"input": input_arr, "name": "uint8_3d_image"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.identity"] = tf_identity_inputs()

np.random.seed(42)

def tf_image_adjust_saturation_inputs():
    list_of_inputs = []

    image = np.random.randint(0, 256, size=(4, 4, 3), dtype=np.uint8)
    saturation_factor = np.float32(0.0)
    name = "zero_sat_uint8_3d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.random.rand(3, 5, 3).astype(np.float32)
    saturation_factor = np.float32(0.5)
    name = "half_sat_float32_3d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.random.randint(0, 256, size=(2, 3, 4, 3), dtype=np.uint8)
    saturation_factor = np.float64(2.0)
    name = "double_sat_uint8_4d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.random.rand(1, 2, 2, 3).astype(np.float32)
    saturation_factor = np.float32(1.0)
    name = "no_change_float32_4d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.random.rand(5, 2, 3).astype(np.float16)
    saturation_factor = np.float16(3.5)
    name = "high_sat_float16_3d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.random.rand(3, 4, 5, 3).astype(np.float64)
    saturation_factor = np.float64(10.0)
    name = "very_high_sat_float64_4d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.tile(np.linspace(0, 1, 9, dtype=np.float32).reshape(3, 3, 1), (1, 1, 3))
    saturation_factor = np.float32(1.25)
    name = "grayscale_like_float32_3d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.random.randint(0, 256, size=(2, 1, 1, 3), dtype=np.uint8)
    saturation_factor = np.float32(4.0)
    name = "tiny_spatial_uint8_4d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.random.rand(64, 64, 3).astype(np.float32)
    saturation_factor = np.float32(1.25)
    name = "mid_sat_float32_3d_large"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = np.random.randint(0, 256, size=(5, 8, 8, 3), dtype=np.uint8)
    saturation_factor = np.float64(0.25)
    name = "quarter_sat_uint8_4d_batch5"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    image = (np.random.rand(10, 10, 3).astype(np.float32) * 2.0)
    saturation_factor = np.float32(2.5)
    name = "over_one_range_float32_3d"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    return list_of_inputs

generated_inputs["tf.image.adjust_saturation_1"] = tf_image_adjust_saturation_inputs()



def tf_image_adjust_saturation_inputs():
    list_of_inputs = []

    # Input 1
    image = np.array([[[0, 128, 255],
                       [30, 60, 90]],
                      [[200, 150, 100],
                       [255, 0, 50]]], dtype=np.uint8)
    saturation_factor = np.array(0.0, dtype=np.float32)
    name = "case1_zero_sat"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 2
    image = np.array([[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]],
                      [[7.0, 8.0, 9.0],
                       [10.0, 11.0, 12.0]]], dtype=np.float32)
    saturation_factor = np.array(0.5, dtype=np.float32)
    name = "case2_half"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 3
    image = np.linspace(-1.0, 1.0, num=27, dtype=np.float32).reshape(3, 3, 3)
    saturation_factor = np.array(1.0, dtype=np.float32)
    name = "case3_negative_values"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 4
    image = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)
    saturation_factor = np.array(2.0, dtype=np.float32)
    name = "case4_float32_small_image"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 5
    rng = np.random.RandomState(0)
    image = rng.rand(2, 3, 4, 3).astype(np.float32)
    saturation_factor = np.array(1.5, dtype=np.float32)
    name = "case5_batched_float32"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 6
    rng = np.random.RandomState(1)
    image = rng.randint(0, 256, size=(3, 2, 2, 3), dtype=np.uint8)
    saturation_factor = np.array(1.2, dtype=np.float32)
    name = "case6_batched_uint8"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 7
    rng = np.random.RandomState(2)
    image = (rng.rand(6, 5, 3) * 255).astype(np.float32)
    saturation_factor = np.array(3.0, dtype=np.float32)
    name = "case7_float32_high_saturation"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 8
    image = np.arange(27, dtype=np.uint8).reshape(1, 3, 3, 3)
    saturation_factor = np.array(0.75, dtype=np.float32)
    name = "case8_single_batch_uint8"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 9
    image = np.zeros((8, 8, 3), dtype=np.float32)
    saturation_factor = np.array(5.0, dtype=np.float32)
    name = "case9_zero_image_high_factor"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 10
    rng = np.random.RandomState(3)
    image = rng.rand(4, 4, 3).astype(np.float32)
    saturation_factor = np.array(10.0, dtype=np.float32)
    name = "case10_extreme_factor"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 11
    grad = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    r = np.tile(grad[:, None], (1, 10))
    g = np.tile(grad[None, :], (10, 1))
    b = np.flipud(r)
    image = np.stack([r, g, b], axis=-1).astype(np.float32)
    saturation_factor = np.array(2.5, dtype=np.float32)
    name = "case11_gradient_float32"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    # Input 12
    image = np.array([[[1000.0, 200.0, 50.0],
                       [500.0, 500.0, 500.0]],
                      [[-100.0, 0.0, 100.0],
                       [1e6, 1e6 - 1e3, 1e6 - 2e3]]], dtype=np.float32)
    saturation_factor = np.array(0.1, dtype=np.float32)
    name = "case12_large_values_float32"
    list_of_inputs.append(copy.deepcopy({"image": image, "saturation_factor": saturation_factor, "name": name}))

    return list_of_inputs

generated_inputs["tf.image.adjust_saturation_2"] = tf_image_adjust_saturation_inputs()



def tf_io_serialize_tensor_inputs():
    list_of_inputs = []

    tensor = np.array(1, dtype=np.int32)
    name = "scalar_int32"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([-1.5, 0.0, 2.5], dtype=np.float32)
    name = "vector_float32"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[-1, 2], [3, -4]], dtype=np.int64)
    name = "matrix_int64"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    rng = np.random.default_rng(42)
    tensor = rng.standard_normal((2, 3, 4)).astype(np.float64)
    name = "tensor3d_float64"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[True, False], [False, True]], dtype=np.bool_)
    name = "bool_2x2"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([], dtype=np.float32)
    name = "empty_float32"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.empty((2, 0, 3), dtype=np.float32)
    name = "zerosize_dim"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    name = "complex64_1d"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([np.nan, np.inf, -np.inf, 1e30], dtype=np.float64)
    name = "nan_inf_float64"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.arange(2 * 3 * 4 * 5, dtype=np.int32).reshape(2, 3, 4, 5)
    name = "rank4_int32"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = (np.arange(60, dtype=np.float16) - 30).reshape(3, 4, 5)
    name = "float16_3d"
    input_dict = {"tensor": tensor, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.io.serialize_tensor"] = tf_io_serialize_tensor_inputs()



def tf_math_atan2_inputs():
    list_of_inputs = []

    y = np.array([1.0, -1.0], dtype=np.float32)
    x = np.array([1.0, 1.0], dtype=np.float32)
    input_dict = {"y": y, "x": x, "name": "basic_1d_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float64)
    x = np.array([[1.0, -1.0], [0.0, -2.0]], dtype=np.float64)
    input_dict = {"y": y, "x": x, "name": "mixed_2d_f64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([0.5, -0.5, 1.5], dtype=np.float16)
    x = np.array(1.0, dtype=np.float16)
    input_dict = {"y": y, "x": x, "name": "broadcast_scalar_x_f16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([[1.0], [-2.0]], dtype=np.float32)
    x = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    input_dict = {"y": y, "x": x, "name": "broadcast_2x1_3_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([[[1.0, -1.0], [2.0, -2.0]], [[-3.0, 3.0], [0.0, 0.5]]], dtype=np.float64)
    x = np.array([[[1.0, 1.0], [-2.0, 2.0]], [[3.0, -3.0], [1.0, -0.5]]], dtype=np.float64)
    input_dict = {"y": y, "x": x, "name": "three_d_f64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.zeros((4,), dtype=np.float32)
    x = np.array([1.0, -1.0, 0.0, 2.0], dtype=np.float32)
    input_dict = {"y": y, "x": x, "name": "zeros_y_axes_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([1e20, -1e20, 3e30], dtype=np.float64)
    x = np.array([1e20, 1e20, -3e30], dtype=np.float64)
    input_dict = {"y": y, "x": x, "name": "large_vals_f64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([1e-30, -1e-30, 5e-40], dtype=np.float64)
    x = np.array([-1e-30, 1e-30, 5e-40], dtype=np.float64)
    input_dict = {"y": y, "x": x, "name": "small_vals_f64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    x = np.array([1.0, -np.inf, np.inf], dtype=np.float32)
    input_dict = {"y": y, "x": x, "name": "nan_inf_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float16)
    x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float16)
    input_dict = {"y": y, "x": x, "name": "equal_yx_f16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    y = base.T
    x = np.array([[2.0, -2.0, 1.0]], dtype=np.float32)
    input_dict = {"y": y, "x": x, "name": "transpose_broadcast_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    rng = np.random.RandomState(0)
    y = rng.randn(5).astype(np.float32)
    x = rng.randn(5).astype(np.float32)
    input_dict = {"y": y, "x": x, "name": "random_1d_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.atan2"] = tf_math_atan2_inputs()



def tf_raw_ops_ComputeAccidentalHits_inputs():
    list_of_inputs = []

    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([2, 5, 4], dtype=np.int64)
    input_dict = {
        "seed": np.int32(0),
        "seed2": np.int32(0),
        "name": "case_1",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[-1], [0], [7]], dtype=np.int64)
    sampled_candidates = np.array([-1, 8, 9, 10], dtype=np.int64)
    input_dict = {
        "seed": np.int32(123),
        "seed2": np.int32(456),
        "name": "case_2",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[10, 10, 10], [10, 11, 12]], dtype=np.int64)
    sampled_candidates = np.array([10, 11, 12, 13], dtype=np.int64)
    input_dict = {
        "seed": np.int32(999),
        "seed2": np.int32(0),
        "name": "case_3",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[100, 200], [300, 400], [500, 600]], dtype=np.int64)
    sampled_candidates = np.array([700, 800, 900], dtype=np.int64)
    input_dict = {
        "seed": np.int32(0),
        "seed2": np.int32(42),
        "name": "case_4",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([
        [np.int64(np.iinfo(np.int64).min + 1), 0, 5, 999999999999],
        [42, -99999999999, np.int64(np.iinfo(np.int64).max - 1), 7]
    ], dtype=np.int64)
    sampled_candidates = np.array([np.int64(np.iinfo(np.int64).max - 1), 123, 999999999999, -99999999999], dtype=np.int64)
    input_dict = {
        "seed": np.int32(2021),
        "seed2": np.int32(2022),
        "name": "case_5",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(4),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.zeros((5, 1), dtype=np.int64)
    sampled_candidates = np.array([0, 1, 2], dtype=np.int64)
    input_dict = {
        "seed": np.int32(7),
        "seed2": np.int32(8),
        "name": "case_6",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[3, -5], [3, -5], [7, 8]], dtype=np.int64)
    sampled_candidates = np.array([3, -5], dtype=np.int64)
    input_dict = {
        "seed": np.int32(11),
        "seed2": np.int32(12),
        "name": "case_7",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[2, 2, 2, 3, 4]], dtype=np.int64)
    sampled_candidates = np.array([2, 2, 2, 3, 5, 6], dtype=np.int64)
    input_dict = {
        "seed": np.int32(21),
        "seed2": np.int32(22),
        "name": "case_8",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(5),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[10], [20], [30], [40]], dtype=np.int64)
    sampled_candidates = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=np.int64)
    input_dict = {
        "seed": np.int32(100),
        "seed2": np.int32(200),
        "name": "case_9",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.int64)
    sampled_candidates = np.array([6, 5, 4, 3, 2, 1], dtype=np.int64)
    input_dict = {
        "seed": np.int32(31415),
        "seed2": np.int32(27182),
        "name": "case_10",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([4, 0, 2, 6, 8], dtype=np.int64)
    input_dict = {
        "seed": np.int32(555),
        "seed2": np.int32(777),
        "name": "case_11",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1000000000000], [-1000000000000], [5]], dtype=np.int64)
    sampled_candidates = np.array([-1000000000000, 7, 1000000000000], dtype=np.int64)
    input_dict = {
        "seed": np.int32(42),
        "seed2": np.int32(24),
        "name": "case_12",
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": np.int32(1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ComputeAccidentalHits"] = tf_raw_ops_ComputeAccidentalHits_inputs()



def tf_raw_ops_empty_inputs():
    list_of_inputs = []

    shape = np.array([2, 3], dtype=np.int32)
    dtype = np.float32
    init = False
    name = "empty_f32_2x3_noinit"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([5], dtype=np.int32)
    dtype = np.int64
    init = True
    name = "empty_i64_5_init"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([], dtype=np.int32)
    dtype = np.bool_
    init = True
    name = "empty_bool_scalar_init"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([0], dtype=np.int32)
    dtype = np.float64
    init = False
    name = "empty_f64_len0_noinit"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([3, 0, 4], dtype=np.int32)
    dtype = np.int32
    init = True
    name = "empty_i32_3x0x4_init"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([1, 1, 1, 1], dtype=np.int32)
    dtype = np.complex64
    init = True
    name = "empty_c64_1x1x1x1_init"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([10, 10], dtype=np.int32)
    dtype = np.uint8
    init = False
    name = "empty_u8_10x10_noinit"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([2, 3, 4], dtype=np.int32)
    dtype = np.float16
    init = True
    name = "empty_f16_2x3x4_init"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([7], dtype=np.int32)
    dtype = np.complex128
    init = False
    name = "empty_c128_len7_noinit"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([0, 0], dtype=np.int32)
    dtype = np.int8
    init = True
    name = "empty_i8_0x0_init"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([2], dtype=np.int32)
    dtype = np.uint16
    init = False
    name = "empty_u16_len2_noinit"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([4, 1, 2, 0], dtype=np.int32)
    dtype = np.float32
    init = True
    name = "empty_f32_4x1x2x0_init"
    input_dict = {"init": init, "name": name, "shape": shape, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Empty"] = tf_raw_ops_empty_inputs()

def tf_raw_ops_greater_equal_inputs():
    list_of_inputs = []

    x = np.array([5, 4, 6, 7], dtype=np.int32)
    y = np.array([5, 2, 5, 10], dtype=np.int32)
    input_dict = {"name": "ge_case_1", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([5, 4, 6, 7], dtype=np.int32)
    y = np.array([5], dtype=np.int32)
    input_dict = {"name": "ge_case_2", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], dtype=np.float32)
    y = np.array([[2.0, 2.0, 2.0],
                  [3.0, 6.0, 6.0]], dtype=np.float32)
    input_dict = {"name": "ge_case_3", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1, 0, 2],
                  [5, -5, 10]], dtype=np.int64)
    y = np.array([0, 0, 1], dtype=np.int64)
    input_dict = {"name": "ge_case_4", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[ -1.5,  0.0,  2.5, -3.2]],
                  [[ 10.1, -5.0,  0.0,  7.7]]], dtype=np.float64)
    y = np.array([[[ -2.0],
                   [  0.0],
                   [  5.0]]], dtype=np.float64)
    input_dict = {"name": "ge_case_5", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-128, -1, 127], dtype=np.int8)
    y = np.array(0, dtype=np.int8)
    input_dict = {"name": "ge_case_6", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0, 255],
                  [128, 64]], dtype=np.uint8)
    y = np.array([100], dtype=np.uint8)
    input_dict = {"name": "ge_case_7", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    y = np.array([0.0, np.inf, -1.0, np.nan], dtype=np.float32)
    input_dict = {"name": "ge_case_8", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.5, -2.0, 0.0, 65504.0]], dtype=np.float16)
    y = np.array(1.0, dtype=np.float16)
    input_dict = {"name": "ge_case_9", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1],
                  [2],
                  [3]], dtype=np.int32)
    y = np.array([[0, 1, 2, 3]], dtype=np.int32)
    input_dict = {"name": "ge_case_10", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, -2],
                   [3, -4]],
                  [[5, -6],
                   [7, -8]]], dtype=np.int16)
    y = np.array([[[0],
                   [2]]], dtype=np.int16)
    input_dict = {"name": "ge_case_11", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-3, dtype=np.int64)
    y = np.array(-3, dtype=np.int64)
    input_dict = {"name": "ge_case_12", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-5, dtype=np.int32)
    y = np.array([[-10, -5, 0, 5]], dtype=np.int32)
    input_dict = {"name": "ge_case_13", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.GreaterEqual"] = tf_raw_ops_greater_equal_inputs()

def tf_raw_ops_real_inputs():
    list_of_inputs = []

    x = np.array([-2.25 + 4.75j, 3.25 + 5.75j], dtype=np.complex64)
    input_dict = {"Tout": np.float32, "name": "real_vec_c64", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1 + 2j, -3 - 4j], [0 + 0j, 5 - 6j]], dtype=np.complex128)
    input_dict = {"Tout": np.float64, "name": "real_matrix_c128", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    imag = (-real - 0.5).astype(np.float32)
    x = (real + 1j * imag).astype(np.complex64)
    input_dict = {"Tout": np.dtype("float32"), "name": "real_3d_c64", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(7.5 - 1.5j, dtype=np.complex64)
    input_dict = {"Tout": np.float32, "name": "real_scalar_c64", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.complex128)
    input_dict = {"Tout": np.float64, "name": "real_empty1d_c128", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = np.linspace(-10, 10, num=24, dtype=np.float32).reshape(1, 2, 3, 4)
    imag = np.linspace(10, -10, num=24, dtype=np.float32).reshape(1, 2, 3, 4)
    x = (real + 1j * imag).astype(np.complex64)
    input_dict = {"Tout": np.dtype("float32"), "name": "real_4d_c64", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base_real = np.arange(12, dtype=np.float64).reshape(3, 4)
    base_imag = np.flip(base_real, axis=1)
    x_full = (base_real + 1j * base_imag).astype(np.complex128)
    x = x_full[::2, ::2]
    input_dict = {"Tout": np.float64, "name": "real_noncontig_c128", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-30 + 1e-30j, -1e-40 + 2e-40j], dtype=np.complex128)
    input_dict = {"Tout": np.dtype("float64"), "name": "real_tinyvals_c128", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e10 - 1e9j, -3.4e20 + 1e19j], dtype=np.complex64)
    input_dict = {"Tout": np.float32, "name": "real_largevals_c64", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan + 1j, np.inf - np.inf * 1j, -np.inf + (np.nan * 1j)], dtype=np.complex128)
    input_dict = {"Tout": np.float64, "name": "real_nan_inf_c128", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.zeros((2, 0, 3), dtype=np.complex64)
    input_dict = {"Tout": np.float32, "name": "real_empty_axes_c64", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-0.0 + 0.0j]], dtype=np.complex128)
    input_dict = {"Tout": np.dtype("float64"), "name": "real_singleton2d_c128", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([3 + 4j], dtype=np.complex64)
    input_dict = {"Tout": np.float32, "name": "real_len1_c64", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = np.array([[1e-6, -2e-6, 3e-6]], dtype=np.float64)
    imag = np.array([[4e-6, -5e-6, 6e-6]], dtype=np.float64)
    x = (real + 1j * imag).astype(np.complex128)
    input_dict = {"Tout": np.float64, "name": "real_small_2d_c128", "input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Real"] = tf_raw_ops_real_inputs()

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_relu_inputs():
    list_of_inputs = []

    features = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
    input_dict = {"name": "relu_f32_1D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-1.2, 2.5], [0.0, -3.4]], dtype=np.float64)
    input_dict = {"name": "relu_f64_2D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[-3, -2, -1], [0, 1, 2]], [[3, -4, 5], [-6, 7, -8]]], dtype=np.int32)
    input_dict = {"name": "relu_i32_3D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([0, 1, 255], dtype=np.uint8)
    input_dict = {"name": "relu_u8_1D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-32768, -1, 0, 1, 32767]], dtype=np.int16)
    input_dict = {"name": "relu_i16_2D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array(-5, dtype=np.int8)
    input_dict = {"name": "relu_i8_scalar_0D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[[-10, 0, 10]]], [[[20, -30, 40]]]], dtype=np.int64)
    input_dict = {"name": "relu_i64_4D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[-1.5, 2.0], [3.3, -4.4], [0.0, 5.5]],
                         [[-6.6, 7.7], [-8.8, 9.9], [10.0, -11.0]]], dtype=np.float16)
    input_dict = {"name": "relu_f16_3D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([], dtype=np.float32)
    input_dict = {"name": "relu_empty_f32", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[
        [
            [[-3.0, 0.0, 3.0]],
            [[4.5, -5.5, 6.5]]
        ],
        [
            [[7.0, -8.0, 9.0]],
            [[-1.0, 2.0, -3.0]]
        ]
    ]], dtype=np.float32)
    input_dict = {"name": "relu_f32_5D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([-1e6, -1.5, 0.0, 1.5, 3.4e5], dtype=np.float64)
    input_dict = {"name": "relu_f64_large_range", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[0, 128, 255], [5, 10, 15]], dtype=np.uint8)
    input_dict = {"name": "relu_u8_2D", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Relu"] = tf_raw_ops_relu_inputs()



def tf_raw_ops_SparseReduceSumSparse_inputs():
    list_of_inputs = []

    input_indices = np.array([[0, 1], [2, 3], [1, 0]], dtype=np.int64)
    input_values = np.array([1.0, 2.5, -3.0], dtype=np.float32)
    input_shape = np.array([3, 4], dtype=np.int64)
    reduction_axes = np.array([1], dtype=np.int32)
    keep_dims = False
    name = "srs_case1"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0, 0], [0, 0], [1, 2], [2, 1]], dtype=np.int64)
    input_values = np.array([5, -2, 7, 3], dtype=np.int64)
    input_shape = np.array([3, 4], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = True
    name = "srs_case2"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[1, 2, 3], [0, 0, 0], [1, 0, 2], [0, 2, 1], [1, 1, 1]], dtype=np.int64)
    input_values = np.array([1, 2, -1, 3, 4], dtype=np.int32)
    input_shape = np.array([2, 3, 4], dtype=np.int64)
    reduction_axes = np.array([-1], dtype=np.int32)
    keep_dims = False
    name = "srs_case3"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0, 1, 2], [1, 2, 3], [1, 0, 0], [0, 2, 1]], dtype=np.int64)
    input_values = np.array([1+2j, -3+0.5j, 2-1j, -0-1j], dtype=np.complex64)
    input_shape = np.array([2, 3, 4], dtype=np.int64)
    reduction_axes = np.array([0, 2], dtype=np.int32)
    keep_dims = True
    name = "srs_case4"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    input_values = np.array([127, -1], dtype=np.int8)
    input_shape = np.array([2, 2], dtype=np.int64)
    reduction_axes = np.array([], dtype=np.int32)
    keep_dims = False
    name = "srs_case5"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0], [2], [4]], dtype=np.int64)
    input_values = np.array([-1.5, 2.0, 3.25], dtype=np.float64)
    input_shape = np.array([5], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = True
    name = "srs_case6"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([
        [0, 0, 0, 0],
        [1, 0, 2, 1],
        [1, 0, 1, 0],
        [0, 0, 2, 1],
        [0, 0, 1, 1],
        [1, 0, 2, 0]
    ], dtype=np.int64)
    input_values = np.array([0.5, -1.0, 3.0, 2.0, -0.5, 4.0], dtype=np.float64)
    input_shape = np.array([2, 1, 3, 2], dtype=np.int64)
    reduction_axes = np.array([1, 2], dtype=np.int32)
    keep_dims = False
    name = "srs_case7"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=np.int64)
    input_values = np.array([10, 20, 30], dtype=np.int32)
    input_shape = np.array([2, 2, 2], dtype=np.int64)
    reduction_axes = np.array([-3, -2, -1], dtype=np.int32)
    keep_dims = True
    name = "srs_case8"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0, 1], [1, 2], [2, 0], [2, 2]], dtype=np.int64)
    input_values = np.array([1+1j, -2+0j, 0+3j, 4-1j], dtype=np.complex64)
    input_shape = np.array([3, 3], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = True
    name = "srs_case9"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.empty((0, 2), dtype=np.int64)
    input_values = np.array([], dtype=np.int8)
    input_shape = np.array([4, 3], dtype=np.int64)
    reduction_axes = np.array([1], dtype=np.int32)
    keep_dims = False
    name = "srs_case10"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0, 0, 0], [1, 2, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int64)
    input_values = np.array([1.0, -2.0, 3.5, -0.5], dtype=np.float32)
    input_shape = np.array([2, 3, 1], dtype=np.int64)
    reduction_axes = np.array([2], dtype=np.int32)
    keep_dims = True
    name = "srs_case11"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0]], dtype=np.int64)
    input_values = np.array([123], dtype=np.int32)
    input_shape = np.array([1], dtype=np.int64)
    reduction_axes = np.array([], dtype=np.int32)
    keep_dims = True
    name = "srs_case12"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    input_values = np.array([100, 200], dtype=np.int32)
    input_shape = np.array([2, 3], dtype=np.int64)
    reduction_axes = np.array([0, 1], dtype=np.int32)
    keep_dims = False
    name = "srs_case13"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    input_indices = np.array([[0, 0, 1], [2, 1, 0], [1, 1, 1]], dtype=np.int64)
    input_values = np.array([300, -200, 100], dtype=np.int32)
    input_shape = np.array([3, 2, 2], dtype=np.int64)
    reduction_axes = np.array([0], dtype=np.int32)
    keep_dims = False
    name = "srs_case14"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input_indices": input_indices,
        "input_values": input_values,
        "input_shape": input_shape,
        "reduction_axes": reduction_axes
    }))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseReduceSumSparse"] = tf_raw_ops_SparseReduceSumSparse_inputs()



def tf_sysconfig_get_include_inputs():
    list_of_inputs = []
    for _ in range(12):
        input_dict = {}
        list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs["tf.sysconfig.get_include"] = tf_sysconfig_get_include_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_compress_inputs():
    list_of_inputs = []

    a = np.array([1, -2, 0, 5, -7], dtype=np.int32)
    condition = np.array([True, False, True, True, False], dtype=bool)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    condition = np.array([1, 0, 1], dtype=np.int32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    condition = np.array([0, 1, 1, 0], dtype=np.int32)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.linspace(-1, 1, 24, dtype=np.float64).reshape(2, 3, 4)
    condition = np.array([True, False, True, True], dtype=bool)
    axis = 2
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = (np.arange(10, dtype=np.float32) - 5).reshape(2, 5)
    condition = np.array([0, 1, 1, 0, 1], dtype=np.int32)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.array([[True, False, True],
                  [False, False, True],
                  [True, True, False],
                  [False, True, True]], dtype=bool)
    condition = np.array([1, 0, 1, 1], dtype=np.int64)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = (np.arange(24, dtype=np.int64).reshape(2, 3, 4)) * -1
    condition = np.array([1, 0, 1, 0], dtype=np.int32)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = (np.arange(24, dtype=np.float32).reshape(2, 3, 4)) + 0.5
    condition = np.array([True, False, True], dtype=bool)
    axis = -2
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    condition = np.array([1, 1, 0, 0, 1], dtype=np.int32)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.empty((0, 4), dtype=np.float32)
    condition = np.array([], dtype=bool)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.arange(12, dtype=np.int32).reshape(3, 2, 2)
    condition = np.array([False, False, False], dtype=bool)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.arange(60, dtype=np.int32).reshape(3, 4, 5)
    condition = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    axis = 2
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    a = np.zeros((2, 0, 3), dtype=np.float64)
    condition = np.array([], dtype=bool)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"condition": condition, "a": a, "axis": axis}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.compress"] = tf_experimental_numpy_compress_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_rot90_inputs():
    list_of_inputs = []

    m = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    k = 1
    axes = (0, 1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.array([[1.1, -2.2, 3.3], [4.4, 5.5, -6.6]], dtype=np.float64)
    k = -1
    axes = (0, 1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    k = 2
    axes = (1, 2)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    k = 3
    axes = (0, 2)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.arange(2 * 2 * 3 * 4, dtype=np.uint8).reshape(2, 2, 3, 4)
    k = 1
    axes = (2, 3)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.array([[1 + 2j, 3 - 4j], [5 + 0j, -6 + 1j]], dtype=np.complex64)
    k = 2
    axes = (0, 1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = (np.arange(6) % 2 == 0).reshape(3, 2)
    k = 3
    axes = (-2, -1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.arange(4, dtype=np.int64).reshape(4, 1)
    k = 7
    axes = (0, 1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.arange(2 * 1 * 3 * 4 * 5, dtype=np.float16).reshape(2, 1, 3, 4, 5)
    k = -2
    axes = (0, 3)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.arange(20, dtype=np.int8).reshape(4, 5)
    k = 1
    axes = (1, 0)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    m = np.empty((0, 3), dtype=np.float32)
    k = 1
    axes = (0, 1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    base = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5)
    m = base + 1j * base
    k = -3
    axes = (-3, -1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k, "axes": axes}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.rot90"] = tf_experimental_numpy_rot90_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_image_adjust_hue_inputs():
    rs = np.random.RandomState(42)
    list_of_inputs = []

    # Input 1
    image = np.linspace(0.0, 1.0, num=12, dtype=np.float32).reshape(2, 2, 3)
    delta = 0.2
    name = "hue_pos_0_2_2x2x3_f32"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 2
    image = np.arange(1, 19, dtype=np.int32).reshape(3, 2, 3)
    delta = -0.5
    name = "hue_neg_0_5_3x2x3_i32"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 3
    image = rs.randint(0, 256, size=(1, 2, 2, 3), dtype=np.uint8)
    delta = 1.0
    name = "hue_pos_1_0_1x2x2x3_u8"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 4
    image = (rs.rand(4, 5, 3) * 10.0).astype(np.float64)
    delta = 0.75
    name = "hue_pos_0_75_4x5x3_f64"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 5
    image = rs.randint(-1000, 1000, size=(2, 3, 4, 4, 3), dtype=np.int32)
    delta = -1.0
    name = "hue_neg_1_0_2x3x4x4x3_i32"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 6
    image = rs.rand(10, 3, 3).astype(np.float16)
    delta = 0.0
    name = "hue_zero_10x3x3_f16"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 7
    image = (np.arange(8 * 8 * 3, dtype=np.int64).reshape(8, 8, 3) + 1000)
    delta = 1.0 / 3.0
    name = "hue_pos_third_8x8x3_i64"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 8
    image = np.array([[[[100, 200, 30]]], [[[40, 50, 60]]]], dtype=np.uint8)
    delta = -0.75
    name = "hue_neg_0_75_2x1x1x3_u8"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 9
    image = rs.randint(0, 256, size=(5, 6, 3), dtype=np.uint8)
    delta = 0.999
    name = "hue_pos_0_999_5x6x3_u8"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 10
    image = (rs.rand(3, 3, 3) * 2.5).astype(np.float32)
    delta = -0.25
    name = "hue_neg_0_25_3x3x3_f32"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 11
    image = (rs.rand(2, 4, 3) * 255).astype(np.float32)
    delta = 0.1
    name = "hue_pos_0_1_2x4x3_f32"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    # Input 12
    image = rs.rand(1, 7, 7, 3).astype(np.float32)
    delta = -0.1
    name = "hue_neg_0_1_1x7x7x3_f32"
    list_of_inputs.append(copy.deepcopy({"image": image, "delta": delta, "name": name}))

    return list_of_inputs

generated_inputs["tf.image.adjust_hue"] = tf_image_adjust_hue_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_image_flip_left_right_inputs():
    list_of_inputs = []

    # Input 1: 3D float32 RGB
    image = np.array(
        [[[1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0]],
         [[7.0, 8.0, 9.0],
          [10.0, 11.0, 12.0]]],
        dtype=np.float32
    )
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 2: 4D float32 batch with RGB
    image = np.arange(1 * 2 * 4 * 3, dtype=np.float32).reshape(1, 2, 4, 3)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 3: 3D int32 mixed values
    image = np.random.randint(-100, 100, size=(5, 4, 3), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 4: 4D int32 grayscale (channels=1)
    image = np.arange(2 * 3 * 3 * 1, dtype=np.int32).reshape(2, 3, 3, 1)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 5: 3D bool single channel
    image = np.array([[[True], [False]],
                      [[False], [True]]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 6: 4D float64 RGBA
    image = np.random.randn(3, 1, 5, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 7: 3D float32 width=1 with 2 channels
    image = np.array([[[1.0, -1.0]],
                      [[2.5, -2.5]],
                      [[3.75, -3.75]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 8: 4D int64 small batch RGB
    image = np.array(range(-24, 0), dtype=np.int64).reshape(2, 2, 2, 3)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 9: 3D float16 4-channel
    image = (np.random.randn(3, 4, 4)).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 10: 4D float32 larger HxW RGB
    image = np.random.randn(4, 8, 7, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 11: 4D bool grayscale batch
    image = (np.random.rand(1, 4, 5, 1) > 0.5).astype(bool)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    # Input 12: 3D float32 non-contiguous view (stride on width)
    base = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3)
    image = base[:, ::2, :]
    list_of_inputs.append(copy.deepcopy({"image": image}))

    return list_of_inputs

generated_inputs["tf.image.flip_left_right"] = tf_image_flip_left_right_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_crop_inputs():
    list_of_inputs = []

    value = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
    size = np.array([4, 4, 3], dtype=np.int32)
    seed = np.array([123, 456], dtype=np.int32)
    name = "crop_rgb_uint8"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.randn(10, 15).astype(np.float32)
    size = np.array([5, 7], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    name = "crop_2d_float32"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.arange(20, dtype=np.int32)
    size = np.array([10], dtype=np.int32)
    seed = np.array([42, 24], dtype=np.int32)
    name = "crop_1d_int32"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.rand(5, 32, 32, 3).astype(np.float64)
    size = np.array([3, 16, 16, 3], dtype=np.int32)
    seed = np.array([7, 11], dtype=np.int32)
    name = "crop_4d_batch"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.randint(-100, 100, (6, 4, 1), dtype=np.int32)
    size = np.array([3, 2, 1], dtype=np.int32)
    seed = np.array([101, 202], dtype=np.int32)
    name = "crop_neg_ints"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.randn(2, 3, 4, 5, 6).astype(np.float64)
    size = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    seed = np.array([31415, 92653], dtype=np.int32)
    name = "crop_5d_float64"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = (np.random.rand(9, 9, 1) > 0.5).astype(np.bool_)
    size = np.array([5, 5, 1], dtype=np.int32)
    seed = np.array([0, 1], dtype=np.int32)
    name = "crop_bool_mask"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.rand(4, 3, 28, 28).astype(np.float32)
    size = np.array([2, 3, 14, 14], dtype=np.int32)
    seed = np.array([1234, 5678], dtype=np.int32)
    name = "crop_channels_first"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.randint(0, 10, (7, 11), dtype=np.int16)
    size = np.array([7, 11], dtype=np.int32)
    seed = np.array([9, 99], dtype=np.int32)
    name = "crop_full_nochange"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.randint(0, 5, (12, 12, 4), dtype=np.int8)
    size = np.array([12, 8, 4], dtype=np.int32)
    seed = np.array([555, 666], dtype=np.int32)
    name = "crop_reduce_width"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.rand(3, 6, 7, 2).astype(np.float16)
    size = np.array([3, 5, 6, 2], dtype=np.int32)
    seed = np.array([2021, 2022], dtype=np.int32)
    name = "crop_float16"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    value = np.random.randint(-1000, 1000, (2, 5), dtype=np.int64)
    size = np.array([1, 4], dtype=np.int32)
    seed = np.array([888888, 999999], dtype=np.int32)
    name = "crop_2d_int64"
    list_of_inputs.append(copy.deepcopy({"value": value, "size": size, "seed": seed, "name": name}))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_crop"] = tf_image_stateless_random_crop_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_math_bessel_i0e_inputs():
    list_of_inputs = []

    x = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
    name = "vector_f32_basic"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[0.0, 2.0, -2.0], [10.0, -10.0, 1e-3]], dtype=np.float64)
    name = "matrix_f64_mixed_vals"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array(3.0, dtype=np.float32)
    name = "scalar_f32_positive"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([], dtype=np.float32)
    name = "empty_vector_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.linspace(-20.0, 20.0, 9, dtype=np.float64)
    name = "linspace_f64_wide_range"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.arange(-5, 6, dtype=np.float32)
    name = "range_f32_neg_to_pos"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.random.uniform(-50, 50, size=(3, 4, 2)).astype(np.float16)
    name = "random_3d_f16"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[np.inf, -np.inf, np.nan]], dtype=np.float32)
    name = "special_values_inf_nan_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x_base = np.arange(24, dtype=np.float64).reshape(4, 6)
    x = x_base[:, ::2] - 12.5
    name = "noncontiguous_slice_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.zeros((0, 3), dtype=np.float32)
    name = "empty_2d_rows_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([1e-8, -1e-8, 1e-12, -1e-12], dtype=np.float64)
    name = "very_small_values_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([100.0, -100.0, 300.0, -300.0], dtype=np.float32)
    name = "large_magnitude_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = (np.eye(5, dtype=np.float32) - 0.5)
    name = "identity_shifted_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.transpose(np.arange(8, dtype=np.float16).reshape(2, 2, 2), (1, 0, 2)).astype(np.float16) - 4
    name = "transposed_3d_f16"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    return list_of_inputs

generated_inputs["tf.math.bessel_i0e"] = tf_math_bessel_i0e_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_l2_loss_inputs():
    list_of_inputs = []

    t = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    name = "basic_float32_1d"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = (np.arange(12, dtype=np.float64).reshape(3, 4) - 5.5)
    name = "float64_2d_arange_shifted"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([[[1, -1], [2, -2]], [[3, -3], [4, -4]]], dtype=np.float16)
    name = "float16_3d_mixed"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array(3.14159265, dtype=np.float32)
    name = "float32_scalar"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.zeros((2, 3, 4, 1), dtype=np.float32)
    name = "float32_4d_zeros"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([], dtype=np.float32)
    name = "float32_empty"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    t = base[::2, 1::2, ::2]
    name = "float64_3d_strided_view"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([1e20, -1e20, 3e19], dtype=np.float32)
    name = "float32_large_magnitudes"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    if hasattr(np, "bfloat16"):
        t = np.linspace(-1.0, 1.0, 10, dtype=np.float32).astype(np.bfloat16)
        name = "bfloat16_1d_linspace"
    else:
        t = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
        name = "float32_1d_linspace_fallback"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = -3 * np.ones((5, 5), dtype=np.float16)
    name = "float16_2d_all_negative"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float64)
    name = "float64_with_nan_inf"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.ones((1, 2, 1, 2, 3), dtype=np.float32)
    name = "float32_5d_ones"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([-7.0], dtype=np.float32)
    name = "float32_single_element"
    input_dict = {"t": t, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.l2_loss"] = tf_nn_l2_loss_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_selu_inputs():
    list_of_inputs = []

    features = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    name = "selu_vec_f32"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([[-1.5, 0.0, 2.5], [3.3, -4.4, 5.5]], dtype=np.float64)
    name = "selu_matrix_f64"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([[[0.1, -0.1], [0.5, -0.5]], [[-1.2, 1.2], [2.0, -2.0]]], dtype=np.float16)
    name = "selu_3d_f16"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array(0.0, dtype=np.float32)
    name = "selu_scalar_zero"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.linspace(-2.0, 2.0, num=24, dtype=np.float32).reshape(2, 3, 4, 1)
    name = "selu_4d_f32"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([], dtype=np.float32)
    name = "selu_empty_f32"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    base = np.linspace(-3.0, 3.0, 12, dtype=np.float32)
    features = base[::2]
    name = "selu_noncontig_vec"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([[-10.0, -1.0, 0.0], [1.0, 5.5, 10.0]], dtype=np.float32)
    name = "selu_2x3_f32"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([-100.0, -20.0, 20.0, 100.0], dtype=np.float32)
    name = "selu_extreme_vals"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
    name = "selu_nan_inf"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    arr = np.arange(16, dtype=np.float64) - 8.0
    features = (arr / 4.0).reshape(1, 2, 2, 2, 2)
    name = "selu_5d_f64"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([1e-4, -1e-4, 1e-2, -1e-2], dtype=np.float16)
    name = "selu_small_f16"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    return list_of_inputs

generated_inputs["tf.nn.selu"] = tf_nn_selu_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_ones_inputs():
    list_of_inputs = []

    shape = [3, 4]
    dtype = np.int32
    name = "ones_int32_2d"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = []
    dtype = np.float32
    name = "ones_scalar_f32"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [0]
    dtype = np.float64
    name = "ones_len0_dim"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [2, 0]
    dtype = np.int64
    name = "ones_zero_second_dim"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [1, 2, 3]
    dtype = np.float16
    name = "ones_3d_f16"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [2, 2, 2, 2]
    dtype = np.complex64
    name = "ones_4d_c64"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [5]
    dtype = np.bool_
    name = "ones_1d_bool"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [2, 3, 4, 5, 6]
    dtype = np.uint8
    name = "ones_5d_u8"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [10]
    dtype = np.int8
    name = "ones_1d_i8"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [2, 3]
    dtype = np.float32
    name = "ones_2d_f32"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [3, 1, 0]
    dtype = np.int16
    name = "ones_zero_last_dim"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = [4, 4]
    dtype = np.complex128
    name = "ones_2d_c128"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    return list_of_inputs

generated_inputs["tf.ones_1"] = tf_ones_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_ones_2_inputs():
    list_of_inputs = []

    shape = (3, 4)
    dtype = np.int32
    name = "ones_i32_3x4"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (0,)
    dtype = np.float32
    name = "ones_f32_len0"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = ()
    dtype = np.float64
    name = "ones_scalar_f64"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (2, 0, 3)
    dtype = np.bool_
    name = "ones_bool_2x0x3"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (1,)
    dtype = np.complex64
    name = "ones_c64_len1"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (5, 1)
    dtype = np.uint8
    name = "ones_u8_5x1"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (2, 3, 4, 5)
    dtype = np.dtype('float16')
    name = "ones_f16_2x3x4x5"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (1, 2, 3)
    dtype = np.int64
    name = "ones_i64_1x2x3"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (10,)
    dtype = np.dtype('complex128')
    name = "ones_c128_len10"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (2, 2, 2, 2, 2)
    dtype = np.int16
    name = "ones_i16_5d"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (1, 0, 0)
    dtype = np.dtype('uint16')
    name = "ones_u16_1x0x0"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    shape = (7,)
    dtype = np.float32
    name = "ones_f32_len7_unicode_名"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    return list_of_inputs

generated_inputs["tf.ones_2"] = tf_ones_2_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_ones_inputs():
    list_of_inputs = []

    # 1
    shape = np.array([3, 4], dtype=np.int32)
    dtype = np.int32
    name = "ones_int32_2d"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 2
    shape = np.array([2, 0], dtype=np.int32)
    dtype = np.float32
    name = "ones_float32_with_zero_dim"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 3
    shape = np.array([5], dtype=np.int32)
    dtype = np.bool_
    name = "ones_bool_1d"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 4
    shape = np.array([1, 1, 1], dtype=np.int32)
    dtype = np.float64
    name = "ones_float64_3d"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 5
    shape = np.array([], dtype=np.int32)
    dtype = np.float32
    name = "ones_scalar_default_float32"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 6
    shape = np.array([2, 3, 4], dtype=np.int32)
    dtype = np.complex64
    name = "ones_complex64_3d"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 7
    shape = np.array([1, 2, 3, 4], dtype=np.int32)
    dtype = np.int64
    name = "ones_int64_4d"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 8
    shape = np.array([10], dtype=np.int32)
    dtype = np.uint8
    name = "ones_uint8_1d_len10"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 9
    shape = np.array([2, 2], dtype=np.int32)
    dtype = np.float16
    name = "ones_float16_2x2"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 10
    shape = np.array([0, 3, 0], dtype=np.int32)
    dtype = np.float16
    name = "ones_float16_with_zeros"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 11
    shape = np.array([4, 1], dtype=np.int32)
    dtype = np.complex128
    name = "ones_complex128_2d"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    # 12
    shape = np.array([1], dtype=np.int32)
    dtype = np.int8
    name = "ones_int8_singleton"
    layout = None
    list_of_inputs.append(copy.deepcopy({"shape": shape, "dtype": dtype, "name": name, "layout": layout}))

    return list_of_inputs

generated_inputs["tf.ones_3"] = tf_ones_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_raw_ops_DrawBoundingBoxes_inputs():
    list_of_inputs = []

    images = np.array([[[[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]],
                        [[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]]]], dtype=np.float32)
    boxes = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    input_dict = {
        "name": "case1_basic_rgb",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = np.linspace(0, 1, 2 * 5 * 4 * 1, dtype=np.float32).reshape(2, 5, 4, 1)
    boxes = np.array([
        [[0.0, 0.0, 1.0, 1.0], [0.2, 0.2, 0.6, 0.8]],
        [[0.1, 0.1, 0.9, 0.9], [0.0, 0.5, 1.0, 0.5]]
    ], dtype=np.float32)
    input_dict = {
        "name": "case2_batch_grayscale",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = np.zeros((1, 100, 200, 1), dtype=np.float16)
    boxes = np.array([[
        [0.05, 0.1, 0.3, 0.4],
        [0.4, 0.2, 0.9, 0.8],
        [0.0, 0.0, 1.0, 1.0]
    ]], dtype=np.float32)
    input_dict = {
        "name": "case3_large_float16",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = (np.random.randn(3, 10, 10, 4)).astype(np.float32)
    boxes = np.zeros((3, 0, 4), dtype=np.float32)
    input_dict = {
        "name": "case4_no_boxes",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = (np.random.randn(1, 8, 8, 3)).astype(np.float32)
    boxes = np.array([[
        [-0.1, -0.1, 1.2, 1.3],
        [0.3, 0.3, 0.7, 0.7]
    ]], dtype=np.float32)
    input_dict = {
        "name": "case5_out_of_range_boxes",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = np.array([[[[0.25]]]], dtype=np.float32)
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    input_dict = {
        "name": "case6_minimal",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = (np.random.rand(2, 4, 6, 3)).astype(np.float16)
    boxes = np.array([
        [[0.1, 0.1, 0.9, 0.9]],
        [[0.3, 0.0, 0.7, 1.0]]
    ], dtype=np.float32)
    input_dict = {
        "name": "case7_float16_rgb",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = np.linspace(0, 1, 1 * 64 * 64 * 1, dtype=np.float32).reshape(1, 64, 64, 1)
    boxes = np.array([[
        [0.0, 0.0, 1.0, 1.0],
        [0.01, 0.01, 0.99, 0.99],
        [0.25, 0.25, 0.75, 0.75],
        [0.5, 0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0, 0.5]
    ]], dtype=np.float32)
    input_dict = {
        "name": "case8_many_boxes",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = (np.random.rand(4, 16, 16, 3)).astype(np.float32)
    boxes = np.array([
        [[0.1, 0.1, 0.4, 0.4], [0.6, 0.6, 0.9, 0.9]],
        [[0.2, 0.2, 0.8, 0.7], [0.0, 0.0, 0.2, 0.3]],
        [[0.0, 0.8, 1.0, 1.0], [0.4, 0.4, 0.6, 0.6]],
        [[0.3, 0.1, 0.5, 0.9], [0.1, 0.7, 0.2, 0.9]]
    ], dtype=np.float32)
    input_dict = {
        "name": "case9_batch4_rgb",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = (np.random.randn(5, 3, 3, 4)).astype(np.float32)
    boxes = np.array([
        [[0.0, 0.0, 1.0, 1.0]],
        [[0.2, 0.2, 0.8, 0.8]],
        [[0.3, 0.1, 0.7, 0.9]],
        [[0.0, 0.4, 1.0, 0.6]],
        [[0.5, 0.0, 0.5, 1.0]]
    ], dtype=np.float32)
    input_dict = {
        "name": "case10_depth4",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = (np.random.rand(1, 32, 48, 4)).astype(np.float32)
    boxes = np.array([[
        [0.1, 0.1, 0.1, 0.5],
        [0.2, 0.2, 0.8, 0.2],
        [0.0, 0.0, 1.0, 0.2],
        [0.3, 0.4, 0.9, 0.95]
    ]], dtype=np.float32)
    input_dict = {
        "name": "case11_rgba_degenerate",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    images = (np.random.rand(2, 7, 3, 1)).astype(np.float16)
    boxes = np.array([
        [[0.0, 0.0, 0.5, 1.0], [0.2, 0.2, 0.6, 0.8], [0.6, 0.1, 1.0, 0.9]],
        [[0.1, 0.2, 0.9, 0.7], [0.0, 0.5, 0.4, 0.9], [0.3, 0.0, 0.7, 0.4]]
    ], dtype=np.float32)
    input_dict = {
        "name": "case12_float16_small_wide",
        "images": images,
        "boxes": boxes
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DrawBoundingBoxes"] = tf_raw_ops_DrawBoundingBoxes_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_fake_quant_with_min_max_vars_per_channel_inputs():
    list_of_inputs = []

    rs = np.random.RandomState(123)

    inputs = np.array([-1.2, 0.0, 0.7], dtype=np.float32)
    min_v = np.array([-1.0, 0.0, -0.5], dtype=np.float32)
    max_v = np.array([1.0, 2.0, 0.75], dtype=np.float32)
    input_dict = {"num_bits": 8, "narrow_range": False, "name": "fqpc_1", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = np.array([[-2.0, -1.0, 0.0, 1.0],
                       [2.0, 3.0, -3.0, 4.0]], dtype=np.float32)
    min_v = np.array([-1.0, 0.1, -0.5, 2.0], dtype=np.float32)
    max_v = np.array([0.0, 2.0, 1.0, 3.0], dtype=np.float32)
    input_dict = {"num_bits": 2, "narrow_range": True, "name": "fqpc_2", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3) / 3.0 - 2.0
    min_v = np.array([-1.0, 0.2, 0.0], dtype=np.float32)
    max_v = np.array([0.9, 2.0, 3.5], dtype=np.float32)
    input_dict = {"num_bits": 6, "narrow_range": False, "name": "fqpc_3", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = rs.randn(3, 4, 4, 1).astype(np.float32)
    min_v = np.array([-2.0], dtype=np.float32)
    max_v = np.array([-0.5], dtype=np.float32)
    input_dict = {"num_bits": 16, "narrow_range": False, "name": "fqpc_4", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = np.array([[10.0, -10.0],
                       [0.5, -0.25],
                       [2.0, -3.0],
                       [-1.0, 1.0]], dtype=np.float32)
    min_v = np.array([-1.0, -2.0], dtype=np.float32)
    max_v = np.array([0.0, 0.0], dtype=np.float32)
    input_dict = {"num_bits": 12, "narrow_range": True, "name": "fqpc_5", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = np.array([0.75], dtype=np.float32)
    min_v = np.array([-0.1], dtype=np.float32)
    max_v = np.array([0.25], dtype=np.float32)
    input_dict = {"num_bits": 7, "narrow_range": False, "name": "fqpc_6", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = rs.randn(2, 3, 4, 5).astype(np.float32) * 2.0
    min_v = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    max_v = np.array([0.0, 1.0, 2.0, 2.5, 3.0], dtype=np.float32)
    input_dict = {"num_bits": 9, "narrow_range": False, "name": "fqpc_7", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = (np.arange(35, dtype=np.float32).reshape(5, 7) - 17.0) / 5.0
    min_v = np.array([-1.5, -1.0, -0.5, 0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    max_v = np.array([0.0, 0.5, 0.6, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    input_dict = {"num_bits": 3, "narrow_range": True, "name": "fqpc_8", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = np.linspace(-2.0, 2.0, 8, dtype=np.float32)
    min_v = np.array([-1.0] * 8, dtype=np.float32)
    max_v = np.array([1.0] * 8, dtype=np.float32)
    input_dict = {"num_bits": 10, "narrow_range": False, "name": "fqpc_9", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = rs.uniform(-1.0, 1.0, size=(2, 2, 2, 4)).astype(np.float32)
    min_v = np.array([-0.2, -0.1, 0.0, 0.3], dtype=np.float32)
    max_v = np.array([0.2, 0.1, 1.0, 0.9], dtype=np.float32)
    input_dict = {"num_bits": 15, "narrow_range": True, "name": "fqpc_10", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel"] = tf_raw_ops_fake_quant_with_min_max_vars_per_channel_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_segment_sum_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([[1, 2, 3, 4],
                     [-1, -2, -3, -4],
                     [5, 6, 7, 8]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "basic_one_segment",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([[1, 2, 3, 4],
                     [-1, -2, -3, -4],
                     [5, 6, 7, 8]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "two_segments",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([[1, 2, 3, 4],
                     [-1, -2, -3, -4],
                     [5, 6, 7, 8]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "select_all_two_segments",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([-3, -2, -1, 0, 1], dtype=np.int32)
    indices = np.array([0, 2, 4], dtype=np.int64)
    segment_ids = np.array([0, 1, 1], dtype=np.int64)
    input_dict = {
        "sparse_gradient": False,
        "name": "int32_vector_multi_segments",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
            [[13, 14], [15, 16]],
        ],
        dtype=np.uint8
    )
    indices = np.array([0, 3, 1, 1], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 2], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "uint8_3d_var_segments",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array(
        [
            [1.5, -2.0, 3.25],
            [-4.5, 5.5, -6.75],
            [7.125, -8.875, 9.0],
            [0.5, 0.25, -0.75],
            [2.5, -3.5, 4.5],
            [-1.25, 2.0, -3.0]
        ],
        dtype=np.float64
    )
    indices = np.array([5, 3, 0, 2], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    input_dict = {
        "sparse_gradient": False,
        "name": "float64_matrix_two_segments",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = np.array(
        [
            [[[1, -1], [2, -2]], [[3, -3], [4, -4]]],
            [[[5, -5], [6, -6]], [[7, -7], [8, -8]]],
            [[[9, -9], [10, -10]], [[11, -11], [12, -12]]]
        ],
        dtype=np.int64
    )
    indices = np.array([0, 2, 2], dtype=np.int32)
    segment_ids = np.array([1, 1, 5], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "int64_4d_with_gapped_segments",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (int8)
    data = np.array([[-128, -1, 0, 1, 127],
                     [10, -20, 30, -40, 50]], dtype=np.int8)
    indices = np.array([1, 0], dtype=np.int32)
    segment_ids = np.array([3, 3], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "int8_two_rows_single_segment_id3",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    data = np.array(
        [
            [0.5, -1.0, 1.5, -2.0],
            [2.5, -3.0, 3.5, -4.0],
            [4.5, -5.0, 5.5, -6.0],
            [6.5, -7.0, 7.5, -8.0],
            [8.5, -9.0, 9.5, -10.0]
        ],
        dtype=np.float16
    )
    indices = np.array([0, 4, 2, 1, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "float16_multi_segments_more_pairs",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (replace previous uint16 with int32)
    data = np.array([[[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]]], dtype=np.int32)
    indices = np.array([0], dtype=np.int32)
    segment_ids = np.array([0], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "int32_single_row_single_segment",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (replace previous uint32 with float32)
    data = np.array([10.0, 20.0, -30.0, 40.0, -50.0, 60.0, -70.0], dtype=np.float32)
    indices = np.array([1, 1, 6, 0], dtype=np.int64)
    segment_ids = np.array([0, 1, 1, 3], dtype=np.int64)
    input_dict = {
        "sparse_gradient": False,
        "name": "float32_vector_with_duplicates",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (replace previous uint64 with int16)
    data = (np.arange(15, dtype=np.int16) - 7).reshape(3, 5)
    indices = np.array([2, 0, 1], dtype=np.int32)
    segment_ids = np.array([0, 2, 2], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "int16_matrix_nonconsecutive_segments",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13
    data = np.array([-1000, 2000, -3000, 4000], dtype=np.int16)
    indices = np.array([0, 3, 2, 2, 1], dtype=np.int64)
    segment_ids = np.array([0, 1, 1, 1, 3], dtype=np.int64)
    input_dict = {
        "sparse_gradient": False,
        "name": "int16_vector_three_segments",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14
    data = np.arange(3*1*2*1*2, dtype=np.float32).reshape(3, 1, 2, 1, 2) - 5.0
    indices = np.array([2, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 2, 2], dtype=np.int32)
    input_dict = {
        "sparse_gradient": True,
        "name": "float32_5d_gapped_segments",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSegmentSum"] = tf_sparse_segment_sum_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_fft2d_inputs():
    rs = np.random.RandomState(0)
    list_of_inputs = []

    arr1 = np.array([[1+2j, -3+4j], [0-1j, 2+0j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr1, "name": "simple_2x2_c64"}))

    arr2 = np.array(
        [
            [1-1j, 2+3j, -4-5j, 6+0j, -7+8j],
            [9-10j, 0+0j, -1+2j, 3-4j, 5+6j],
            [-2-3j, 4+5j, -6+7j, 8-9j, 10+11j],
        ],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"input": arr2, "name": "rect_3x5_c128"}))

    arr3 = (rs.randn(4, 8, 8) + 1j * rs.randn(4, 8, 8)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr3, "name": "batch_4x8x8_c64"}))

    arr4 = (rs.randn(2, 3, 4, 6) + 1j * rs.randn(2, 3, 4, 6)).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr4, "name": "nd_2x3x4x6_c128"}))

    arr5 = np.array([[3-4j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr5, "name": "single_1x1_c64"}))

    arr6 = (np.arange(7).reshape(1, 7) + 1j * (-np.arange(7).reshape(1, 7))).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr6, "name": "row_1x7_c128"}))

    arr7 = (np.linspace(-4, 4, 9).reshape(9, 1) + 1j * (-2.0) * np.ones((9, 1))).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr7, "name": "col_9x1_c64"}))

    arr8 = np.arange(36, dtype=np.float64).reshape(6, 6).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr8, "name": "real_only_6x6_c128"}))

    arr9 = (1j * np.arange(20).reshape(5, 4)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr9, "name": "imag_only_5x4_c64"}))

    arr10 = (rs.randn(3, 2, 5, 5) + 1j * rs.randn(3, 2, 5, 5)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr10, "name": "channels_3x2x5x5_c64"}))

    arr11 = (1e5 * (rs.randn(10, 10) + 1j * rs.randn(10, 10))).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr11, "name": "large_magnitude_10x10_c64"}))

    base = np.arange(49, dtype=np.float32).reshape(7, 7)
    arr12 = (base.T - 25 + 1j * (base[::-1, :] - 10)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr12, "name": "transposed_and_sliced_7x7_c64"}))

    return list_of_inputs

generated_inputs["tf.signal.fft2d"] = tf_signal_fft2d_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
rs = np.random.RandomState(0)

def tf_signal_rfft2d_inputs():
    list_of_inputs = []

    input_tensor = rs.randn(4, 6).astype(np.float32)
    fft_length = [4, 6]
    name = "case_exact_2d_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.uniform(-5, 5, size=(5, 7)).astype(np.float32)
    fft_length = [4, 6]
    name = "case_crop_both_2d_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.randn(3, 3).astype(np.float32)
    fft_length = [5, 4]
    name = "case_pad_both_2d_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.randn(2, 8, 5).astype(np.float32)
    fft_length = [8, 5]
    name = "case_batch_exact_3d_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.randn(3, 4, 4).astype(np.float32)
    fft_length = [2, 6]
    name = "case_batch_crop_pad_3d_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.randn(2, 3, 6, 10).astype(np.float32)
    fft_length = [6, 10]
    name = "case_4d_exact_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.randn(1, 2, 7, 5).astype(np.float64)
    fft_length = [8, 8]
    name = "case_4d_pad_both_f64"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.randn(2, 2, 2, 4, 4).astype(np.float32)
    fft_length = [4, 4]
    name = "case_5d_exact_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = np.linspace(-1.0, 1.0, num=8, dtype=np.float32).reshape(1, 8)
    fft_length = [1, 8]
    name = "case_degenerate_row_2d_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = np.full((10, 1, 1), 3.14, dtype=np.float64)
    fft_length = [1, 1]
    name = "case_degenerate_inner_3d_f64"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.randn(9, 1).astype(np.float64)
    fft_length = [16, 1]
    name = "case_pad_rows_2d_f64"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    input_tensor = rs.uniform(-1, 1, size=(5, 2, 9)).astype(np.float32)
    fft_length = [5, 7]
    name = "case_batch_pad_crop_3d_f32"
    list_of_inputs.append(copy.deepcopy({"input_tensor": input_tensor, "fft_length": fft_length, "name": name}))

    return list_of_inputs

generated_inputs["tf.signal.rfft2d"] = tf_signal_rfft2d_inputs()

import tensorflow as tf
import numpy as np
import torch
import copy

def tf_raw_ops_abort_inputs():
    list_of_inputs = []

    # Input 1
    error_msg = ""
    exit_without_error = False
    name = "abort_op_empty_msg"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    error_msg = "Abort now"
    exit_without_error = True
    name = "abort_ok_exit"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    error_msg = "Early termination requested"
    exit_without_error = False
    name = "early_termination"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    error_msg = "Line1\nLine2\tTabbed"
    exit_without_error = False
    name = "with_newlines"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    error_msg = "x" * 1024
    exit_without_error = True
    name = "very_long_error_msg"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    error_msg = "Ошибка завершения процесса"
    exit_without_error = False
    name = "cyrillic_name"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    error_msg = "終了します"
    exit_without_error = True
    name = "japanese_message"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    error_msg = "Aborting due to invalid state: -1"
    exit_without_error = False
    name = "negative_state"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    error_msg = "Special chars !@#$%^&*()[]{};:,.<>/?|`~"
    exit_without_error = True
    name = "special_chars"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    error_msg = "CaseSensitiveMessage"
    exit_without_error = False
    name = "MixedCaseName"
    input_dict = {
        "error_msg": error_msg,
        "exit_without_error": exit_without_error,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Abort"] = tf_raw_ops_abort_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_all_inputs():
    list_of_inputs = []

    input_arr = np.array([True, True, False, True, True], dtype=bool)
    axis = np.int32(0)
    keep_dims = False
    name = "all_case_1"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[True, False, True, True],
                          [True, True, True, False],
                          [False, True, True, True]], dtype=bool)
    axis = np.array([0], dtype=np.int64)
    keep_dims = True
    name = "all_case_2"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[True, False, True, True],
                           [True, True, True, True],
                           [True, True, False, True]],
                          [[True, True, True, True],
                           [True, False, True, True],
                           [True, True, True, True]]], dtype=bool)
    axis = np.array([-1], dtype=np.int32)
    keep_dims = False
    name = "all_case_3"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[True, True, False],
                           [True, False, True],
                           [True, True, True],
                           [False, True, True]],
                          [[True, True, True],
                           [True, True, False],
                           [True, False, True],
                           [True, True, True]]], dtype=bool)
    axis = np.array([0, 2], dtype=np.int64)
    keep_dims = True
    name = "all_case_4"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array(False, dtype=bool)
    axis = np.array([], dtype=np.int32)
    keep_dims = False
    name = "all_case_5_scalar_empty_axis"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[[True], [True], [False]]],
                          [[[True], [True], [True]]]], dtype=bool)
    axis = np.int64(2)
    keep_dims = True
    name = "all_case_6"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[[True, True],
                            [True, True]],
                           [[True, True],
                            [True, False]]],
                          [[[True, True],
                            [True, True]],
                           [[True, True],
                            [True, True]]]], dtype=bool)
    axis = np.array([0, 1, 2, 3], dtype=np.int32)
    keep_dims = False
    name = "all_case_7_reduce_all_axes"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[True, False, True, True, True]], dtype=bool)
    axis = np.array([-2], dtype=np.int64)
    keep_dims = True
    name = "all_case_8_negative_axis_scalar"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[True, True, True],
                           [True, True, True],
                           [True, True, True]],
                          [[True, True, True],
                           [True, True, True],
                           [True, True, True]],
                          [[True, True, True],
                           [True, True, True],
                           [False, True, True]]], dtype=bool)
    axis = np.int32(-3)
    keep_dims = False
    name = "all_case_9_negative_scalar_axis"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[[True], [True], [False]],
                           [[True], [True], [True]]],
                          [[[True], [False], [True]],
                           [[True], [True], [True]]]], dtype=bool)
    axis = np.array([1, 3], dtype=np.int64)
    keep_dims = True
    name = "all_case_10_multi_axes"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[True, True],
                          [True, True]], dtype=bool)
    axis = np.array([0, 1], dtype=np.int32)
    keep_dims = True
    name = "all_case_11_reduce_all_keepdims"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[True, False],
                           [True, True],
                           [True, True],
                           [True, True]],
                          [[True, True],
                           [False, True],
                           [True, True],
                           [True, True]],
                          [[True, True],
                           [True, True],
                           [True, False],
                           [True, True]],
                          [[True, True],
                           [True, True],
                           [True, True],
                           [True, True]]], dtype=bool)
    axis = np.array([1], dtype=np.int64)
    keep_dims = False
    name = "all_case_12_axis1"
    input_dict = {"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.All"] = tf_raw_ops_all_inputs()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(42)
tf.random.set_seed(42)

def tf_raw_ops_any_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.array([True, False, False, True], dtype=np.bool_)
    axis = np.array(0, dtype=np.int32)
    keep_dims = False
    name = "any_case_1"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 2
    input_arr = np.array([[True, False, True], [False, False, True]], dtype=np.bool_)
    axis = np.array([0], dtype=np.int64)
    keep_dims = False
    name = "any_case_2"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 3
    input_arr = np.array([[False, False, True], [True, False, False]], dtype=np.bool_)
    axis = np.array([1], dtype=np.int32)
    keep_dims = True
    name = "any_case_3"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 4
    input_arr = np.array([[[True, False], [False, False]],
                          [[True, True], [False, True]]], dtype=np.bool_)
    axis = np.array([0, 2], dtype=np.int64)
    keep_dims = False
    name = "any_case_4"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 5
    input_arr = (np.random.rand(2, 3, 4) > 0.7).astype(np.bool_)
    axis = np.array(-1, dtype=np.int32)
    keep_dims = True
    name = "any_case_5"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 6
    input_arr = np.array(True, dtype=np.bool_)
    axis = np.array([], dtype=np.int32)
    keep_dims = False
    name = "any_case_6_scalar_empty_axis"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 7
    input_arr = np.zeros((2, 0, 3, 1), dtype=np.bool_)
    axis = np.array([1], dtype=np.int64)
    keep_dims = False
    name = "any_case_7_zero_size_dim"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 8
    input_arr = np.array([], dtype=np.bool_)
    axis = np.array(0, dtype=np.int32)
    keep_dims = True
    name = "any_case_8_empty_vector"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 9
    input_arr = (np.random.rand(3, 4, 5) > 0.8).astype(np.bool_)
    axis = np.array([0, 1, 2], dtype=np.int64)
    keep_dims = False
    name = "any_case_9_reduce_all_axes"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 10
    input_arr = (np.random.rand(3, 4, 5) > 0.3).astype(np.bool_)
    axis = np.array([-2], dtype=np.int32)
    keep_dims = True
    name = "any_case_10_negative_axis"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 11
    input_arr = np.array([[False]], dtype=np.bool_)
    axis = np.array([0, 1], dtype=np.int32)
    keep_dims = False
    name = "any_case_11_single_element_all_axes"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    # Input 12
    input_arr = np.array([[True, False, True, False, False, True],
                          [False, False, False, False, True, False],
                          [True, True, False, False, False, False],
                          [False, True, True, False, False, False],
                          [False, False, True, True, False, True]], dtype=np.bool_)
    axis = np.array(-1, dtype=np.int64)
    keep_dims = False
    name = "any_case_12_last_axis"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims,
        "name": name,
        "input": input_arr,
        "axis": axis
    }))

    return list_of_inputs

generated_inputs["tf.raw_ops.Any"] = tf_raw_ops_any_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_asinh_inputs():
    list_of_inputs = []

    x = np.array([-np.inf, -2.0, -0.5, 0.0, 1.0, 1.2, 200.0, 10000.0, np.inf], dtype=np.float32)
    input_dict = {"name": "asinh_f32_1d_range", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-5.0, dtype=np.float64)
    input_dict = {"name": "asinh_f64_scalar_negative", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-10.0, -1.0], [0.0, 1.0], [10.0, 1000.0]], dtype=np.float64)
    input_dict = {"name": "asinh_f64_2d_mixed", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]], dtype=np.float16)
    input_dict = {"name": "asinh_f16_row_vector", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1+0j, 0+1j, -1+2j, -3-4j], dtype=np.complex64)
    input_dict = {"name": "asinh_c64_1d_complex_values", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1-1j, 2+2j], [-3+0.5j, 0-4j], [5+6j, -7-8j]], dtype=np.complex128)
    input_dict = {"name": "asinh_c128_2d_mixed", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([
        [[-np.inf, -3.0, np.nan], [-0.5, 0.0, 0.5]],
        [[1.0, 10.0, 1000.0], [np.inf, -1e-6, 1e-6]]
    ], dtype=np.float32)
    input_dict = {"name": "asinh_f32_3d_with_nan_inf", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.zeros((1, 2, 3, 4), dtype=np.float32)
    input_dict = {"name": "asinh_f32_4d_zeros", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    input_dict = {"name": "asinh_f32_empty_1d", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-12, -1e-12, 1e12, -1e12], dtype=np.float64)
    input_dict = {"name": "asinh_f64_extremes", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([complex(np.inf, 0.0), complex(np.nan, 1.0), complex(1.0, np.inf), complex(-np.inf, -np.inf)], dtype=np.complex64)
    input_dict = {"name": "asinh_c64_with_nan_inf", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.empty((2, 0, 3), dtype=np.float32)
    input_dict = {"name": "asinh_f32_empty_middle_dim", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Asinh"] = tf_raw_ops_asinh_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_checknumerics_inputs():
    list_of_inputs = []

    tensor = np.array(1.0, dtype=np.float32)
    input_dict = {
        "name": "check_scalar_f32",
        "tensor": tensor,
        "message": "Scalar float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([-3.5, 0.0, 2.2], dtype=np.float64)
    input_dict = {
        "name": "check_vector_f64",
        "tensor": tensor,
        "message": "Vector float64 with negatives"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[1e-4, -1e-3], [2.0, -2.5]], dtype=np.float16)
    input_dict = {
        "name": "check_matrix_f16",
        "tensor": tensor,
        "message": "Matrix float16 small values"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    input_dict = {
        "name": "check_3d_no_nan",
        "tensor": tensor,
        "message": "3D tensor finite values"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[[[1.0, -1.0], [0.0, 123.456]], [[-789.0, 1e-10], [1e20, -1e20]]]], dtype=np.float32)
    input_dict = {
        "name": "check_4d_finite_extremes",
        "tensor": tensor,
        "message": "4D tensor finite extremes"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([], dtype=np.float32)
    input_dict = {
        "name": "check_empty_f32",
        "tensor": tensor,
        "message": "Empty tensor"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[1.0, 2.0, -3.0], [4.5, -5.5, 6.25]], dtype=np.float64)
    input_dict = {
        "name": "check_matrix_f64",
        "tensor": tensor,
        "message": "Matrix float64 finite"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[1e-5, -1e-5], [6e-8, -6e-8]], dtype=np.float16)
    input_dict = {
        "name": "check_subnormal_f16",
        "tensor": tensor,
        "message": "Float16 with small magnitudes"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([1e10, -1e10, 3.1415927, -2.7182818], dtype=np.float32)
    input_dict = {
        "name": "check_vector_f32_large",
        "tensor": tensor,
        "message": "Vector float32 large finite"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[[0.1, 0.2, 0.3], [-0.4, -0.5, 0.6]], [[7.7, -8.8, 9.9], [10.01, -11.11, 12.12]]], dtype=np.float64)
    input_dict = {
        "name": "check_3d_f64",
        "tensor": tensor,
        "message": "3D float64 finite"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([42.0], dtype=np.float32)
    input_dict = {
        "name": "check_single_element",
        "tensor": tensor,
        "message": "Single finite element"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array(-123.456, dtype=np.float64)
    input_dict = {
        "name": "check_scalar_neg_f64",
        "tensor": tensor,
        "message": "Scalar negative float64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.CheckNumerics"] = tf_raw_ops_checknumerics_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_controltrigger_inputs():
    list_of_inputs = []

    input_dict = {"name": "control_trigger"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "ct_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "op_001"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "a"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "Zzz_Trigger_Name"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "scope1/scope2/ct"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "under_score_leading__"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "_private_name_"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "with_digits_1234567890"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "NESTED_scope/inner/ControlTrigger_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "long_name_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"name": "mixedCASE_Name_Op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ControlTrigger"] = tf_raw_ops_controltrigger_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_cosh_inputs():
    list_of_inputs = []

    # Input 1: float32 vector with infinities
    x = np.array([-np.inf, -9.0, -0.5, 1.0, 1.2, 2.0, 10.0, np.inf], dtype=np.float32)
    name = "cosh_case_float32_vector_inf"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 scalar
    x = np.array(0.0, dtype=np.float64)
    name = "cosh_case_float64_scalar_zero"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16 2D matrix with negatives and positives
    x = np.array([[-1.0, -2.0], [3.0, 4.0]], dtype=np.float16)
    name = "cosh_case_float16_matrix"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32 3D tensor
    x = np.array([[[0.1, -0.2, 0.3]], [[-0.4, 0.5, -0.6]]], dtype=np.float32)
    name = "cosh_case_float32_3d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 vector
    x = np.array([1+0j, 0+1j, -1+2j, -3-4j], dtype=np.complex64)
    name = "cosh_case_complex64_vector"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128 2D matrix
    x = np.array([[1+1j, -2+3j], [0-1j, -0.5+0.5j]], dtype=np.complex128)
    name = "cosh_case_complex128_matrix"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32 large magnitudes
    x = np.array([100.0, -100.0, 50.0, -50.0], dtype=np.float32)
    name = "cosh_case_float32_large_magnitudes"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32 empty vector
    x = np.array([], dtype=np.float32)
    name = "cosh_case_float32_empty"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64 empty first-dimension (0x3)
    x = np.zeros((0, 3), dtype=np.float64)
    name = "cosh_case_float64_empty_rows"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 with NaNs
    x = np.array([np.nan, -np.nan, 1.0, -1.0], dtype=np.float32)
    name = "cosh_case_float32_nans"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float32 very small values (subnormals)
    x = np.array([1e-8, -1e-8, 1e-10, -1e-10], dtype=np.float32)
    name = "cosh_case_float32_small_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: float32 4D tensor
    x = (np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2) - 12.0) / 5.0
    name = "cosh_case_float32_4d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Cosh"] = tf_raw_ops_cosh_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DataFormatDimMap_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array(2, dtype=np.int32)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "map_case_1",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([-1, 0, 1, 2, 3], dtype=np.int32)
    input_dict = {
        "src_format": "NCHW",
        "dst_format": "NHWC",
        "name": "map_case_2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[-4, -3], [-2, -1]], dtype=np.int64)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "map_case_3",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[[-4, -3], [-2, -1]], [[0, 1], [2, 3]]], dtype=np.int64)
    input_dict = {
        "src_format": "NCHW",
        "dst_format": "NCHW",
        "name": "map_case_4",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.zeros((7,), dtype=np.int32)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NHWC",
        "name": "map_case_5",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([
        [-4, -3, -2],
        [-1, 0, 1],
        [2, 3, -4],
        [-2, 1, 0]
    ], dtype=np.int32)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "map_case_6",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array(-4, dtype=np.int64)
    input_dict = {
        "src_format": "NCHW",
        "dst_format": "NHWC",
        "name": "map_case_7",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([[[-4, -1, 0, 3]]], dtype=np.int32)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "map_case_8",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    base = np.arange(25, dtype=np.int64)
    x = (base % 8) - 4
    x = x.reshape(5, 5)
    input_dict = {
        "src_format": "NCHW",
        "dst_format": "NHWC",
        "name": "map_case_9",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[[[-4, -3, -2]], [[-1, 0, 1]]]], dtype=np.int64)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "map_case_10",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    x = np.array([3, 2, 1, 0, -1, -2, -3, -4, -1, 0], dtype=np.int32)
    input_dict = {
        "src_format": "NCHW",
        "dst_format": "NCHW",
        "name": "map_case_11",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    x = np.array([[-2], [3], [-4]], dtype=np.int64)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NHWC",
        "name": "map_case_12",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DataFormatDimMap"] = tf_raw_ops_DataFormatDimMap_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_data_format_vec_permute_inputs():
    list_of_inputs = []

    x = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "case_1_nhwc_to_nchw_vec4",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 20], dtype=np.int64)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "case_2_nhwc_to_nchw_vec2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, 6], [2, 7], [3, 8], [4, 9]], dtype=np.int32)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "case_3_nhwc_to_nchw_mat4x2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[5, -5], [6, -6]], dtype=np.int64)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "case_4_nhwc_to_nchw_mat2x2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1, 0, -3, 4], dtype=np.int64)
    input_dict = {
        "src_format": "NCHW",
        "dst_format": "NHWC",
        "name": "case_5_nchw_to_nhwc_vec4",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([9, -8, 7, -6, 5], dtype=np.int32)
    input_dict = {
        "src_format": "NDHWC",
        "dst_format": "NCDHW",
        "name": "case_6_ndhwc_to_ncdhw_vec5",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([100000, -200000, 300000], dtype=np.int64)
    input_dict = {
        "src_format": "NDHWC",
        "dst_format": "NCDHW",
        "name": "case_7_ndhwc_to_ncdhw_vec3",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[10, -10], [20, -20], [30, -30], [40, -40], [50, -50]], dtype=np.int64)
    input_dict = {
        "src_format": "NDHWC",
        "dst_format": "NCDHW",
        "name": "case_8_ndhwc_to_ncdhw_mat5x2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    input_dict = {
        "src_format": "NDHWC",
        "dst_format": "NCDHW",
        "name": "case_9_ndhwc_to_ncdhw_mat3x2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([5, 4, 3, 2, 1], dtype=np.int32)
    input_dict = {
        "src_format": "NCDHW",
        "dst_format": "NDHWC",
        "name": "case_10_ncdhw_to_ndhwc_vec5",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[7, 14], [8, 16], [9, 18]], dtype=np.int64)
    input_dict = {
        "src_format": "NCDHW",
        "dst_format": "NDHWC",
        "name": "case_11_ncdhw_to_ndhwc_mat3x2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1000, 2000], [3000, 4000], [5000, 6000], [7000, 8000]], dtype=np.int32)
    input_dict = {
        "src_format": "NCHW",
        "dst_format": "NHWC",
        "name": "case_12_nchw_to_nhwc_mat4x2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DataFormatVecPermute"] = tf_raw_ops_data_format_vec_permute_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_debuggradientidentity_inputs():
    list_of_inputs = []

    arr = np.array(3.14, dtype=np.float32)
    input_dict = {"name": "scalar_f32_pi", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([-5, 0, 7, 42], dtype=np.int32)
    input_dict = {"name": "vec_i32_mixed", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[np.nan, -np.inf], [np.inf, -1.5]], dtype=np.float64)
    input_dict = {"name": "mat_f64_nan_inf", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[[True, False, True], [False, False, True]], [[True, True, False], [False, True, False]]], dtype=bool)
    input_dict = {"name": "tensor_bool_3d", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([], dtype=np.float32)
    input_dict = {"name": "empty_f32", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.zeros((1, 2, 0, 3), dtype=np.int64)
    input_dict = {"name": "int64_zero_dim_axis", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([1+2j, -3+0.5j, 0+0j], dtype=np.complex64)
    input_dict = {"name": "vec_c64", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[-1.2, 3.4], [0.0, 65504.0]], dtype=np.float16)
    input_dict = {"name": "mat_f16_range", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.arange(12, dtype=np.uint8).reshape(2, 1, 2, 3)
    input_dict = {"name": "tensor_u8_4d", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array(3-4j, dtype=np.complex128)
    input_dict = {"name": "scalar_c128", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DebugGradientIdentity"] = tf_raw_ops_debuggradientidentity_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_raw_ops_DiagPart_inputs():
    list_of_inputs = []

    # Input 1: 2D int32 square matrix
    input_arr = np.array(
        [[1, 0, 0, 0],
         [0, 2, 0, 0],
         [0, 0, 3, 0],
         [0, 0, 0, 4]], dtype=np.int32
    )
    input_dict = {"input": input_arr, "name": "case_2d_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 matrix with negatives
    input_arr = np.array(
        [[3.0, -1.2, 0.0],
         [2.3, -5.5, 7.8],
         [9.1, 4.2, -8.3]], dtype=np.float64
    )
    input_dict = {"input": input_arr, "name": "case_2d_float64_neg"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D float32 tensor [2,3,2,3]
    input_arr = np.arange(2*3*2*3, dtype=np.float32).reshape(2, 3, 2, 3)
    input_dict = {"input": input_arr, "name": "case_4d_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D int64 tensor [1,5,1,5] with random integers (including negatives)
    input_arr = np.random.randint(-50, 50, size=(1, 5, 1, 5), dtype=np.int64)
    input_dict = {"input": input_arr, "name": "case_4d_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 6D complex64 tensor [2,1,3,2,1,3]
    real = np.random.randn(2, 1, 3, 2, 1, 3).astype(np.float32)
    imag = np.random.randn(2, 1, 3, 2, 1, 3).astype(np.float32)
    input_arr = (real + 1j * imag).astype(np.complex64)
    input_dict = {"input": input_arr, "name": "case_6d_complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float16 1x1 tensor
    input_arr = np.array([[7.5]], dtype=np.float16)
    input_dict = {"input": input_arr, "name": "case_2d_float16_1x1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D complex128 tensor [3,3,3,3]
    real = np.random.randn(3, 3, 3, 3)
    imag = np.random.randn(3, 3, 3, 3)
    input_arr = (real + 1j * imag).astype(np.complex128)
    input_dict = {"input": input_arr, "name": "case_4d_complex128"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 empty square matrix [0,0]
    input_arr = np.empty((0, 0), dtype=np.float32)
    input_dict = {"input": input_arr, "name": "case_2d_empty_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 8D float32 tensor [2,2,1,3,2,2,1,3]
    input_arr = np.random.randn(2, 2, 1, 3, 2, 2, 1, 3).astype(np.float32)
    input_dict = {"input": input_arr, "name": "case_8d_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 6D int32 tensor [4,2,1,4,2,1] with negatives
    input_arr = (np.arange(4*2*1*4*2*1, dtype=np.int32) - 20).reshape(4, 2, 1, 4, 2, 1)
    input_dict = {"input": input_arr, "name": "case_6d_int32_neg"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D int64 square matrix with mixed values
    input_arr = np.array(
        [[-10, 2, 3],
         [4, 0, -6],
         [7, 8, 15]], dtype=np.int64
    )
    input_dict = {"input": input_arr, "name": "case_2d_int64_mixed"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 4D float64 tensor [1,1,1,1]
    input_arr = np.array([[[[42.0]]]], dtype=np.float64)
    input_dict = {"input": input_arr, "name": "case_4d_float64_singleton"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DiagPart"] = tf_raw_ops_DiagPart_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_digamma_inputs():
    list_of_inputs = []

    x = np.array(3.5, dtype=np.float32)
    name = "digamma_scalar_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([1.0, 2.0, 0.5, -0.5, -1.5], dtype=np.float32)
    name = "digamma_vector_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
    name = "digamma_matrix_f64"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[-0.1, -0.2], [1.0, 10.0]]], dtype=np.float16)
    name = "digamma_3d_f16"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[[1.0], [2.0]], [[-2.5], [3.5]]]], dtype=np.float32)
    name = "digamma_4d_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([], dtype=np.float32)
    name = "digamma_empty_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([1e-6, 1e6, 12345.678], dtype=np.float64)
    name = "digamma_large_vals_f64"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([-1.0001, -2.0001, -3.5, -0.9999, 0.0001], dtype=np.float32)
    name = "digamma_near_poles_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
    name = "digamma_nan_inf_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[[[0.5, 1.5]], [[2.5, 3.5]]], [[[4.5, 5.5]], [[6.5, 7.5]]]]], dtype=np.float64)
    name = "digamma_5d_f64"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[[2.0]]]], dtype=np.float16)
    name = "digamma_singleton_f16"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Digamma"] = tf_raw_ops_digamma_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Elu_inputs():
    list_of_inputs = []

    features = np.array(1.0, dtype=np.float32)
    input_dict = {"name": "elu_case_1", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array(0.0, dtype=np.float64)
    input_dict = {"name": "elu_case_2", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array(-1000.0, dtype=np.float32)
    input_dict = {"name": "elu_case_3", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"name": "elu_case_4", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-3.5, -0.1, 0.0], [0.1, 2.3, -7.8]], dtype=np.float16)
    input_dict = {"name": "elu_case_5", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.linspace(-3, 3, num=12).astype(np.float64).reshape(2, 2, 3)
    input_dict = {"name": "elu_case_6", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([], dtype=np.float32)
    input_dict = {"name": "elu_case_7", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([np.inf, -np.inf, np.nan, -0.0, 0.0, 3.14], dtype=np.float32)
    input_dict = {"name": "elu_case_8", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-1.5, 0.0, 1.5], [2.5, -2.5, 0.5]], dtype=np.float32).reshape(1, 2, 1, 3)
    input_dict = {"name": "elu_case_9", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([-20.0, -5.0, 5.0, 20.0], dtype=np.float64)
    input_dict = {"name": "elu_case_10", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.arange(-6, 6, dtype=np.float32).reshape(3, 4)[:, ::2]
    input_dict = {"name": "elu_case_11", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = (np.arange(100, dtype=np.float16).reshape(10, 10) - np.float16(50)) / np.float16(10)
    input_dict = {"name": "elu_case_12", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Elu"] = tf_raw_ops_Elu_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Exp_inputs():
    list_of_inputs = []

    x = np.array(0.0, dtype=np.float32)
    input_dict = {"name": "exp_input_1", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([2.0, 8.0, -1.5, 0.0], dtype=np.float64)
    input_dict = {"name": "exp_input_2", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float16)
    input_dict = {"name": "exp_input_3", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[-3.0, -0.5], [0.5, 1.5]], [[2.5, -2.5], [4.0, -4.0]]], dtype=np.float32)
    input_dict = {"name": "exp_input_4", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[[0.0, 1.0, -1.0]], [[2.0, -2.0, 3.0]]]], dtype=np.float64)
    input_dict = {"name": "exp_input_5", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(1.0 + 1.0j, dtype=np.complex64)
    input_dict = {"name": "exp_input_6", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0 + 2.0j, 0.0 - 0.5j, 3.0 + 0.0j], dtype=np.complex128)
    input_dict = {"name": "exp_input_7", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([15.0, -15.0, 5.5, -7.25], dtype=np.float32)
    input_dict = {"name": "exp_input_8", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10.0, -10.0, 20.0, -20.0], dtype=np.float16)
    input_dict = {"name": "exp_input_9", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan, np.inf, -np.inf, 1.0, -1.0], dtype=np.float32)
    input_dict = {"name": "exp_input_10", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan + 1j, 1.0 + np.nan * 1j, np.inf + 0j, -np.inf + 2j], dtype=np.complex64)
    input_dict = {"name": "exp_input_11", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    input_dict = {"name": "exp_input_12", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0 + 0.0j, -1.0 + 1.0j], [0.5 - 0.5j, 2.0 + 2.0j]]], dtype=np.complex128)
    input_dict = {"name": "exp_input_13", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Exp"] = tf_raw_ops_Exp_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Fact_inputs():
    list_of_inputs = []

    name = "fact"
    list_of_inputs.append(copy.deepcopy({"name": name}))

    name = "fact_op"
    list_of_inputs.append(copy.deepcopy({"name": name}))

    name = "FactOp"
    list_of_inputs.append(copy.deepcopy({"name": name}))

    name = "a"
    list_of_inputs.append(copy.deepcopy({"name": name}))

    name = "scope/fact"
    list_of_inputs.append(copy.deepcopy({"name": name}))

    name = "scope1/scope2/fact"
    list_of_inputs.append(copy.deepcopy({"name": name}))

    name = "fact.op"
    list_of_inputs.append(copy.deepcopy({"name": name}))

    name = "scope.with.dots/fact_op"
    list_of_inputs.append(copy.deepcopy({"name": name}))

    name = "fact123"
    list_of_inputs.append(copy.deepcopy({"name": name}))

    name = "fact_op_v2"
    list_of_inputs.append(copy.deepcopy({"name": name}))

    name = "long_fact_name_with_many_parts_and_numbers_001"
    list_of_inputs.append(copy.deepcopy({"name": name}))

    name = "SCOPE/Deep/FactOp"
    list_of_inputs.append(copy.deepcopy({"name": name}))

    name = "fact__double__underscore"
    list_of_inputs.append(copy.deepcopy({"name": name}))

    name = "fact0_scope/sub1/sub2/fn"
    list_of_inputs.append(copy.deepcopy({"name": name}))

    name = "x_y_z/fact.op.name"
    list_of_inputs.append(copy.deepcopy({"name": name}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Fact"] = tf_raw_ops_Fact_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_floor_inputs():
    list_of_inputs = []

    x = np.array(3.7, dtype=np.float32)
    name = "floor_case_1"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-2.3, dtype=np.float64)
    name = "floor_case_2"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, -0.0, 1.999, -1.001, 1e10], dtype=np.float64)
    name = "floor_case_3"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.1, 0.9], [-0.1, -0.9]], dtype=np.float16)
    name = "floor_case_4"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.linspace(-5, 5, num=24, dtype=np.float32).reshape(2, 3, 4) + 0.499
    name = "floor_case_5"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    name = "floor_case_6"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.1, np.pi, -np.pi, 2.999, -2.001, 100.999], dtype=np.float64).reshape(1, 2, 1, 3)
    name = "floor_case_8_4d"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.inf, -np.inf, np.nan, 5.5, -5.5], dtype=np.float32)
    name = "floor_case_9_inf_nan"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = (np.arange(20, dtype=np.float32) / 3.0) - 3.0
    x = base[::2]
    name = "floor_case_10_noncontig"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.asfortranarray(np.array([[1.2, -3.4, 5.6], [7.8, -9.0, 0.0]], dtype=np.float32))
    name = "floor_case_11_fortran_order"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1e308, 1e-308, -1e-308, 1e308], dtype=np.float64)
    name = "floor_case_12_extremes"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Floor"] = tf_raw_ops_floor_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
np.random.seed(42)
tf.random.set_seed(42)

def tf_raw_ops_GuaranteeConst_inputs():
    list_of_inputs = []

    arr = np.array(5, dtype=np.int32)
    input_dict = {"name": "gc_scalar_int32", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([-1.5, 0.0, 2.5], dtype=np.float32)
    input_dict = {"name": "gc_1d_float32_neg_pos", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.arange(6, dtype=np.int64).reshape(2, 3)
    input_dict = {"name": "gc_2d_int64_matrix", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.random.randn(2, 2, 3).astype(np.float64)
    input_dict = {"name": "gc_3d_float64_random", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[True, False], [False, True]], dtype=bool)
    input_dict = {"name": "gc_2d_bool", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([1+2j, -3+0.5j], dtype=np.complex64)
    input_dict = {"name": "gc_1d_complex64", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([0, 255, 128], dtype=np.uint8)
    input_dict = {"name": "gc_1d_uint8", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([b"hello", b"world"], dtype=object)
    input_dict = {"name": "gc_1d_bytes_strings", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([], dtype=np.float32)
    input_dict = {"name": "gc_empty_float32", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = (np.random.randint(-1000, 1000, size=(2, 3, 4, 5))).astype(np.int16)
    input_dict = {"name": "gc_4d_int16_random", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float64)
    input_dict = {"name": "gc_special_float64", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = np.array([[1.0, -2.0], [3.5, 0.0]], dtype=np.float64)
    imag = np.array([[0.5, 1.5], [-4.0, 2.25]], dtype=np.float64)
    arr = real + 1j * imag
    arr = arr.astype(np.complex128)
    input_dict = {"name": "gc_2d_complex128", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array(["こんにちは", "世界"], dtype=object)
    input_dict = {"name": "gc_unicode_strings", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([2147483647, -2147483647], dtype=np.int32)
    input_dict = {"name": "gc_edge_int32_values", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.GuaranteeConst"] = tf_raw_ops_GuaranteeConst_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_identity_inputs():
    list_of_inputs = []

    arr = np.array([-3, 0, 7, -1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "id_int32_1d", "input": arr}))

    arr = np.array([[1.5, -2.3], [np.nan, np.inf]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"name": "id_float32_2d_nan_inf", "input": arr}))

    arr = np.array([[[True, False], [False, True]], [[True, True], [False, False]]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"name": "id_bool_3d", "input": arr}))

    arr = np.array([[1+2j, -3-0j, 0+0j], [4-5j, -6+7j, 8+0j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"name": "id_complex64_2x3", "input": arr}))

    arr = np.array([b"a", b"bb", b"ccc", b""], dtype=object)
    list_of_inputs.append(copy.deepcopy({"name": "id_bytes_string_1d", "input": arr}))

    arr = np.array(-0.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"name": "id_float64_scalar_negzero", "input": arr}))

    arr = np.array([[0, 255, 128], [64, 32, 16]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"name": "id_uint8_2d", "input": arr}))

    arr = np.array([[[[1, -1, 2]], [[3, -3, 4]]]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "id_int64_4d", "input": arr}))

    arr = np.array([[-1.5, 0.0, 2.25], [3.5, -4.75, 5.125]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"name": "id_float16_2d", "input": arr}))

    arr = np.array([], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "id_empty_int32_1d", "input": arr}))

    arr = np.empty((2, 0), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"name": "id_empty_float32_2x0", "input": arr}))

    arr = np.array([np.nan + 1j, 2 - np.inf*1j, -3 + 0j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"name": "id_complex128_1d_nan_inf", "input": arr}))

    arr = np.array([["hello", "世界"], ["🌟", ""]], dtype=object)
    list_of_inputs.append(copy.deepcopy({"name": "id_unicode_string_2d", "input": arr}))

    arr = np.array(True, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"name": "id_bool_scalar", "input": arr}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Identity"] = tf_raw_ops_identity_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_invert_permutation_inputs():
    list_of_inputs = []

    x = np.array([0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "single_element_int32", "x": x}))

    x = np.array([1, 0], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "two_elements_swap_int64", "x": x}))

    x = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "identity_len5_int32", "x": x}))

    x = np.array([4, 3, 2, 1, 0], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "reverse_len5_int64", "x": x}))

    x = np.array([3, 0, 6, 1, 4, 2, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "random_len7_int32", "x": x}))

    x = np.array([7, 2, 9, 0, 5, 1, 8, 6, 4, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "random_len10_int64", "x": x}))

    x = np.array([2, 8, 4, 0, 6, 1, 3, 5, 7], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "mixed_pattern_len9_int32", "x": x}))

    x = np.array([2, 0, 1], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "cycle_len3_int64", "x": x}))

    x = np.array([1, 5, 3, 2, 4, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "custom_len6_int32", "x": x}))

    x = np.array([11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "reverse_len12_int64", "x": x}))

    x = np.array([1, 2, 3, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "rotate_left_len4_int32", "x": x}))

    x = np.array([(i + 3) % 16 for i in range(16)], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "cyclic_shift_len16_int64", "x": x}))

    return list_of_inputs

generated_inputs["tf.raw_ops.InvertPermutation"] = tf_raw_ops_invert_permutation_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_isfinite_inputs():
    list_of_inputs = []

    x = np.array([5.0, 4.8, 6.8, np.inf, np.nan], dtype=np.float32)
    input_dict = {"name": "finite_vector_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.0, -1.0, 2.5], [np.inf, -np.inf, np.nan]], dtype=np.float64)
    input_dict = {"name": "matrix_mixed_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0, -2.0], [3.0, -4.0]], [[np.nan, np.inf], [-np.inf, 0.0]]], dtype=np.float16)
    input_dict = {"name": "tensor3d_f16", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(0.0, dtype=np.float32)
    input_dict = {"name": "scalar_zero_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(np.nan, dtype=np.float64)
    input_dict = {"name": "scalar_nan_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    input_dict = {"name": "empty_1d_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.empty((2, 0), dtype=np.float64)
    input_dict = {"name": "empty_2d_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[[1.0, np.inf, -np.inf]], [[-0.0, 3.14, np.nan]]]], dtype=np.float32)
    input_dict = {"name": "tensor4d_mixed_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e308, -1e308, np.inf, -np.inf, np.nan], dtype=np.float64)
    input_dict = {"name": "large_values_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-45, -1e-45, 1.1754944e-38, -1.1754944e-38], dtype=np.float32)
    input_dict = {"name": "subnormal_normal_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, 65504.0, 1e5, -1e5, np.nan], dtype=np.float16)
    input_dict = {"name": "f16_extremes", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[-np.inf], [np.nan]], [[1.0], [2.0]]], dtype=np.float32)
    input_dict = {"name": "mixed_column_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.IsFinite"] = tf_raw_ops_isfinite_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_isinf_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array(np.inf, dtype=np.float32)
    input_dict = {"name": "isinf_scalar_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([5.0, np.inf, 6.8, -np.inf], dtype=np.float64)
    input_dict = {"name": "isinf_vector_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[0.0, np.inf, -np.inf],
                  [np.nan, 65504.0, 1e5]], dtype=np.float16)
    input_dict = {"name": "isinf_matrix_f16", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([
        [[1.0, -2.0, np.inf], [np.finfo(np.float32).max, -np.inf, 0.0]],
        [[np.nan, 3.14, -1e40], [1e20, 4e38, 5e38]]
    ], dtype=np.float32)
    input_dict = {"name": "isinf_3d_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[[[np.inf, -np.inf, 1e308, 1e309]],
                   [[-1e309, 0.0, 42.0, -3.14]]]], dtype=np.float64)
    input_dict = {"name": "isinf_4d_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([], dtype=np.float32)
    input_dict = {"name": "isinf_empty_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([[np.nan, -np.nan, np.inf],
                  [np.nan, 0.0, -np.inf]], dtype=np.float64)
    input_dict = {"name": "isinf_mixed_nan_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([1e-45, 1e-38, 1e38, 3.5e38, -3.6e38], dtype=np.float32)
    input_dict = {"name": "isinf_large_range_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array(-np.inf, dtype=np.float64)
    input_dict = {"name": "isinf_scalar_neg_inf_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    data = np.array([0.0, 1e5, -1e5, 65504.0, -65504.0, np.inf,
                     -np.inf, np.nan, 1.0, -2.0, 3.0, -4.0], dtype=np.float16)
    x = data.reshape(2, 3, 2, 1)
    input_dict = {"name": "isinf_high_dim_f16", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    x = np.array([0.0, -0.0, np.finfo(np.float64).max,
                  -np.finfo(np.float64).max, np.inf], dtype=np.float64)
    input_dict = {"name": "isinf_row_vector_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    x = np.arange(25, dtype=np.float32).reshape(5, 5)
    x[0, 0] = np.inf
    x[4, 4] = -np.inf
    input_dict = {"name": "isinf_big2d_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.IsInf"] = tf_raw_ops_isinf_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_isnan_inputs():
    list_of_inputs = []

    x = np.array([5.0, np.nan, -3.2, np.inf, -np.inf, 0.0], dtype=np.float32)
    name = "isnan_case_1"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[np.nan, 1.0], [2.5, -np.inf], [np.inf, 0.0]], dtype=np.float64)
    name = "isnan_case_2"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[0.0, np.nan, 3.0], [4.0, 5.0, np.nan]], [[-1.0, -2.0, -3.0], [np.inf, -np.inf, 7.0]]], dtype=np.float16)
    name = "isnan_case_3"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array(np.nan, dtype=np.float32)
    name = "isnan_case_4_scalar_nan"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
    name = "isnan_case_5_4d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([], dtype=np.float64)
    name = "isnan_case_6_empty_1d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    base = np.arange(20, dtype=np.float32)
    base[3] = np.nan
    base[10] = np.nan
    x = base[::2]
    name = "isnan_case_7_noncontig"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([-0.0, 0.0, np.nan, 1e308, -1e308, np.nan], dtype=np.float64)
    name = "isnan_case_8_large_vals"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[np.inf, -np.inf, np.nan], [1.0, 2.0, -3.0]], dtype=np.float16)
    name = "isnan_case_9_f16_mixed"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.zeros((2, 0, 3), dtype=np.float32)
    name = "isnan_case_10_empty_3d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([np.float32(1e-45), np.float32(0.0), np.nan, np.float32(-1e-45)], dtype=np.float32)
    name = "isnan_case_11_subnormals"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[[[np.nan, 1.0, 2.0]], [[3.0, np.nan, 4.0]]]]], dtype=np.float32)
    name = "isnan_case_12_5d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    return list_of_inputs

generated_inputs["tf.raw_ops.IsNan"] = tf_raw_ops_isnan_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_raw_ops_L2Loss_inputs():
    list_of_inputs = []

    t = np.array(3.5, dtype=np.float32)
    input_dict = {"name": "scalar_f32", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([-3.0, -1.5, 0.0, 2.5, 4.0], dtype=np.float32)
    input_dict = {"name": "vector_f32_neg_pos", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([[-1.0, 2.0, -3.0], [4.5, 0.0, -6.5]], dtype=np.float64)
    input_dict = {"name": "matrix_f64", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.random.randn(2, 3, 4).astype(np.float16)
    input_dict = {"name": "tensor3d_f16", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([], dtype=np.float32)
    input_dict = {"name": "empty_1d_f32", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.empty((2, 0), dtype=np.float32)
    input_dict = {"name": "empty_2d_f32", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([1e154, -1e154, 3.14e153], dtype=np.float64)
    input_dict = {"name": "large_vals_f64", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([np.nan, np.inf, -np.inf, 1.0, -2.0], dtype=np.float32)
    input_dict = {"name": "nan_inf_f32", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = (np.ones((1, 2, 1, 3, 2), dtype=np.float32) * -0.75).astype(np.float32)
    input_dict = {"name": "high_dim_f32", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.arange(30, dtype=np.float64).reshape(5, 6)
    t = a[::2, ::2]
    input_dict = {"name": "non_contiguous_slice_f64", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([1e-5, -1e-5, 2e-6, -2e-6], dtype=np.float16)
    input_dict = {"name": "subnormal_like_f16", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.random.uniform(-5.0, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {"name": "random_4d_f32", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.L2Loss"] = tf_raw_ops_L2Loss_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_Lgamma_inputs():
    list_of_inputs = []

    x = np.array(5.0, dtype=np.float32)
    name = "lgamma_scalar_f32"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0.0, 0.5, 1.0, 4.5, -4.0, -5.6], dtype=np.float32)
    name = "lgamma_vector_examples_f32"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-0.5, -1.3, -2.7], dtype=np.float64)
    name = "lgamma_neg_nonints_f64"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.0, -3.0, -10.0], dtype=np.float64)
    name = "lgamma_neg_integers_f64"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.1, 1.2, 2.3], [3.4, 10.5, 20.0]], dtype=np.float16)
    name = "lgamma_matrix_f16"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([
        [[1e-7, 2.0, -0.1], [0.0, 3.14, -3.14]],
        [[1.5, 2.5, 3.5], [-0.25, -0.75, 0.75]]
    ], dtype=np.float32)
    name = "lgamma_3d_mixed_f32"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[[0.25, 0.75, 1.25]]], [[[5.5, -7.2, 12.0]]]], dtype=np.float64)
    name = "lgamma_4d_f64"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    name = "lgamma_empty_f32"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10.0, 50.0, 100.0], dtype=np.float64)
    name = "lgamma_large_values_f64"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    x = (base + 0.5)[:, ::2]
    name = "lgamma_noncontiguous_f32"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan, np.inf, -np.inf, 2.0, -2.5], dtype=np.float32)
    name = "lgamma_special_values_f32"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1e-6, 1e-6, 1e-12, -1e-12], dtype=np.float64)
    name = "lgamma_tiny_values_f64"
    input_dict = {"name": name, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Lgamma"] = tf_raw_ops_Lgamma_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_log_inputs():
    list_of_inputs = []

    x = np.array([0.0, 0.5, 1.0, 5.0], dtype=np.float32)
    input_dict = {"name": "log_case_1", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1e-3, 2.5], [10.0, 100.0]], dtype=np.float64)
    input_dict = {"name": "log_case_2", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[-1.0, 0.0], [2.0, 3.5]], [[-5.0, 7.0], [0.1, -0.2]]], dtype=np.float16)
    input_dict = {"name": "log_case_3", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1+1j, -1+0j, 0+2j, -3-4j], dtype=np.complex64)
    input_dict = {"name": "log_case_4", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1-1j, 2+0.5j], [0.001+3j, -2-2j]], dtype=np.complex128)
    input_dict = {"name": "log_case_5", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e10, 1e20, 1e-10, 3.14159265], dtype=np.float32)
    input_dict = {"name": "log_case_6", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-300, 1e-200, 1e-100, 1e-50], dtype=np.float64)
    input_dict = {"name": "log_case_7", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -0.5, -10.0, -1e-6], dtype=np.float32)
    input_dict = {"name": "log_case_8", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(2.718281828, dtype=np.float64)
    input_dict = {"name": "log_case_9_scalar_float64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.ones((2, 3, 4, 5), dtype=np.float32) * 2.0
    input_dict = {"name": "log_case_10_4d_float32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(1+0j, dtype=np.complex64)
    input_dict = {"name": "log_case_11_scalar_complex64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    input_dict = {"name": "log_case_12_empty_float32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.0, -0.0], [-np.finfo(np.float32).tiny, np.finfo(np.float32).tiny]], dtype=np.float32)
    input_dict = {"name": "log_case_13_edge_float32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.linspace(0.1, 10.0, 15, dtype=np.float64).reshape(3, 5)
    input_dict = {"name": "log_case_14_linspace_float64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Log"] = tf_raw_ops_log_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_LoopCond_inputs():
    list_of_inputs = []

    inp = np.array(True, dtype=np.bool_)
    input_dict = {"name": "loopcond_true_scalar_array", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.array(False, dtype=np.bool_)
    input_dict = {"name": "loopcond_false_scalar_array", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.bool_(True)
    input_dict = {"name": "loopcond_true_numpy_scalar", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.bool_(False)
    input_dict = {"name": "loopcond_false_numpy_scalar", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.array(1, dtype=np.bool_)
    input_dict = {"name": "loopcond_from_int_one", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.array(0, dtype=np.bool_)
    input_dict = {"name": "loopcond_from_int_zero", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.asarray(True, dtype=bool)
    input_dict = {"name": "loopcond_asarray_true", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.asarray(False, dtype=bool)
    input_dict = {"name": "loopcond_asarray_false", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.full((), True, dtype=np.bool_)
    input_dict = {"name": "loopcond_full_true", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.full((), False, dtype=np.bool_)
    input_dict = {"name": "loopcond_full_false", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.array(True, dtype='?')
    input_dict = {"name": "loopcond_dtype_questionmark_true", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = np.array(np.bool_(True))
    input_dict = {"name": "loopcond_wrapped_numpy_bool_true", "input": inp}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.LoopCond"] = tf_raw_ops_LoopCond_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_max_inputs():
    list_of_inputs = []

    inp = np.array([1, -3, 2, 7], dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"keep_dims": False, "name": "max_case_1", "input": inp, "axis": axis}))

    inp = np.array([[1.0, -2.5, 3.1], [4.2, 0.0, -7.3]], dtype=np.float32)
    axis = np.array(-1, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"keep_dims": True, "name": "max_case_2", "input": inp, "axis": axis}))

    inp = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4) - 5.5
    axis = np.array([1], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"keep_dims": False, "name": "max_case_3", "input": inp, "axis": axis}))

    inp = np.array(
        [
            [[[1.0], [2.0], [3.0]], [[-4.0], [5.0], [6.0]]],
            [[[7.5], [8.5], [9.5]], [[-10.0], [11.0], [12.0]]],
        ],
        dtype=np.float16,
    )
    axis = np.array([0, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"keep_dims": True, "name": "max_case_4", "input": inp, "axis": axis}))

    inp = np.array([-128, -1, 0, 127], dtype=np.int8)
    axis = np.array(0, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"keep_dims": False, "name": "max_case_5", "input": inp, "axis": axis}))

    inp = np.array([[100, 2], [400, -5], [7, 800]], dtype=np.int16)
    axis = np.array(-2, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"keep_dims": True, "name": "max_case_6", "input": inp, "axis": axis}))

    inp = np.array(
        [
            [[1, -2], [3, 4]],
            [[-5, 6], [7, -8]],
        ],
        dtype=np.int64,
    )
    axis = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"keep_dims": False, "name": "max_case_7", "input": inp, "axis": axis}))

    inp = np.arange(2 * 1 * 3 * 1 * 4, dtype=np.float32).reshape(2, 1, 3, 1, 4) - 10.0
    axis = np.array([0, 4], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"keep_dims": True, "name": "max_case_8", "input": inp, "axis": axis}))

    inp = np.array([[0.5], [1.5], [-2.0], [3.0]], dtype=np.float64)
    axis = np.array(0, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"keep_dims": False, "name": "max_case_9", "input": inp, "axis": axis}))

    inp = np.arange(2 * 3 * 4, dtype=np.int32).reshape(2, 3, 4) - 3
    axis = np.array(-1, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"keep_dims": True, "name": "max_case_10", "input": inp, "axis": axis}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Max"] = tf_raw_ops_max_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_mean_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.array([1.0, 2.5, -3.0, 0.0, 4.5], dtype=np.float32)
    axis = np.array(0, dtype=np.int32)
    keep_dims = False
    name = "mean_f32_1d_axis0"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 2
    input_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis = np.array(1, dtype=np.int64)
    keep_dims = True
    name = "mean_i32_2d_axis1_keep"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 3
    input_arr = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    axis = np.array([0, 2], dtype=np.int32)
    keep_dims = False
    name = "mean_u8_3d_axes_0_2"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 4
    input_arr = np.arange(-24, 0, dtype=np.int16).reshape(2, 1, 3, 4)
    axis = np.array(-1, dtype=np.int64)
    keep_dims = True
    name = "mean_i16_4d_axis_-1_keep"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 5
    input_arr = np.array([[-10, 20, -30, 40],
                          [50, -60, 70, -80],
                          [90, -100, 110, -120]], dtype=np.int8)
    axis = np.array([0], dtype=np.int32)
    keep_dims = False
    name = "mean_i8_2d_axis0"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 6
    base = np.arange(6, dtype=np.float32).reshape(3, 2)
    input_arr = (base + 1j * base).astype(np.complex64)
    axis = np.array(0, dtype=np.int32)
    keep_dims = True
    name = "mean_c64_2d_axis0_keep"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 7
    input_arr = (np.ones((2, 2, 2, 2, 2), dtype=np.int64) * 7)
    axis = np.array([1, 3], dtype=np.int64)
    keep_dims = False
    name = "mean_i64_5d_axes_1_3"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 8
    input_arr = np.linspace(-1, 1, num=24, dtype=np.float64).reshape(2, 3, 4)
    axis = np.array([-3], dtype=np.int32)
    keep_dims = True
    name = "mean_f64_3d_axis_-3_keep"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 9
    input_arr = np.array([1 + 2j, -3 + 4j, 5 - 6j], dtype=np.complex128)
    axis = np.array(0, dtype=np.int64)
    keep_dims = False
    name = "mean_c128_1d_axis0"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 10
    input_arr = np.array([[1.5, -2.5],
                          [3.0, 4.0],
                          [-1.0, 0.5]], dtype=np.float16)
    axis = np.array([0, 1], dtype=np.int32)
    keep_dims = False
    name = "mean_f16_2d_all_axes"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 11
    input_arr = np.arange(2*3*4, dtype=np.float32).reshape(2, 3, 4) - 10.0
    axis = np.array([-1, -2], dtype=np.int32)
    keep_dims = True
    name = "mean_f32_3d_axes_-1_-2_keep"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    # Input 12
    input_arr = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    axis = np.array([], dtype=np.int32)
    keep_dims = False
    name = "mean_i64_1d_empty_axis"
    list_of_inputs.append(copy.deepcopy({"keep_dims": keep_dims, "name": name, "input": input_arr, "axis": axis}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Mean"] = tf_raw_ops_mean_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_NextIteration_inputs():
    list_of_inputs = []

    data = np.array(5, dtype=np.int32)
    input_dict = {"name": "next_iter_case_1", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([1.0, -2.5, np.nan, np.inf], dtype=np.float32)
    input_dict = {"name": "next_iter_case_2", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[-1, 0, 1], [2, -3, 4]], dtype=np.int64)
    input_dict = {"name": "next_iter_case_3", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[[1.5, -0.5], [2.25, -3.75]]], dtype=np.float16)
    input_dict = {"name": "next_iter_case_4", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[True, False, True], [False, True, False]], dtype=bool)
    input_dict = {"name": "next_iter_case_5", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([1+2j, -3-4j, 0+0j], dtype=np.complex64)
    input_dict = {"name": "next_iter_case_6", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([b"hello", b"world", b"tf"], dtype=np.object_)
    input_dict = {"name": "next_iter_case_7", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[0, 255, 128], [64, 32, 16], [200, 100, 50]], dtype=np.uint8)
    input_dict = {"name": "next_iter_case_8", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([], dtype=np.float32)
    input_dict = {"name": "next_iter_case_9", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.random.RandomState(0).randn(2, 1, 3, 4).astype(np.float32)
    input_dict = {"name": "next_iter_case_10", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.empty((2, 0), dtype=np.int32)
    input_dict = {"name": "next_iter_case_11", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
    input_dict = {"name": "next_iter_case_12", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.NextIteration"] = tf_raw_ops_NextIteration_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_noop_inputs():
    list_of_inputs = []

    name = "noop"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "noop_1"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "NoOpControl"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "_hidden_noop"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "test_noop_alpha"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "noopUpperCASE"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "n1234567890"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "noop__double__underscore"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "control_dep_noop_v2"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "op_x"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "placeholder_op_noop"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    name = "noop_final_case"
    input_dict = {"name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.NoOp"] = tf_raw_ops_noop_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_OnesLike_inputs():
    list_of_inputs = []

    x = np.array([[True, False], [False, True]], dtype=bool)
    name = "bool_2x2"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([-5, 0, 7], dtype=np.int8)
    name = "int8_1d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([0, 127, 255], dtype=np.uint8)
    name = "uint8_1d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[-12345, 0, 12345], [32767, -32768, 42]], dtype=np.int16)
    name = "int16_2x3"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.empty((0, 3), dtype=np.int32)
    name = "int32_empty_0x3"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.empty((2, 0, 4), dtype=np.int64)
    name = "int64_2x0x4"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([-1.5, 0.0, 3.25, 7.75], dtype=np.float16)
    name = "float16_1d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[1.0, -2.0], [3.5, 4.25]], [[-5.75, 6.125], [0.0, -0.5]]], dtype=np.float32)
    name = "float32_2x2x2"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array(-3.141592653589793, dtype=np.float64)
    name = "float64_scalar"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([1+2j, -3-4j, 0+0j], dtype=np.complex64)
    name = "complex64_1d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array(-42, dtype=np.int32)
    name = "int32_scalar_neg"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.empty((0,), dtype=np.float32)
    name = "float32_empty_1d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    return list_of_inputs

generated_inputs["tf.raw_ops.OnesLike"] = tf_raw_ops_OnesLike_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_OrderedMapClear_inputs():
    list_of_inputs = []

    input_dict = {
        "capacity": int(np.int64(0)),
        "memory_limit": int(np.int64(0)),
        "container": "",
        "shared_name": "",
        "name": "cleardefault",
        "dtypes": [np.dtype(np.int32)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int32(10)),
        "memory_limit": int(np.int64(0)),
        "container": "mapa",
        "shared_name": "shareda",
        "name": "clearA",
        "dtypes": [np.dtype(np.float32), np.dtype(np.int64)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(1000)),
        "memory_limit": int(np.int64(1024 * 1024)),
        "container": "containerascii",
        "shared_name": "sharedascii",
        "name": "opascii",
        "dtypes": [np.dtype(np.bool_), np.dtype(np.float64)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(1)),
        "memory_limit": int(np.int64(1)),
        "container": "containeralpha",
        "shared_name": "sharedempty",
        "name": "n4",
        "dtypes": [np.dtype(np.complex64)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(2**10)),
        "memory_limit": int(np.int64(0)),
        "container": "bigcapacity",
        "shared_name": "s5",
        "name": "n5",
        "dtypes": [np.dtype(np.uint8), np.dtype(np.int16)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(5)),
        "memory_limit": int(np.int64(100)),
        "container": "",
        "shared_name": "sharedonly",
        "name": "n6",
        "dtypes": [np.dtype(np.float16)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(0)),
        "memory_limit": int(np.int64(999999)),
        "container": "x" * 64,
        "shared_name": "y",
        "name": "zzz",
        "dtypes": [np.dtype(np.int8), np.dtype(np.float64), np.dtype(np.int64)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int8(7)),
        "memory_limit": int(np.int16(0)),
        "container": "uintcontainer",
        "shared_name": "ushared",
        "name": "ucase",
        "dtypes": [np.dtype(np.uint16), np.dtype(np.uint32)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(3)),
        "memory_limit": int(np.int64(3)),
        "container": "container3",
        "shared_name": "shared3",
        "name": "name3",
        "dtypes": [np.dtype(np.int64)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(50)),
        "memory_limit": int(np.int64(0)),
        "container": "mixedtypes",
        "shared_name": "mix",
        "name": "mixclear",
        "dtypes": [np.dtype(np.float32), np.dtype(np.float64), np.dtype(np.int32), np.dtype(np.int16)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(2)),
        "memory_limit": int(np.int64(2048)),
        "container": "complexc",
        "shared_name": "cx",
        "name": "cxclear",
        "dtypes": [np.dtype(np.complex128)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "capacity": int(np.int64(8)),
        "memory_limit": int(np.int64(0)),
        "container": "boolcontainer",
        "shared_name": "bshared",
        "name": "boolclear",
        "dtypes": [np.dtype(np.bool_)]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.OrderedMapClear"] = tf_raw_ops_OrderedMapClear_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_placeholder_with_default_inputs():
    list_of_inputs = []

    # Input 1
    arr = np.array([1.0, -2.5, 3.3], dtype=np.float32)
    shape = [-1]
    input_dict = {"name": "case1_f32_vec_partial", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    shape = [2, 3]
    input_dict = {"name": "case2_i32_mat", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    arr = np.array(True, dtype=bool)
    shape = []
    input_dict = {"name": "case3_bool_scalar", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    arr = np.empty((2, 0, 4), dtype=np.complex64)
    shape = [2, 0, 4]
    input_dict = {"name": "case4_c64_empty_axis", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    arr = (np.arange(1 * 2 * 3 * 4, dtype=np.float64).reshape(1, 2, 3, 4) * 0.5) - 10.0
    shape = [-1, 2, -1, 4]
    input_dict = {"name": "case5_f64_4d_partial", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    arr = np.array([], dtype=np.int64)
    shape = [0]
    input_dict = {"name": "case6_i64_empty1d", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    arr = np.array([[255, 0], [128, 64]], dtype=np.uint8)
    shape = [2, 2]
    input_dict = {"name": "case7_u8_mat", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    arr = np.array(
        [
            [[-1.0, 2.0, -3.5], [4.2, 0.0, -0.1], [7.7, -8.8, 9.9]],
            [[1.1, -2.2, 3.3], [-4.4, 5.5, -6.6], [7.7, -8.8, 9.9]],
            [[0.0, 0.0, 0.0], [1.5, -1.5, 1.5], [-1.5, 1.5, -1.5]],
        ],
        dtype=np.float16,
    )
    shape = [3, 3, 3]
    input_dict = {"name": "case8_f16_3d", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    arr = np.ones((2, 3, 4, 5, 6), dtype=np.int16) * -7
    shape = [-1, -1, 4, 5, 6]
    input_dict = {"name": "case9_i16_5d_partial", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    arr = np.array(42, dtype=np.int8)
    shape = []
    input_dict = {"name": "case10_i8_scalar", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    arr = np.array(
        [
            [True, False, True],
            [False, False, True],
            [True, True, False],
            [False, True, False],
        ],
        dtype=bool,
    )
    shape = [-1, 3]
    input_dict = {"name": "case11_bool_2d_partial", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    shape = [3, -1]
    input_dict = {"name": "case12_f32_2d_partial", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.PlaceholderWithDefault"] = tf_raw_ops_placeholder_with_default_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_preventgradient_inputs():
    list_of_inputs = []

    x1 = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"message": "no gradient for int32", "name": "pg_int32_vec", "input": x1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x2 = np.array([-1.5, 0.0, 2.5], dtype=np.float32)
    input_dict = {"message": "no gradient for float32", "name": "pg_float32_vec", "input": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x3 = np.array([[True, False], [False, True]], dtype=np.bool_)
    input_dict = {"message": "block grad on bool", "name": "pg_bool_mat", "input": x3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x4 = np.array(42, dtype=np.int64)
    input_dict = {"message": "scalar int64 no grad", "name": "pg_int64_scalar", "input": x4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x5 = np.array([[1+2j, 3-4j]], dtype=np.complex64)
    input_dict = {"message": "complex64 not differentiable here", "name": "pg_complex64_row", "input": x5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x6 = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    input_dict = {"message": "float64 3D no grad", "name": "pg_float64_3d", "input": x6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x7 = np.array([b'alpha', b'beta', b'gamma'], dtype=object)
    input_dict = {"message": "string tensor no grad", "name": "pg_string_vec", "input": x7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x8 = np.array([], dtype=np.int32)
    input_dict = {"message": "empty int32 vector", "name": "pg_empty_int32", "input": x8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x9 = np.ones((5, 0, 3), dtype=np.float32)
    input_dict = {"message": "empty-dim float32", "name": "pg_empty_dim", "input": x9}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x10 = np.array([[[1], [2]], [[3], [4]]], dtype=np.uint8)
    input_dict = {"message": "uint8 small 3D", "name": "pg_uint8_3d", "input": x10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x11 = np.array(-3.14, dtype=np.float16)
    input_dict = {"message": "float16 scalar", "name": "pg_float16_scalar", "input": x11}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x12 = np.linspace(0, 1, 7, dtype=np.float32).reshape(7, 1)
    input_dict = {"message": "linspace float32", "name": "pg_float32_col", "input": x12}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.PreventGradient"] = tf_raw_ops_preventgradient_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_prod_inputs():
    list_of_inputs = []

    input_arr = np.array([1, 2, 3], dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    input_dict = {
        "keep_dims": False,
        "name": "prod_i1",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
    axis = np.array(-1, dtype=np.int32)
    input_dict = {
        "keep_dims": False,
        "name": "prod_i2",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.linspace(-1.5, 2.5, num=24, dtype=np.float64).reshape(2, 3, 4)
    axis = np.array([0, 2], dtype=np.int64)
    input_dict = {
        "keep_dims": True,
        "name": "prod_i3",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([0, 5, 10], dtype=np.uint8)
    axis = np.array(0, dtype=np.int32)
    input_dict = {
        "keep_dims": True,
        "name": "prod_i4",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[-2, 3], [4, -5]], dtype=np.int16)
    axis = np.array([0], dtype=np.int32)
    input_dict = {
        "keep_dims": False,
        "name": "prod_i5",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[1], [2], [3]], [[-1], [0], [4]]], dtype=np.int8)
    axis = np.array([1], dtype=np.int64)
    input_dict = {
        "keep_dims": True,
        "name": "prod_i6",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array(
        [[[1+2j, 3+4j], [5-1j, 2+0j]],
         [[0+1j, -1-1j], [2+2j, -3+0j]]],
        dtype=np.complex64
    )
    axis = np.array(-2, dtype=np.int32)
    input_dict = {
        "keep_dims": False,
        "name": "prod_i7",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([2, 3, 4, 5], dtype=np.int64)
    axis = np.array(0, dtype=np.int64)
    input_dict = {
        "keep_dims": True,
        "name": "prod_i8",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.arange(-12, 12, dtype=np.float16).reshape(2, 3, 4)
    axis = np.array([1], dtype=np.int32)
    input_dict = {
        "keep_dims": False,
        "name": "prod_i9",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    axis = np.array(-1, dtype=np.int64)
    input_dict = {
        "keep_dims": True,
        "name": "prod_i10",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.arange(1, 17, dtype=np.int32).reshape(2, 2, 2, 2)
    axis = np.array([-1, -2, -3, -4], dtype=np.int32)
    input_dict = {
        "keep_dims": False,
        "name": "prod_i11",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[1+0j, 2-1j, 3+3j]], dtype=np.complex64)
    axis = np.array([0], dtype=np.int32)
    input_dict = {
        "keep_dims": False,
        "name": "prod_i12",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[10, 20, 30], [2, 3, 4]], dtype=np.int64)
    axis = np.array(0, dtype=np.int32)
    input_dict = {
        "keep_dims": False,
        "name": "prod_i13",
        "input": input_arr,
        "axis": axis
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Prod"] = tf_raw_ops_prod_inputs()

import numpy as np
import tensorflow as tf
import torch
import copy

def tf_raw_ops_Selu_inputs():
    list_of_inputs = []

    features = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    name = "selu_case_1"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = np.array([[-2.5, 0.0, 2.5], [3.3, -4.4, 5.5]], dtype=np.float16)
    name = "selu_case_2"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = np.linspace(-3, 3, 24, dtype=np.float64).reshape(2, 3, 4)
    name = "selu_case_3"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = np.array(-1.23, dtype=np.float32)
    name = "selu_case_4_scalar"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = (((np.arange(24).reshape(2, 1, 3, 4) - 12) / 5.0)).astype(np.float32)
    name = "selu_case_5_4d"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = np.array([20.0, -20.0, 10.0, -10.0], dtype=np.float32)
    name = "selu_case_6_extremes"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = np.array([-1e-8, 0.0, 1e-8], dtype=np.float64)
    name = "selu_case_7_small_vals"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0], dtype=np.float32)
    name = "selu_case_8_specials"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = np.array([], dtype=np.float32)
    name = "selu_case_9_empty"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    features = base[:, ::2]
    name = "selu_case_10_strided"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    features = (np.arange(12, dtype=np.float16).reshape(1, 2, 1, 2, 3) - 6) / np.float16(3.0)
    name = "selu_case_11_5d_f16"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    mat = (np.arange(16, dtype=np.float64).reshape(4, 4) - 8.0) / 4.0
    features = mat.T
    name = "selu_case_12_transposed_f64"
    list_of_inputs.append(copy.deepcopy({"name": name, "features": features}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Selu"] = tf_raw_ops_Selu_inputs()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(42)

def tf_raw_ops_serialize_tensor_inputs():
    list_of_inputs = []

    tensor = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    input_dict = {"name": "serialize_case_int32_2d", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[[0.1, np.nan, -np.inf], [np.inf, -3.5, 0.0]],
                       [[1.2, -2.3, 4.5], [6.7, -8.9, 10.11]]], dtype=np.float32)
    input_dict = {"name": "serialize_case_float32_3d_nan_inf", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([True, False, True, True, False], dtype=bool)
    input_dict = {"name": "serialize_case_bool_1d", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array(np.pi, dtype=np.float64)
    input_dict = {"name": "serialize_case_float64_scalar", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[1+2j, -3-4j], [5-6j, -7+8j]], dtype=np.complex64)
    input_dict = {"name": "serialize_case_complex64_2d", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([1+0j, 0-1j, -2+3j, 4-5j], dtype=np.complex128)
    input_dict = {"name": "serialize_case_complex128_1d", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.random.randint(0, 256, size=(2, 3, 4, 1), dtype=np.uint8)
    input_dict = {"name": "serialize_case_uint8_4d", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.empty((0, 3), dtype=np.int64)
    input_dict = {"name": "serialize_case_int64_empty_2d", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[-1.5, 2.25, -3.75], [4.5, -5.125, 6.0]], dtype=np.float16)
    input_dict = {"name": "serialize_case_float16_2d", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor = np.array([[[[[1, -1, 2]]], [[[3, -3, 4]]]]], dtype=np.int8)
    input_dict = {"name": "serialize_case_int8_5d", "tensor": tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SerializeTensor"] = tf_raw_ops_serialize_tensor_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sin_inputs():
    list_of_inputs = []

    x = np.array([-np.inf, -9.0, -0.5, 1.0, 1.2, 200.0, 10.0, np.inf], dtype=np.float32)
    name = "sin_vec_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array(0.0, dtype=np.float64)
    name = "sin_scalar_f64"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[-3.0, -1.5, -0.0], [0.5, 1.0, 3.0]], dtype=np.float16)
    name = "sin_mat_f16"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[-np.pi], [-np.pi/2], [0.0]], [[np.pi/2], [np.pi], [3.14]]], dtype=np.float32)
    name = "sin_3d_angles_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([1+2j, -3-4j, 0+1j, -2+0j], dtype=np.complex64)
    name = "sin_complex64_vec"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[1e-3 - 2e-3j, -5.5 + 0.0j], [0.0 + 3.14159j, -2.0 - 1.0j]], dtype=np.complex128)
    name = "sin_complex128_mat"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([np.nan, 1e20, -1e-20, -7.0, 7.0], dtype=np.float64)
    name = "sin_specials_f64"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([-3., -2., -1., 0., 1., 2., 3., 4., 5., 6., 7., 8.], dtype=np.float32).reshape(2,1,3,2)
    name = "sin_4d_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([], dtype=np.float32)
    name = "sin_empty_f32"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.empty((2, 0, 3), dtype=np.float64)
    name = "sin_zerosize_axis_f64"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([1e-4, 1e-5, -1e-5, -2e-4], dtype=np.float16)
    name = "sin_small_f16"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    x = np.array([[[0.0+0.0j, 1.0+0.0j], [0.0+1.0j, -1.0+0.0j]]], dtype=np.complex64)
    name = "sin_complex64_3d"
    list_of_inputs.append(copy.deepcopy({"name": name, "x": x}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Sin"] = tf_raw_ops_sin_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_sinh_inputs():
    list_of_inputs = []

    x = np.array([-np.inf, -9.0, -0.5, 0.0, 1.0, 2.0, 10.0, np.inf], dtype=np.float32)
    input_dict = {"name": "case_vec_f32_basic", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1.5, 0.0, 1.5], [2.5, -3.3, 4.0]], dtype=np.float64)
    input_dict = {"name": "case_mat_f64_mixed", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(1.2, dtype=np.float16)
    input_dict = {"name": "case_scalar_f16", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.linspace(-2, 2, num=24, dtype=np.float32).reshape(2, 3, 4)
    input_dict = {"name": "case_3d_f32_range", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    input_dict = {"name": "case_empty_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1 + 2j, -1 - 1j, 0 + 0j, 3 - 4j], dtype=np.complex64)
    input_dict = {"name": "case_complex64_vec", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0 + np.pi * 1j, -2.5 + 0j], [1.5 - 1.2j, -0.0 + 0j]], dtype=np.complex128)
    input_dict = {"name": "case_complex128_mat", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.arange(24, dtype=np.float64).reshape(1, 2, 3, 4)
    input_dict = {"name": "case_4d_f64_arange", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-100.0, 100.0, 50.0, -50.0], dtype=np.float32)
    input_dict = {"name": "case_large_magnitude_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan, -np.nan, 0.0], dtype=np.float64)
    input_dict = {"name": "case_nan_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.0, -0.0], [1e-3, -1e-3]], dtype=np.float16)
    input_dict = {"name": "case_f16_small_values", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-0.0, dtype=np.float64)
    input_dict = {"name": "case_scalar_neg_zero_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Sinh"] = tf_raw_ops_sinh_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_Softplus_inputs():
    list_of_inputs = []

    features = np.array(-1.5, dtype=np.float32)
    name = "softplus_scalar_f32_neg"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
    name = "softplus_vector_f32_mixed"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-100.0, -5.0, 0.0], [1.0, 5.0, 100.0]], dtype=np.float64)
    name = "softplus_matrix_f64_extremes"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[-2.0, -0.1, 0.0, 0.1]], [[2.0, 10.0, -10.0, 3.0]]], dtype=np.float16)
    name = "softplus_3d_f16"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[[-8.0, -1.0], [0.0, 1.0]], [[2.0, 5.0], [10.0, 15.0]]]], dtype=np.float32)
    name = "softplus_4d_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([], dtype=np.float32)
    name = "softplus_empty_1d_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.empty((0, 3), dtype=np.float64)
    name = "softplus_empty_2d_f64"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([15.0, -15.0, 12.0, -12.0, 0.0], dtype=np.float16)
    name = "softplus_extreme_f16"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([np.nan, np.inf, -np.inf, 0.0, -3.5], dtype=np.float32)
    name = "softplus_specials_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    features = base[::2, ::2]
    name = "softplus_noncontiguous_f64"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array(
        [
            [[[[ -2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]]],
            [[[[10.0, -10.0, 0.5], [-0.5, 3.0, -3.0]]]]
        ],
        dtype=np.float64
    )
    name = "softplus_5d_f64"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Softplus"] = tf_raw_ops_Softplus_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Softsign_inputs():
    rs = np.random.RandomState(0)
    rs2 = np.random.RandomState(123)
    list_of_inputs = []

    features = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    name = "basic_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[1.5, -2.5], [10.0, -0.0]], dtype=np.float64)
    name = "matrix_f64"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array(0.5, dtype=np.float32)
    name = "scalar_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([], dtype=np.float32)
    name = "empty_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.arange(-10, 10, dtype=np.float16)
    name = "range_f16"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = rs.uniform(-5, 5, size=(2, 3, 4)).astype(np.float16)
    name = "rand3d_f16"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.linspace(-100, 100, num=21, dtype=np.float64)[::3]
    name = "strided_view_f64"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([np.inf, -np.inf, np.nan, 1.0, -1.0], dtype=np.float32)
    name = "nan_inf_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))
    name = "fortran_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([1e-12, -1e-12, 1e12, -1e12], dtype=np.float64)
    name = "magnitude_f64"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = rs2.normal(loc=0.0, scale=3.0, size=(2, 2, 2, 3)).astype(np.float32)
    name = "rand4d_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[0.0, -7.5, 7.5, 15.0, -15.0]], dtype=np.float16)
    name = "row2d_f16"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([0.123456789, -0.987654321], dtype=np.float64)
    name = "hi_precision_f64"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.zeros((0, 3), dtype=np.float32)
    name = "empty_2d_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Softsign"] = tf_raw_ops_Softsign_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_SparseSliceGrad_inputs():
    def add_case(name, backprop_vals, backprop_dtype, start_list, output_indices_list, extra_inputs=0):
        start = np.array(start_list, dtype=np.int64)
        output_indices = np.array(output_indices_list, dtype=np.int64)
        rank = output_indices.shape[1]
        keys_o = tuple(output_indices[:, i] for i in reversed(range(rank)))
        order_o = np.lexsort(keys_o)
        output_indices = output_indices[order_o]
        backprop_val_grad = np.array(backprop_vals, dtype=backprop_dtype)[order_o]

        I_sel = output_indices + start
        if extra_inputs > 0:
            base = I_sel.max(axis=0) + 5
            extras = []
            for e in range(extra_inputs):
                extras.append(base + e + np.arange(rank))
            input_indices = np.vstack([I_sel, np.array(extras, dtype=np.int64)])
        else:
            input_indices = I_sel

        input_indices = np.array(input_indices, dtype=np.int64)
        keys_i = tuple(input_indices[:, i] for i in reversed(range(rank)))
        order_i = np.lexsort(keys_i)
        input_indices = input_indices[order_i]

        input_dict = {
            "name": name,
            "backprop_val_grad": backprop_val_grad,
            "input_indices": input_indices,
            "input_start": start,
            "output_indices": output_indices
        }
        return input_dict

    list_of_inputs = []

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_1_float32_rank1",
        backprop_vals=[1.0, -2.5],
        backprop_dtype=np.float32,
        start_list=[2],
        output_indices_list=[[0], [3]],
        extra_inputs=1
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_2_int32_rank2",
        backprop_vals=[5, -1, 0],
        backprop_dtype=np.int32,
        start_list=[1, 2],
        output_indices_list=[[0, 1], [1, 2], [2, 3]],
        extra_inputs=2
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_3_float64_rank3",
        backprop_vals=[-0.5, 2.0, -3.5, 4.75],
        backprop_dtype=np.float64,
        start_list=[0, 0, 1],
        output_indices_list=[[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]],
        extra_inputs=2
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_4_uint8_rank2",
        backprop_vals=[255],
        backprop_dtype=np.uint8,
        start_list=[5, 0],
        output_indices_list=[[2, 2]],
        extra_inputs=3
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_5_int16_rank4",
        backprop_vals=[-1000, 0, 1000, -2000, 2000],
        backprop_dtype=np.int16,
        start_list=[1, 1, 1, 1],
        output_indices_list=[[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
        extra_inputs=1
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_6_int8_rank3",
        backprop_vals=[-128, 127],
        backprop_dtype=np.int8,
        start_list=[3, 0, 2],
        output_indices_list=[[0, 0, 0], [2, 1, 0]],
        extra_inputs=1
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_7_complex64_rank2",
        backprop_vals=[1+2j, -3+0.5j, -1j],
        backprop_dtype=np.complex64,
        start_list=[0, 4],
        output_indices_list=[[1, 0], [2, 1], [3, 2]],
        extra_inputs=0
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_8_int64_rank1",
        backprop_vals=[0, -10, 20, -30],
        backprop_dtype=np.int64,
        start_list=[0],
        output_indices_list=[[0], [1], [2], [4]],
        extra_inputs=2
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_9_float32_rank2_sorted",
        backprop_vals=[0.1, -0.2, 0.3],
        backprop_dtype=np.float32,
        start_list=[2, 3],
        output_indices_list=[[0, 0], [0, 1], [1, 0]],
        extra_inputs=0
    )))

    list_of_inputs.append(copy.deepcopy(add_case(
        name="sparse_slice_grad_case_10_complex128_rank1",
        backprop_vals=[3.5-2.5j],
        backprop_dtype=np.complex128,
        start_list=[10],
        output_indices_list=[[5]],
        extra_inputs=2
    )))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSliceGrad"] = tf_raw_ops_SparseSliceGrad_inputs()

import tensorflow as tf
import numpy as np
import torch
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_sum_inputs():
    list_of_inputs = []

    # Input 1
    inp = np.array([1, 2, 3], dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    keep_dims = False
    name = "sum_int32_1d_axis0"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 2
    inp = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    axis = np.array(1, dtype=np.int64)
    keep_dims = True
    name = "sum_float32_2d_axis1_keep"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 3
    inp = np.array([[[1, -1, 2], [3, -3, 4]]], dtype=np.int64)
    axis = np.array([-1], dtype=np.int32)
    keep_dims = False
    name = "sum_int64_3d_last_axis"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 4
    inp = np.array([[1 + 2j, -3 + 0.5j], [4 - 1j, -2 - 2j]], dtype=np.complex64)
    axis = np.array([0, 1], dtype=np.int64)
    keep_dims = False
    name = "sum_complex64_all_axes"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 5
    inp = np.arange(2 * 1 * 3 * 4, dtype=np.float64).reshape(2, 1, 3, 4)
    axis = np.array(-2, dtype=np.int32)
    keep_dims = True
    name = "sum_float64_4d_negaxis_keep"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 6
    inp = np.array([-5, 0, 5, 10, -10], dtype=np.int16)
    axis = np.array([0], dtype=np.int64)
    keep_dims = False
    name = "sum_int16_vector_axis0"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 7
    inp = np.array([[[1], [2], [3]], [[4], [5], [6]]], dtype=np.float16)
    axis = np.array([0, 1, 2], dtype=np.int32)
    keep_dims = True
    name = "sum_float16_reduce_all_keep"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 8
    inp = np.array([[1 + 0j, 2 - 1j, -3 + 2j], [0 + 0j, -1 - 1j, 4 + 0j]], dtype=np.complex128)
    axis = np.array(-1, dtype=np.int64)
    keep_dims = False
    name = "sum_complex128_last_axis"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 9
    inp = np.empty((2, 0, 3), dtype=np.int8)
    axis = np.array([-2], dtype=np.int64)
    keep_dims = True
    name = "sum_int8_zero_len_axis_keep"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 10
    inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis = np.array([], dtype=np.int32)
    keep_dims = False
    name = "sum_int32_empty_axis_noop"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    # Input 11
    inp = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    axis = np.array([0, 2, 3], dtype=np.int32)
    keep_dims = False
    name = "sum_float32_4d_axes_0_2_3"
    list_of_inputs.append(copy.deepcopy({
        "keep_dims": keep_dims, "name": name, "input": inp, "axis": axis
    }))

    return list_of_inputs

generated_inputs["tf.raw_ops.Sum"] = tf_raw_ops_sum_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_unicode_script_inputs():
    list_of_inputs = []

    # Input 1: 1D small integers
    name = "basic_1d"
    input_arr = np.array([1, 31, 38], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 2: 2D ASCII-like values
    name = "ascii_2d"
    input_arr = np.array([[72, 101, 108, 108, 111],
                          [87, 111, 114, 108, 100]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 3: Negative and zero values
    name = "negatives_and_zero"
    input_arr = np.array([-1, -100, 0, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 4: Empty 1D array
    name = "empty_1d"
    input_arr = np.array([], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 5: Scalar (0-D) tensor
    name = "scalar_65"
    input_arr = np.array(65, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 6: 3D array with mixed valid and invalid code points
    name = "mixed_3d"
    input_arr = np.array([
        [[0x10FFFF, 0x110000, 0x0041],
         [0xAC00,    0x3042,   0x30A2]],
        [[0x4E00,    0x09FF,   0x3400],
         [0xD800,    0xDBFF,   0xDC00]]
    ], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 7: Boundary values and common scripts
    name = "boundaries_and_common"
    input_arr = np.array([0, 0x10FFFF, 0x007A, 0x0416], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 8: Devanagari-related code points
    name = "devanagari_2x2"
    input_arr = np.array([[0x0905, 0x0939],
                          [0x0966, 0x096F]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 9: Surrogate range and adjacent
    name = "surrogates_and_pua"
    input_arr = np.array([0xD800, 0xDFFF, 0xE000], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 10: 4D array with CJK characters
    name = "cjk_4d"
    input_arr = np.array([[[[28450, 23383, 20013]],
                           [[22283, 20108, 24180]]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 11: Arange reshaped to 3x3
    name = "arange_3x3"
    input_arr = np.arange(9, dtype=np.int32).reshape(3, 3)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 12: Extreme int32 values and boundary checks
    name = "extreme_int32_and_boundary"
    input_arr = np.array([-2147483648, 2147483647, 1114111, 1114112], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    return list_of_inputs

generated_inputs["tf.raw_ops.UnicodeScript"] = tf_raw_ops_unicode_script_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_where_inputs():
    list_of_inputs = []

    # Input 1: bool 2D
    condition = tf.constant(np.array([[True, False], [True, False]], dtype=np.bool_))
    input_dict = {"name": "where_bool_2x2", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 3D
    condition = tf.constant(
        np.array(
            [
                [[1.5, 0.0], [-0.5, 0.0]],
                [[0.0, 0.25], [0.0, 0.75]],
                [[0.0, 0.0], [0.0, 0.01]],
            ],
            dtype=np.float32,
        )
    )
    input_dict = {"name": "where_float32_3d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 2D
    condition = tf.constant(
        np.array([[-1.0, 0.0, 3.14], [0.0, -2.71, 0.0]], dtype=np.float64)
    )
    input_dict = {"name": "where_float64_2d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int32 empty 1D
    condition = tf.constant(np.array([], dtype=np.int32))
    input_dict = {"name": "where_int32_empty", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int64 4D
    condition = tf.constant(
        np.array(
            [
                [[[0, 1], [0, 0]]],
                [[[2, 0], [0, 3]]],
            ],
            dtype=np.int64,
        )
    )
    input_dict = {"name": "where_int64_4d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint8 1D
    condition = tf.constant(np.array([0, 255, 1, 0, 128], dtype=np.uint8))
    input_dict = {"name": "where_uint8_1d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int8 2D
    condition = tf.constant(
        np.array([[-1, 0, 1], [0, -128, 127]], dtype=np.int8)
    )
    input_dict = {"name": "where_int8_2d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex64 2D
    condition = tf.constant(
        np.array([[1.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.5j, 0.0 + 0.0j]], dtype=np.complex64)
    )
    input_dict = {"name": "where_complex64_2d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex128 3D
    condition = tf.constant(
        np.array(
            [
                [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 1.0j]],
                [[0.0 + 0.0j, 2.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
            ],
            dtype=np.complex128,
        )
    )
    input_dict = {"name": "where_complex128_3d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int16 3D
    condition = tf.constant(
        np.array(
            [
                [[0, -5, 0], [10, 0, 0]],
                [[0, 0, 0], [0, 3, -2]],
            ],
            dtype=np.int16,
        )
    )
    input_dict = {"name": "where_int16_3d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: bool 3D all false
    condition = tf.constant(np.zeros((2, 3, 1), dtype=np.bool_))
    input_dict = {"name": "where_bool_3d_all_false", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: int32 5D small
    condition = tf.constant(
        np.array(
            [[[[[0, 1]]], [[[2, 0]]]]],
            dtype=np.int32,
        )
    )
    input_dict = {"name": "where_int32_5d", "condition": condition}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Where"] = tf_raw_ops_where_inputs()

