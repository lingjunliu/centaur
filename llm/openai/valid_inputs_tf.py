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

import tensorflow as tf
import numpy as np
import torch
import copy

def tf_IndexedSlices_inputs():
    list_of_inputs = []

    # Input 1
    values = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=np.float32)
    indices = np.array([1, 3], dtype=np.int32)
    dense_shape = np.array([5, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 2
    values = np.arange(3 * 2 * 2).reshape(3, 2, 2).astype(np.int32)
    indices = np.array([0, 2, 4], dtype=np.int32)
    dense_shape = np.array([6, 2, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 3
    values = np.array([0.1, -2.3, 4.5, 6.7], dtype=np.float64)
    indices = np.array([9, 1, 7, 3], dtype=np.int64)
    dense_shape = np.array([10], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 4
    values = np.array([[[True], [False], [True], [True]]], dtype=np.bool_)
    indices = np.array([2], dtype=np.int32)
    dense_shape = np.array([3, 4, 1], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 5
    values = np.array([[1 + 2j, 3 + 4j],
                       [5 + 6j, 7 + 8j]], dtype=np.complex64)
    indices = np.array([5, 1], dtype=np.int64)
    dense_shape = np.array([10, 2], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 6 (empty slices)
    values = np.empty((0, 5), dtype=np.float32)
    indices = np.array([], dtype=np.int32)
    dense_shape = np.array([4, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 7 (duplicate indices)
    values = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]], dtype=np.int64)
    indices = np.array([2, 2, 4], dtype=np.int64)
    dense_shape = np.array([6, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 8 (float16, 3D values)
    values = np.arange(2 * 4 * 3).reshape(2, 4, 3).astype(np.float16)
    indices = np.array([10, 999], dtype=np.int32)
    dense_shape = np.array([1000, 4, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 9 (negative indices)
    values = np.array([[1, -1],
                       [127, -128]], dtype=np.int8)
    indices = np.array([-1, -3], dtype=np.int32)
    dense_shape = np.array([5, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 10 (uint8, 1D values)
    values = np.array([0, 255, 128], dtype=np.uint8)
    indices = np.array([0, 1, 2], dtype=np.int64)
    dense_shape = np.array([3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 11 (string dtype)
    values = np.array([["a", "b"], ["c", ""]], dtype=object)
    indices = np.array([1, 4], dtype=np.int32)
    dense_shape = np.array([5, 2], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 12 (NaN/Inf)
    values = np.array([[[np.nan, np.inf], [-np.inf, 0.0]]], dtype=np.float32)
    indices = np.array([3], dtype=np.int32)
    dense_shape = np.array([8, 2, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    # Input 13 (zero-sized inner dim)
    values = np.empty((4, 0, 3), dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    dense_shape = np.array([10, 0, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    return list_of_inputs

generated_inputs["tf.IndexedSlices_1"] = tf_IndexedSlices_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_IndexedSlices_2_inputs():
    list_of_inputs = []

    values = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=np.float32)
    indices = np.array([1, 3], dtype=np.int32)
    dense_shape = [10, 3]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.arange(12, dtype=np.float64).reshape(3, 2, 2)
    indices = np.array([0, 2, 6], dtype=np.int32)
    dense_shape = [7, 2, 2]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.array([[10, -1, 5, 3]], dtype=np.int32)
    indices = np.array([3], dtype=np.int32)
    dense_shape = [4, 4]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.array([True, False, True], dtype=bool)
    indices = np.array([0, 2, 4], dtype=np.int64)
    dense_shape = [5]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.empty((0, 5), dtype=np.float32)
    indices = np.empty((0,), dtype=np.int32)
    dense_shape = [8, 5]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.empty((2, 0, 3), dtype=np.float32)
    indices = np.array([2, 5], dtype=np.int32)
    dense_shape = [6, 0, 3]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.empty((2, 2, 0, 1), dtype=np.float16)
    indices = np.array([0, 1], dtype=np.int64)
    dense_shape = [9, 2, 0, 1]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.array([[-3, 7],
                       [8, -2],
                       [1, 1]], dtype=np.int16)
    indices = np.array([1, 1, 2], dtype=np.int32)
    dense_shape = [4, 2]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.array([[[ -1.0]],
                       [[ -2.5]],
                       [[  0.0]],
                       [[  3.14]],
                       [[ -7.2]]], dtype=np.float32)
    indices = np.array([9, 2, 7, 3, 5], dtype=np.int32)
    dense_shape = [10, 1, 1]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = (np.arange(18).reshape(2, 3, 3).astype(np.float32) + 1j * np.arange(18).reshape(2, 3, 3).astype(np.float32)).astype(np.complex64)
    indices = np.array([4, 1], dtype=np.int32)
    dense_shape = [50, 3, 3]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.array([[[1, 2],
                        [3, 4]],
                       [[5, 6],
                        [7, 8]],
                       [[9, 10],
                        [11, 12]],
                       [[13, 14],
                        [15, 16]]], dtype=np.uint8)
    indices = np.array([0, 1, 3, 4], dtype=np.int32)
    dense_shape = [5, 2, 2]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    values = np.empty((0,), dtype=np.float64)
    indices = np.empty((0,), dtype=np.int32)
    dense_shape = [10]
    list_of_inputs.append(copy.deepcopy({"values": values, "indices": indices, "dense_shape": dense_shape}))

    return list_of_inputs

generated_inputs["tf.IndexedSlices_2"] = tf_IndexedSlices_2_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_IndexedSlices_inputs():
    list_of_inputs = []

    # Input 1
    values = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=np.float32)
    indices = np.array([1, 3], dtype=np.int32)
    dense_shape = (5, 3)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    values = np.array([[[1.0, -2.0],
                        [3.5, 4.5]],
                       [[-1.2, 0.0],
                        [2.2, -3.4]],
                       [[5.5, 6.6],
                        [-7.7, 8.8]]], dtype=np.float64)
    indices = np.array([0, 2, 4], dtype=np.int64)
    dense_shape = (6, 2, 2)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    values = np.array([10, -20, 30, -40], dtype=np.int32)
    indices = np.array([2, 0, 5, 3], dtype=np.int32)
    dense_shape = (6,)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    values = np.array([[[[1, 2],
                         [3, 4]],
                        [[5, 6],
                         [7, 8]]]], dtype=np.int64)
    indices = np.array([7], dtype=np.int32)
    dense_shape = (8, 2, 2, 2)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    values = np.array([[1+2j, -3+4j, 5-6j, 7+0j, -1-1j],
                       [0+0j, 2+2j, -2-3j, 4+5j, -5+4j],
                       [9-1j, -8+2j, 7-3j, -6+6j, 5-5j]], dtype=np.complex64)
    indices = np.array([1, 1, 4], dtype=np.int32)
    dense_shape = (6, 5)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    values = np.array([[[True], [False], [True]],
                       [[False], [True], [False]]], dtype=bool)
    indices = np.array([0, 3], dtype=np.int64)
    dense_shape = (4, 3, 1)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (empty slices)
    values = np.empty((0, 4), dtype=np.float16)
    indices = np.empty((0,), dtype=np.int32)
    dense_shape = (10, 4)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (uint8 with duplicates)
    values = np.array([[255],
                       [0],
                       [128]], dtype=np.uint8)
    indices = np.array([9, 0, 9], dtype=np.int32)
    dense_shape = (10, 1)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    values = np.array([[[ -1.0,  2.0, -3.0],
                        [  4.5, -5.5,  6.5]]], dtype=np.float32)
    indices = np.array([5], dtype=np.int64)
    dense_shape = (7, 2, 3)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    values = np.array([1+2j, -3+4j], dtype=np.complex128)
    indices = np.array([2, 4], dtype=np.int32)
    dense_shape = (5,)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (zero-sized inner dimension)
    values = np.empty((2, 0), dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    dense_shape = (3, 0)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (5D, int8)
    values = np.array([[[[[ -1 ]]]],
                       [[[[  2 ]]]]], dtype=np.int8)
    indices = np.array([3, 0], dtype=np.int64)
    dense_shape = (4, 1, 1, 1, 1)
    input_dict = {"values": values, "indices": indices, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.IndexedSlices_3"] = tf_IndexedSlices_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
np.random.seed(42)
tf.random.set_seed(42)

def tf_autodiff_ForwardAccumulator_inputs():
    list_of_inputs = []

    # Input 1: scalar float32
    primals = np.array(3.14, dtype=np.float32)
    tangents = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 2: 1D vector float32 with negatives
    primals = np.array([-2.5, 0.0, 3.5, 7.2], dtype=np.float32)
    tangents = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 3: 2D matrix float64
    primals = np.array([[1.0, -1.0, 2.0],
                        [3.5, 0.0, -4.2],
                        [5.1, 6.3, -7.7]], dtype=np.float64)
    tangents = np.array([[0.5, 0.5, -0.5],
                         [1.0, -1.0, 1.5],
                         [0.0, 2.0, -2.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 4: 3D tensor float32
    primals = np.random.uniform(-1.0, 1.0, size=(2, 3, 1)).astype(np.float32)
    tangents = np.random.uniform(-0.5, 0.5, size=(2, 3, 1)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 5: 4D tensor float16
    primals = (np.random.randn(2, 1, 3, 4)).astype(np.float16)
    tangents = (np.random.randn(2, 1, 3, 4)).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 6: large 1D vector float32
    primals = np.linspace(-10, 10, 100).astype(np.float32)
    tangents = np.ones_like(primals, dtype=np.float32) * 0.01
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 7: complex64 vector
    primals = (np.array([1+2j, -3+0.5j, 0-1j], dtype=np.complex64))
    tangents = (np.array([0.1-0.2j, -0.3+0.4j, 0.5+0.6j], dtype=np.complex64))
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 8: 2D zeros tensor float32, non-zero tangents
    primals = np.zeros((5, 5), dtype=np.float32)
    tangents = np.full((5, 5), 2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 9: arange reshaped float64
    primals = np.arange(10, dtype=np.float64).reshape(2, 5)
    tangents = np.ones((2, 5), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 10: 3D tensor float32 with negatives
    primals = np.array([[[-1.0, -2.0], [3.0, -4.0]],
                        [[5.0, -6.0], [7.0, -8.0]]], dtype=np.float32)
    tangents = np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 11: very small numbers float64
    primals = np.array([1e-12, -2e-12, 3e-12, -4e-12], dtype=np.float64)
    tangents = np.array([1e-6, -1e-6, 2e-6, -2e-6], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 12: 2D non-contiguous view float32
    base = np.random.randn(6, 6).astype(np.float32)
    primals = base[::2, ::2]
    tangents = np.random.randn(*primals.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    return list_of_inputs

generated_inputs["tf.autodiff.ForwardAccumulator"] = tf_autodiff_ForwardAccumulator_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_bitwise_bitwise_xor_inputs():
    list_of_inputs = []

    x = np.array([0, 5, -3, 14], dtype=np.int32)
    y = np.array([5, 0, 7, 11], dtype=np.int32)
    name = "xor_int32_vec"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, 2, -3], [4, 5, -6]], dtype=np.int64)
    y = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int64)
    name = "xor_int64_mat"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array(5, dtype=np.int32)
    name = "xor_int32_broadcast_scalar"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.int64)  # shape (2,1,3)
    y = np.array([[[1], [2]]], dtype=np.int64)                # shape (1,2,1)
    name = "xor_int64_3d_broadcast"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-1, dtype=np.int32)
    y = np.array([[0, 1], [2, 3]], dtype=np.int32)
    name = "xor_int32_scalar_mat"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1, -(2**30), 2**30 - 1], dtype=np.int32)
    y = np.array([2**29, -(2**29), 0], dtype=np.int32)
    name = "xor_int32_large_values"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.zeros((3, 4), dtype=np.int64)
    y = np.ones((3, 4), dtype=np.int64)
    name = "xor_int64_zeros_ones"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1], [2], [3]], dtype=np.int32)   # shape (3,1)
    y = np.array([[4, 5, 6, 7]], dtype=np.int32)    # shape (1,4)
    name = "xor_int32_broadcast_2d"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[[1, 2]]], [[[3, 4]]]], dtype=np.int64)  # shape (2,1,1,2)
    y = np.array([[[[5, 6]]]], dtype=np.int64)              # shape (1,1,1,2)
    name = "xor_int64_4d_broadcast"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-1, -2, -3], [4, 5, 6]], dtype=np.int32)
    y = np.array([[6, 5, 4], [-3, -2, -1]], dtype=np.int32)
    name = "xor_int32_neg_pos"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(7, dtype=np.int64)
    y = np.array(13, dtype=np.int64)
    name = "xor_int64_scalar_scalar"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0, -1, 2**40 - 1]], dtype=np.int64)
    y = np.array([[2**35, 2**35 - 1, 0]], dtype=np.int64)
    name = "xor_int64_large_mixed"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.bitwise.bitwise_xor"] = tf_bitwise_bitwise_xor_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_broadcast_dynamic_shape_inputs():
    list_of_inputs = []

    shape_x = np.array([1, 2, 3], dtype=np.int32)
    shape_y = np.array([5, 1, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([], dtype=np.int32)
    shape_y = np.array([7, 8], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([4, 1], dtype=np.int32)
    shape_y = np.array([2, 4, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([0, 3], dtype=np.int32)
    shape_y = np.array([0, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([1, 2, 1, 4, 1], dtype=np.int32)
    shape_y = np.array([3, 1, 5, 1, 7], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([6, 1, 8], dtype=np.int32)
    shape_y = np.array([1, 8], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([], dtype=np.int32)
    shape_y = np.array([], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([1, 1, 1], dtype=np.int32)
    shape_y = np.array([9, 8, 7], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([1], dtype=np.int32)
    shape_y = np.array([2, 3, 4, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([1024, 1, 64], dtype=np.int32)
    shape_y = np.array([1, 32, 64], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([2, 0, 3], dtype=np.int32)
    shape_y = np.array([1, 0, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([3], dtype=np.int32)
    shape_y = np.array([1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([1, 5, 1, 1], dtype=np.int32)
    shape_y = np.array([7, 1, 9, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([2, 3, 4], dtype=np.int32)
    shape_y = np.array([1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([1, 0, 1], dtype=np.int32)
    shape_y = np.array([5, 0, 7], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = np.array([8, 1, 1], dtype=np.int32)
    shape_y = np.array([1, 1, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    return list_of_inputs

generated_inputs["tf.broadcast_dynamic_shape_1"] = tf_broadcast_dynamic_shape_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_broadcast_dynamic_shape_inputs():
    list_of_inputs = []

    shape_x = (np.int32(1), np.int32(2), np.int32(3))
    shape_y = (np.int32(5), np.int32(1), np.int32(3))
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = ()
    shape_y = (np.int64(7),)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = (np.int64(4), np.int64(1))
    shape_y = (np.int64(1), np.int64(5))
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = (np.int32(2), np.int32(3), np.int32(1))
    shape_y = (np.int32(1), np.int32(3), np.int32(4))
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = (np.int64(1),)
    shape_y = (np.int64(1), np.int64(1), np.int64(7))
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = (np.int64(0), np.int64(3))
    shape_y = (np.int64(0), np.int64(1))
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = ()
    shape_y = ()
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = (np.int32(6),)
    shape_y = ()
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = (np.int64(1), np.int64(1), np.int64(1))
    shape_y = (np.int64(9),)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = (np.int32(2), np.int32(3), np.int32(4))
    shape_y = (np.int32(1),)
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = (np.int64(1024), np.int64(1))
    shape_y = (np.int64(1), np.int64(2048))
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = (np.int32(8), np.int32(1), np.int32(5))
    shape_y = (np.int32(1), np.int32(7), np.int32(1))
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    return list_of_inputs

generated_inputs["tf.broadcast_dynamic_shape_2"] = tf_broadcast_dynamic_shape_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_broadcast_dynamic_shape_inputs():
    list_of_inputs = []

    shape_x = [np.int32(1), np.int32(2), np.int32(3)]
    shape_y = [np.int32(5), np.int32(1), np.int32(3)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int64(3), np.int64(1)]
    shape_y = [np.int64(2), np.int64(3), np.int64(4)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int32(1)]
    shape_y = [np.int32(7)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = []
    shape_y = [np.int64(4), np.int64(5)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int64(4), np.int64(1), np.int64(1)]
    shape_y = [np.int64(1), np.int64(5), np.int64(6)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int32(2), np.int32(0), np.int32(3)]
    shape_y = [np.int32(1), np.int32(0), np.int32(1)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int64(6), np.int64(7), np.int64(1)]
    shape_y = [np.int64(1), np.int64(7), np.int64(8)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [1, 1, 1, 1]
    shape_y = [np.int32(9)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int32(32), np.int32(1), np.int32(224), np.int32(224)]
    shape_y = [np.int32(1), np.int32(3), np.int32(1), np.int32(1)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int64(1), np.int64(0)]
    shape_y = [np.int64(5), np.int64(0)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = []
    shape_y = []
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int32(2), np.int32(3), np.int32(4)]
    shape_y = [np.int32(1), np.int32(4)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int64(10), np.int64(11)]
    shape_y = [np.int64(1), np.int64(1)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int32(1), np.int32(2), np.int32(1), np.int32(3)]
    shape_y = [np.int32(3)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int32(7), np.int32(1), np.int32(5)]
    shape_y = [np.int32(1), np.int32(8), np.int32(5)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    shape_x = [np.int64(1), np.int64(1), np.int64(0)]
    shape_y = [np.int64(3), np.int64(4), np.int64(0)]
    list_of_inputs.append(copy.deepcopy({"shape_x": shape_x, "shape_y": shape_y}))

    return list_of_inputs

generated_inputs["tf.broadcast_dynamic_shape_3"] = tf_broadcast_dynamic_shape_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_compat_forward_compatible_inputs():
    list_of_inputs = []

    year = np.int32(2020); month = np.int32(1); day = np.int32(15)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int64(2024); month = np.int8(2); day = np.int8(29)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int16(2023); month = np.int8(12); day = np.int8(31)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int32(1999); month = np.int16(11); day = np.int16(30)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int64(2000); month = np.int16(2); day = np.int16(29)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int16(1904); month = np.int8(2); day = np.int8(29)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int16(1); month = np.int8(1); day = np.int8(1)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int32(9999); month = np.int8(12); day = np.int8(31)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int32(2022); month = np.int8(4); day = np.int8(30)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int16(2021); month = np.int8(2); day = np.int8(28)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int32(2018); month = np.int8(3); day = np.int8(1)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    year = np.int32(2019); month = np.int8(6); day = np.int8(1)
    list_of_inputs.append(copy.deepcopy({"year": year, "month": month, "day": day}))

    return list_of_inputs

generated_inputs["tf.compat.forward_compatible"] = tf_compat_forward_compatible_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_append_inputs():
    list_of_inputs = []

    arr = np.array([1, 2, 3], dtype=np.int32)
    values = np.array([4, -5], dtype=np.int32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.array([[1.5, -2.0, 3.25],
                    [4.0, 5.5, 6.75]], dtype=np.float32)
    values = np.array([[7.0, -8.5, 9.0]], dtype=np.float32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]], dtype=np.float32)
    values = np.array([[7.0, 8.0],
                       [9.0, 10.0]], dtype=np.float32)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.array([[[1, 2],
                     [3, 4]],
                    [[5, 6],
                     [7, 8]]], dtype=np.int64)
    values = np.array([[[9],
                        [10]],
                       [[11],
                        [12]]], dtype=np.int64)
    axis = 2
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]], dtype=np.int16)
    values = np.array([[13, 14],
                       [15, 16],
                       [17, 18]], dtype=np.int16)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.array([[1, -1],
                    [2, -2]], dtype=np.int8)
    values = np.empty((0, 2), dtype=np.int8)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.empty((0, 5), dtype=np.float64)
    values = np.array([[1.0, -1.0, 2.5, -2.5, 0.0],
                       [3.14, 2.71, -0.5, 4.2, -3.3],
                       [9.9, -8.8, 7.7, -6.6, 5.5]], dtype=np.float64)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.array([True, False, True], dtype=np.bool_)
    values = np.array([False, False], dtype=np.bool_)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.array([[1+2j, -3+0.5j]], dtype=np.complex64)
    values = np.array([[4-1j, 5+5j],
                       [0+0j, -2-2j]], dtype=np.complex64)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.ones((2, 3, 4), dtype=np.float32) * -1.0
    values = np.zeros((2, 1, 4), dtype=np.float32) + 5.0
    axis = -2
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.zeros((1, 2, 3, 4), dtype=np.float16)
    values = np.ones((2, 2, 3, 4), dtype=np.float16)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    arr = np.array([[[True],
                     [False]]], dtype=np.bool_)
    values = np.array([[[False],
                        [True]],
                       [[True],
                        [True]]], dtype=np.bool_)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"arr": arr, "values": values, "axis": axis}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.append_1"] = tf_experimental_numpy_append_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_append_inputs():
    list_of_inputs = []

    arr = np.array([1, 2, 3], dtype=np.int32)
    values = np.array([4, 5], dtype=np.int32)
    axis = None
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[1, -2], [3, 4]], dtype=np.int64)
    values = np.array([[5, 6]], dtype=np.int64)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([], dtype=np.float32)
    values = np.array([1.5, -2.5, 0.0], dtype=np.float32)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array(-3, dtype=np.int8)
    values = np.array(10, dtype=np.int8)
    axis = None
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[True, False], [False, True]], dtype=bool)
    values = np.array([[True, False]], dtype=bool)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([np.nan, np.inf, -np.inf, 3.14], dtype=np.float64)
    values = np.array([-1.0, 0.0], dtype=np.float64)
    axis = None
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([1+2j, -3+0j], dtype=np.complex64)
    values = np.array([0-1j, 2+0.5j], dtype=np.complex64)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    values = np.ones((2, 3, 1), dtype=np.int16)
    axis = 2
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.zeros((1, 2, 1, 3), dtype=bool)
    values = np.ones((1, 2, 2, 3), dtype=bool)
    axis = 2
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[1.5, -2.25], [3.75, 4.125]], dtype=np.float16)
    values = np.array([[0.5], [-1.5]], dtype=np.float16)
    axis = 1
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.zeros((0, 3), dtype=np.int32)
    values = np.arange(6, dtype=np.int32).reshape(2, 3)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([0, 255, 128], dtype=np.uint8)
    values = np.array([10, 20], dtype=np.uint8)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.eye(3, dtype=np.complex128)
    values = np.zeros((2, 3), dtype=np.complex128)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([2**30, -2**29], dtype=np.int64)
    values = np.array([2**33], dtype=np.int64)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.arange(6, dtype=np.float32).reshape(3, 2)
    values = np.array([[100.0, 200.0]], dtype=np.float32)
    axis = 0
    input_dict = {"arr": arr, "values": values, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.append_2"] = tf_experimental_numpy_append_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_experimental_numpy_array_inputs():
    list_of_inputs = []

    val = np.array([1, 2, 3], dtype=np.int32)
    dtype = np.int32
    copy_flag = True
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([-5, 0, 7, -9], dtype=np.int64)
    dtype = np.int64
    copy_flag = False
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([[1.5, -2.0], [3.3, 4.4]], dtype=np.float32)
    dtype = np.float32
    copy_flag = False
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([1 + 2j, -3 + 0.5j], dtype=np.complex64)
    dtype = np.complex64
    copy_flag = True
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([[True, False], [False, True]], dtype=np.bool_)
    dtype = np.bool_
    copy_flag = True
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    dtype = np.float64
    copy_flag = False
    ndmin = 3
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([10, 20, 30], dtype=np.uint8)
    dtype = np.uint8
    copy_flag = False
    ndmin = 4
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array(7.25, dtype=np.float64)
    dtype = np.float64
    copy_flag = True
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([], dtype=np.float32)
    dtype = np.float32
    copy_flag = False
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([0, 255], dtype=np.uint8)
    dtype = np.uint16
    copy_flag = True
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.zeros((2, 1, 3, 1), dtype=np.float32)
    dtype = np.float32
    copy_flag = False
    ndmin = 4
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([[np.nan, np.inf], [-np.inf, -1.0]], dtype=np.float64)
    dtype = np.float64
    copy_flag = True
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.arange(100, dtype=np.int64)[::3]
    dtype = np.int64
    copy_flag = False
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array([1.2, -3.4, 5.6], dtype=np.float32)
    dtype = np.int32
    copy_flag = True
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = np.array(True, dtype=np.bool_)
    dtype = np.bool_
    copy_flag = False
    ndmin = 5
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.array_1"] = tf_experimental_numpy_array_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_array_2_inputs():
    list_of_inputs = []

    # Input 1
    val = [1, 2, 3]
    dtype = np.int32
    copy_flag = True
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    val = [[-1, 0, 7]]
    dtype = np.int64
    copy_flag = False
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    val = [1.5, -2.5, 3.0]
    dtype = np.float32
    copy_flag = True
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    val = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    dtype = np.int8
    copy_flag = False
    ndmin = 3
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    val = [True, False, True, False]
    dtype = np.bool_
    copy_flag = True
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    val = [0, 1, 0, 2]
    dtype = np.bool_
    copy_flag = False
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    val = [[1000, -2000, 3000], [4000, -5000, 6000]]
    dtype = np.int32
    copy_flag = True
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    val = [[1.2, 3.4], [5.6, 7.8]]
    dtype = np.dtype('float64')
    copy_flag = False
    ndmin = 4
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    val = [0, 255, 128]
    dtype = np.uint8
    copy_flag = True
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    val = [[0.0]]
    dtype = np.float16
    copy_flag = True
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    val = [[1, 2], [3, 4], [5, 6]]
    dtype = np.int64
    copy_flag = False
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    val = [1.0, 2.0, 3.0, 4.0]
    dtype = np.float64
    copy_flag = True
    ndmin = 3
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.array_2"] = tf_experimental_numpy_array_2_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_array_inputs():
    list_of_inputs = []

    val = (np.int32(1), np.int32(-2), np.int32(3))
    dtype = np.int32
    copy_flag = True
    ndmin = 0
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.array([1.0, -2.5], dtype=np.float64), np.array([3.3, 4.4], dtype=np.float64))
    dtype = np.float64
    copy_flag = False
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.bool_(True), np.bool_(False), np.bool_(True), np.bool_(False))
    dtype = np.bool_
    copy_flag = True
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.complex64(1+2j), np.complex64(-3+0.5j))
    dtype = np.complex64
    copy_flag = False
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = ((np.int16(1), np.int16(2), np.int16(3)), (np.int16(-4), np.int16(-5), np.int16(-6)))
    dtype = np.int16
    copy_flag = True
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.ones((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32), np.full((2, 2), 7, dtype=np.float32))
    dtype = np.float32
    copy_flag = True
    ndmin = 3
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.uint8(255), np.uint8(0), np.uint8(128))
    dtype = np.uint8
    copy_flag = False
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.arange(8, dtype=np.int64).reshape(2, 2, 2), np.arange(8, 16, dtype=np.int64).reshape(2, 2, 2))
    dtype = np.int64
    copy_flag = True
    ndmin = 4
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.array([-1.5, 0.0, 2.5], dtype=np.float16),)
    dtype = np.float16
    copy_flag = False
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.int8(-5), np.float32(3.5), np.int8(10))
    dtype = np.float32
    copy_flag = True
    ndmin = 1
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = ((np.bool_(True), np.bool_(True)), (np.bool_(False), np.bool_(True)))
    dtype = np.bool_
    copy_flag = False
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    val = (np.array([1+0j, 0+2j], dtype=np.complex128), np.array([-3+4j, 5-6j], dtype=np.complex128))
    dtype = np.complex128
    copy_flag = True
    ndmin = 2
    input_dict = {"val": val, "dtype": dtype, "copy": copy_flag, "ndmin": ndmin}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.array_3"] = tf_experimental_numpy_array_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_conj_inputs():
    list_of_inputs = []

    x = np.array([1+2j, -3+4j, -1j, 0+0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([[-1.5, 2.0], [3.0, -4.2]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.arange(-12, 12, dtype=np.int32).reshape(2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array(3+5j, dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    tmp = np.arange(12).reshape(4, 3).astype(np.complex128)
    tmp = tmp + 1j * tmp
    x = tmp[:, ::2]
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array(
        [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
         [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]],
        dtype=np.uint8
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([np.inf + np.nan*1j, -np.inf - np.inf*1j, np.nan + 0j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([[0.0, -0.0], [1.5, -2.5]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.empty((2, 0, 3), dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([-2**40, 2**40 - 1], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([complex(0.0, -0.0), complex(-0.0, 0.0)], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.arange(10, dtype=np.float64)[::-1]
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.conj"] = tf_experimental_numpy_conj_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_diag_inputs():
    list_of_inputs = []

    v = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    k = 0
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], dtype=np.int64)
    k = 1
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([[1.5, -2.0, 3.5, 4.0],
                  [0.0, 7.25, 8.5, -9.0]], dtype=np.float64)
    k = -1
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([], dtype=np.float32)
    k = np.int32(0)
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.zeros((0, 0), dtype=bool)
    k = 0
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([1+2j, -3+0.5j, -1j, 4+4j], dtype=np.complex64)
    k = 2
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    real = np.arange(20, dtype=np.float64).reshape(4, 5)
    imag = np.arange(20, dtype=np.float64).reshape(4, 5)
    v = (real + 1j * imag).astype(np.complex128)
    k = np.int64(3)
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([True, False, True, True, False], dtype=bool)
    k = -2
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([[1, 2, 3, 4, 5]], dtype=np.int16)
    k = 0
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([[10], [20], [30], [40], [50]], dtype=np.int16)
    k = 0
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]], dtype=np.float16)
    k = -3
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([255, 0, 128], dtype=np.uint8)
    k = -1
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    rng = np.random.RandomState(0)
    v = rng.randn(6, 3).astype(np.float32)
    k = 4
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.array([[np.nan, np.inf],
                  [-np.inf, -0.0]], dtype=np.float32)
    k = 0
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    v = np.empty((0, 3), dtype=np.float64)
    k = 0
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.diag"] = tf_experimental_numpy_diag_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_experimental_numpy_expand_dims_inputs():
    list_of_inputs = []

    a = np.array([1, 2, 3], dtype=np.int32)
    axis = 0
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([10, -20, 30, -40], dtype=np.int64)
    axis = 1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1.0, -2.5], [3.2, 4.1]], dtype=np.float64)
    axis = -1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array(5, dtype=np.int16)
    axis = 0
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array(3.14, dtype=np.float32)
    axis = -1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.zeros((0,), dtype=np.float32)
    axis = -2
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[True, False], [False, True]], dtype=bool)
    axis = 2
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.arange(12, dtype=np.int8).reshape(3, 4)
    axis = 1
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    axis = 0
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.linspace(-1, 1, 6, dtype=np.float64).reshape(1, 2, 3)
    axis = -4
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    axis = -2
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.ones((2, 0, 3), dtype=np.float32)
    axis = 3
    input_dict = {"a": a, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.expand_dims_1"] = tf_experimental_numpy_expand_dims_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_expand_dims_inputs():
    list_of_inputs = []

    a = np.array(42, dtype=np.int32)
    axis = (0,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.array(3.14, dtype=np.float64)
    axis = (-1,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.arange(5, dtype=np.float32)
    axis = (0,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.array([255, 0, 128], dtype=np.uint8)
    axis = (-2,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.arange(6, dtype=np.int64).reshape(2, 3)
    axis = (1,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    axis = (-3,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.ones((2, 2, 2), dtype=bool)
    axis = (0,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.zeros((2, 0, 3), dtype=np.int8)
    axis = (-1,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = (np.arange(20, dtype=np.float32).reshape(4, 5).astype(np.complex64))
    axis = (2,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    axis = (2,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.array([], dtype=np.int16)
    axis = (1,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    a = np.zeros((2, 3, 4), dtype=np.float16)
    axis = (-4,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.expand_dims_2"] = tf_experimental_numpy_expand_dims_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_floor_divide_inputs():
    list_of_inputs = []

    x1 = np.array([1, 2, 3, -4, 5], dtype=np.int32)
    x2 = np.array([2, -3, 4, 5, -6], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([[10, -20, 30], [40, -50, 60]], dtype=np.int64)
    x2 = np.array([[3, 4, -5], [-6, 7, 8]], dtype=np.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([[1.5], [-2.5], [3.75]], dtype=np.float32)
    x2 = np.array([[2.0, -3.0, 4.0, 5.0]], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array(123, dtype=np.int32)
    x2 = np.array(7, dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([[100, 200], [250, 255]], dtype=np.uint8)
    x2 = np.array([[3, 5], [7, 9]], dtype=np.uint8)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([-10.0, -20.5, 30.2, 40.8], dtype=np.float64)
    x2 = np.array(-2.5, dtype=np.float64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([[[12, 15, -18], [20, -25, 30]], [[-35, 40, 45], [50, 55, -60]]], dtype=np.int16)
    x2 = np.array([[[3, -4, 5], [6, 7, -8]]], dtype=np.int16)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([7, -8, 9], dtype=np.int32)
    x2 = np.array([2.0, -3.0, 4.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([[10], [20], [-30], [40]], dtype=np.int8)
    x2 = np.array([3, -4, 5, 6], dtype=np.int8)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([[1.0, 2.0, 3.0], [-4.0, 5.5, -6.5], [7.25, -8.75, 9.125]], dtype=np.float16)
    x2 = np.array([[-2.0, 3.0, -4.0], [5.0, -6.0, 7.0], [-8.0, 9.0, -10.0]], dtype=np.float16)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array([-100, -1, 0, 1, 100], dtype=np.int64)
    x2 = np.array([1, 1, 1, 1, 1], dtype=np.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x1 = np.array(10.5, dtype=np.float32)
    x2 = np.array([2.0, -3.0, 4.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.floor_divide"] = tf_experimental_numpy_floor_divide_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_identity_inputs():
    list_of_inputs = []

    n = 0
    dtype = np.float64
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 1
    dtype = np.int32
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 2
    dtype = np.float32
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 3
    dtype = np.bool_
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 4
    dtype = np.complex64
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 5
    dtype = np.complex128
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 6
    dtype = np.uint8
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 7
    dtype = np.int64
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 8
    dtype = np.float16
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 9
    dtype = np.dtype("uint16")
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 10
    dtype = np.dtype("uint32")
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    n = 11
    dtype = np.dtype("int8")
    list_of_inputs.append(copy.deepcopy({"n": n, "dtype": dtype}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.identity"] = tf_experimental_numpy_identity_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_experimental_numpy_isposinf_inputs():
    list_of_inputs = []

    x = np.array(np.inf, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([-np.inf, 0.0, np.inf, np.nan, 1.5, -2.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([[np.inf, -np.inf, 5.0], [3.4e38, np.inf, -np.inf]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.random.randn(2, 3, 4).astype(np.float64)
    x[0, 0, 0] = np.inf
    x[1, 2, 3] = np.inf
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([65504.0, np.inf, -np.inf, 1.0, -0.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([[np.inf]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.zeros((1, 2, 1, 3), dtype=np.float64)
    x[0, 0, 0, 1] = np.inf
    x[0, 1, 0, 2] = np.inf
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([0.0, -0.0, 1e-308, -1e-308, np.inf], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([np.float32(3.402823e38), np.float32(np.inf), np.float32(-np.inf), np.float32(0.0)], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.array([-1.0, -2.0, -np.inf, -1e10], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    x = np.linspace(-10, 10, 11).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.isposinf"] = tf_experimental_numpy_isposinf_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.experimental.numpy.experimental_enable_numpy_behavior()

def tf_experimental_numpy_logaddexp_inputs():
    list_of_inputs = []

    x1 = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    x2 = np.array([1.5, -2.0, 0.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x2 = np.array([[2.0, -4.0, 6.0], [-5.0, 7.0, -9.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.arange(5, dtype=np.float32)
    x2 = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = (np.arange(24, dtype=np.float64).reshape(2, 3, 4) - 12.0)
    x2 = np.ones((1, 3, 1), dtype=np.float64) * -2.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.full((4, 1), -20.0, dtype=np.float16)
    x2 = np.linspace(-30, 30, 4, dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.array([1e-4, -1e-3, 2.5, -3.5], dtype=np.float16)
    x2 = np.array([-2.0, 3.0, -4.0, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.array([np.inf, -np.inf, 0.0, np.nan], dtype=np.float64)
    x2 = np.array([-np.inf, np.inf, -0.0, 1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.array([-1000.0, -10000.0], dtype=np.float64)
    x2 = np.array([-1000.0, -20000.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    base = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    x1 = base[:, ::2]
    x2 = (-base)[:, ::2]
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.array([], dtype=np.float32)
    x2 = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.array([[[0.1, -0.2, 0.3]], [[-0.4, 0.5, -0.6]]], dtype=np.float32)  # (2,1,3)
    x2 = np.array([[[1.0], [-1.0]]], dtype=np.float32)  # (1,2,1)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    x1 = np.random.uniform(-5, 5, size=(2, 3, 1, 4)).astype(np.float32)
    x2 = np.random.uniform(-2, 2, size=(1, 3, 5, 1)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.logaddexp"] = tf_experimental_numpy_logaddexp_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_reciprocal_inputs():
    list_of_inputs = []

    # Input 1: scalar float32
    x = np.array(3.0, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 with negatives and zero
    x = np.array([-1.0, -0.5, 0.0, 2.5], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float16
    x = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float16)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32
    x = np.array([[[1.0, 2.0], [-3.0, 4.0]],
                  [[0.5, -0.25], [1e-3, -1e-6]]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D complex64 including zero
    x = np.array([1+2j, -3+0j, 0+1j, 0+0j], dtype=np.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D complex128
    x = np.array([[1-1j, 2+0j], [0-2j, -0.5+0j]], dtype=np.complex128)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32 with inf and nan
    x = np.array([np.inf, -np.inf, np.nan, 1.0, -1.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: non-contiguous slice (float64)
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    x = base[:, ::2]
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: empty array (0,) float64
    x = np.array([], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: zero-sized middle dimension, float32
    x = np.empty((2, 0, 3), dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: very small magnitudes, float64
    x = np.array([1e-308, -1e-308, 1e-100, -1e-100], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: wide range float32
    x = np.array([1e-2, 1e2, 1e10, -1e-10], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.reciprocal"] = tf_experimental_numpy_reciprocal_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_tril_inputs():
    list_of_inputs = []

    m = np.array([[1, -2], [3, 4]], dtype=np.int32)
    k = np.int32(0)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([[1.5, -2.5, 3.5, -4.5, 5.5],
                  [6.5, -7.5, 8.5, -9.5, 10.5],
                  [11.5, -12.5, 13.5, -14.5, 15.5]], dtype=np.float32)
    k = np.int64(1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([[-1.0, 2.0, -3.0],
                  [4.0, -5.0, 6.0],
                  [-7.0, 8.0, -9.0],
                  [10.0, -11.0, 12.0],
                  [-13.0, 14.0, -15.0]], dtype=np.float64)
    k = np.int16(-1)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([[1+2j, -3+4j, 5-6j],
                  [7-8j, -9+10j, 11+12j],
                  [-13-14j, 15+16j, -17-18j]], dtype=np.complex64)
    k = np.int8(2)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([[[1, -2, 3],
                   [4, -5, 6],
                   [-7, 8, -9]],
                  [[-1, 2, -3],
                   [-4, 5, -6],
                   [7, -8, 9]]], dtype=np.int16)
    k = np.int8(0)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([
        [[[0.1, -0.2, 0.3, -0.4],
          [0.5, -0.6, 0.7, -0.8],
          [0.9, -1.0, 1.1, -1.2],
          [1.3, -1.4, 1.5, -1.6]],
         [[-0.1, 0.2, -0.3, 0.4],
          [-0.5, 0.6, -0.7, 0.8],
          [-0.9, 1.0, -1.1, 1.2],
          [-1.3, 1.4, -1.5, 1.6]]],
        [[[2.1, -2.2, 2.3, -2.4],
          [2.5, -2.6, 2.7, -2.8],
          [2.9, -3.0, 3.1, -3.2],
          [3.3, -3.4, 3.5, -3.6]],
         [[-2.1, 2.2, -2.3, 2.4],
          [-2.5, 2.6, -2.7, 2.8],
          [-2.9, 3.0, -3.1, 3.2],
          [-3.3, 3.4, -3.5, 3.6]]]
    ], dtype=np.float16)
    k = np.int32(-2)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.empty((0, 0), dtype=np.float64)
    k = np.int32(0)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([[42]], dtype=np.int64)
    k = np.int64(5)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([
        [[1, 2, 3, 4, 5],
         [6, 7, 8, 9, 10]],
        [[11, 12, 13, 14, 15],
         [16, 17, 18, 19, 20]],
        [[21, 22, 23, 24, 25],
         [26, 27, 28, 29, 30]]
    ], dtype=np.uint8)
    k = np.int16(-5)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([[0.0, -1.0, 2.0, -3.0, 4.0]], dtype=np.float32)
    k = np.int8(0)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    m = np.array([[1], [-2], [3], [-4], [5]], dtype=np.int8)
    k = np.int32(-10)
    list_of_inputs.append(copy.deepcopy({"m": m, "k": k}))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.tril"] = tf_experimental_numpy_tril_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_zeros_like_inputs():
    list_of_inputs = []

    a = np.array([1, 2, 3], dtype=np.int32)
    dtype = np.int32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1.5, -2.3], [np.nan, np.inf]], dtype=np.float64)
    dtype = np.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array(True, dtype=np.bool_)
    dtype = np.bool_
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.zeros((2, 0), dtype=np.int64)
    dtype = np.int64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[[1 + 2j, -3 + 4j], [5 - 6j, 7 + 0j]]], dtype=np.complex64)
    dtype = np.complex64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    dtype = np.int32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[-1, 0, 1], [-2, 3, -4]], dtype=np.int16)
    dtype = np.int32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.linspace(-1.0, 1.0, num=5, dtype=np.float32)
    dtype = np.float64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([], dtype=np.float32)
    dtype = np.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.broadcast_to(np.array(7, dtype=np.int64), (2, 2, 2)).copy()
    dtype = np.int64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([[1], [2], [3]], dtype=np.int8)
    dtype = np.float32
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1 + 0j, 0 + 1j], dtype=np.complex128)
    dtype = np.complex128
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    dtype = np.int64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.arange(10, dtype=np.float64)[::-2]
    dtype = np.float64
    input_dict = {"a": a, "dtype": dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.zeros_like"] = tf_experimental_numpy_zeros_like_inputs()

import numpy as np
import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_adjust_gamma_inputs():
    list_of_inputs = []

    image = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    gamma = np.float32(0.5)
    gain = np.float32(1.0)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    gamma = np.float32(2.2)
    gain = np.float32(1.0)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(12, dtype=np.int32).reshape(3, 4, 1)
    gamma = np.float32(1.5)
    gain = np.float32(0.8)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.array([[-10, 0], [10, 100]], dtype=np.int32)
    gamma = np.float32(1.0)
    gain = np.float32(-1.5)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.array([[[0.0, 0.2], [0.8, 1.0]]], dtype=np.float64)
    gamma = np.float64(0.2)
    gain = np.float64(0.5)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.array([0.0, 1.0, 0.5, 0.25], dtype=np.float16)
    gamma = np.float16(5.0)
    gain = np.float16(1.0)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(2 * 3 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3, 3)
    gamma = np.float32(2.0)
    gain = np.float32(2.0)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.array([-0.5, 0.0, 0.5, 1.5], dtype=np.float32)
    gamma = np.float32(2.0)
    gain = np.float32(1.0)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.array([[[[0.1], [0.2]], [[0.3], [0.4]]]], dtype=np.float32)
    gamma = np.float32(0.75)
    gain = np.float32(1.0)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.array([[[0.1, 0.5, 0.9, 1.0], [0.0, 0.2, 0.4, 0.6]],
                      [[0.3, 0.7, 0.8, 0.2], [0.9, 0.1, 0.5, 0.3]]], dtype=np.float32)
    gamma = np.float32(1.8)
    gain = np.float32(0.7)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.linspace(1e-6, 1e-2, 5, dtype=np.float32)
    gamma = np.float32(0.1)
    gain = np.float32(1.0)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.array([[100.0, 500.0, 1000.0], [0.1, 1.0, 10.0]], dtype=np.float32)
    gamma = np.float32(0.3)
    gain = np.float32(1.2)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = (np.arange(2 * 2 * 3, dtype=np.int32).reshape(2, 2, 3) - 5)
    gamma = np.float32(0.0)
    gain = np.float32(0.5)
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.adjust_gamma_1"] = tf_image_adjust_gamma_inputs()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(42)

def tf_image_adjust_gamma_inputs():
    list_of_inputs = []

    image = np.random.randint(0, 256, size=(3, 3, 3), dtype=np.uint8)
    gamma = np.float32(0.5)
    gain = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = (np.random.rand(2, 2, 1) * 2.0).astype(np.float32)
    gamma = np.float32(2.2)
    gain = np.array([0.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.randint(0, 1000, size=(5,), dtype=np.int32)
    gamma = np.float32(1.0)
    gain = np.linspace(0.5, 1.5, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(2, 3, 4, 3).astype(np.float32)
    gamma = np.float32(0.0)
    gain = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.randint(0, 101, size=(4, 4), dtype=np.int32)
    gamma = np.float32(0.8)
    gain = np.array([[0.5], [1.0], [1.5], [2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(5, 5, 1).astype(np.float32)
    gamma = np.float32(1.5)
    gain = np.array(-1.2, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.arange(1 * 2 * 3 * 4, dtype=np.uint8).reshape(1, 2, 3, 4)
    gamma = np.float32(3.0)
    gain = np.array([[[[0.5, 1.0, 1.5, 2.0]]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(6, 6, 3).astype(np.float32)
    gamma = np.float32(0.2)
    gain = (np.random.rand(6, 6, 3) * 2.0).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.arange(15, dtype=np.uint8).reshape(3, 5)
    gamma = np.float32(0.9)
    gain = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.array([0.0, 1.0], dtype=np.float32)
    gamma = np.float32(0.7)
    gain = np.array([1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    return list_of_inputs

generated_inputs["tf.image.adjust_gamma_3"] = tf_image_adjust_gamma_inputs()

import tensorflow as tf
import numpy as np
import torch
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_image_adjust_gamma_inputs():
    list_of_inputs = []

    image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    gamma = np.array(2.2, dtype=np.float32)
    gain = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(32, 32, 3).astype(np.float32)
    gamma = np.array(0.5, dtype=np.float32)
    gain = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(48, 48).astype(np.float32)
    gamma = np.array(1.5, dtype=np.float32)
    gain = np.array(0.8, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(2, 32, 32, 1).astype(np.float32)
    gamma = np.array(0.8, dtype=np.float32)
    gain = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(4, 16, 16, 3).astype(np.float32)
    gamma = np.array(1.2, dtype=np.float32)
    gain = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    gamma = np.array(1.0, dtype=np.float32)
    gain = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(1, 10, 10, 4).astype(np.float32)
    gamma = np.array(0.9, dtype=np.float32)
    gain = np.array(1.1, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(256, 256, 3).astype(np.float16)
    gamma = np.array(0.6, dtype=np.float16)
    gain = np.array(1.0, dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.randint(0, 256, (5, 5, 3), dtype=np.uint8)
    gamma = np.array(0.0, dtype=np.float32)
    gain = np.array(128.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.linspace(0.0, 1.0, num=9, dtype=np.float32).reshape(3, 3)
    gamma = np.array(0.9, dtype=np.float32)
    gain = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.random.rand(3, 4, 4, 3).astype(np.float32)
    gamma = np.array(1.1, dtype=np.float32)
    gain = np.array(0.8, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    image = np.linspace(-1.0, 1.0, num=48, dtype=np.float32).reshape(4, 4, 3)
    gamma = np.array(2.0, dtype=np.float32)
    gain = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    return list_of_inputs

generated_inputs["tf.image.adjust_gamma_4"] = tf_image_adjust_gamma_inputs()

import numpy as np
import tensorflow as tf
import copy

np.random.seed(42)

def tf_image_central_crop_inputs():
    list_of_inputs = []

    image = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
    central_fraction = np.float32(0.5)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.random.randn(5, 7, 1).astype(np.float64)
    central_fraction = np.float64(0.8)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.random.rand(2, 8, 6, 3).astype(np.float32)
    central_fraction = np.float32(1.0)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.linspace(-1, 1, num=9 * 9 * 4, dtype=np.float16).reshape(9, 9, 4)
    central_fraction = np.float16(0.33)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.arange(3 * 10 * 10 * 1, dtype=np.float32).reshape(3, 10, 10, 1)
    central_fraction = np.float32(0.25)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.random.uniform(-5, 5, size=(7, 3, 2)).astype(np.float32)
    central_fraction = np.float32(0.95)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.random.randn(1, 12, 5, 3).astype(np.float64)
    central_fraction = np.float64(0.6)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.random.rand(100, 100, 3).astype(np.float32)
    central_fraction = np.float32(0.01)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.random.rand(5, 13, 13, 3).astype(np.float32)
    central_fraction = np.float32(2.0 / 3.0)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.random.rand(2, 50, 1).astype(np.float32)
    central_fraction = np.float32(0.5)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = (np.random.rand(4, 15, 20, 2).astype(np.float16) * 2 - 1).astype(np.float16)
    central_fraction = np.float16(0.75)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    image = np.random.rand(11, 11, 3).astype(np.float64)
    central_fraction = np.float64(0.999999)
    list_of_inputs.append(copy.deepcopy({"image": image, "central_fraction": central_fraction}))

    return list_of_inputs

generated_inputs["tf.image.central_crop"] = tf_image_central_crop_inputs()

import numpy as np
import tensorflow as tf
import copy

np.random.seed(42)

def tf_image_crop_to_bounding_box_inputs():
    list_of_inputs = []

    # Input 1
    image = np.arange(5*7*3, dtype=np.float32).reshape(5, 7, 3)
    offset_height = np.int32(0)
    offset_width = np.int32(0)
    target_height = np.int32(2)
    target_width = np.int32(4)
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    image = (np.random.rand(10, 10, 1) * 255).astype(np.uint8)
    offset_height = 3
    offset_width = 4
    target_height = 5
    target_width = 6
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    image = np.arange(4*5*2, dtype=np.int32).reshape(4, 5, 2)
    offset_height = np.int64(1)
    offset_width = np.int64(1)
    target_height = np.int64(3)
    target_width = np.int64(4)
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    image = (np.random.randn(2, 8, 9, 3)).astype(np.float16)
    offset_height = 2
    offset_width = 3
    target_height = 4
    target_width = 5
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    image = (np.random.randint(0, 256, size=(1, 6, 6, 4))).astype(np.uint8)
    offset_height = 0
    offset_width = 0
    target_height = 6
    target_width = 6
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    image = (np.random.rand(3, 3, 2) > 0.5).astype(np.float32)
    offset_height = np.int32(1)
    offset_width = np.int32(1)
    target_height = np.int32(2)
    target_width = np.int32(2)
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    image = np.random.randint(-1000, 1000, size=(3, 7, 8, 1), dtype=np.int64)
    offset_height = 5
    offset_width = 6
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

    # Input 8
    image = np.linspace(0, 1, 100*50*3, dtype=np.float64).reshape(100, 50, 3)
    offset_height = np.int64(10)
    offset_width = np.int64(5)
    target_height = np.int64(80)
    target_width = np.int64(40)
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    image = np.arange(12*12*1, dtype=np.float32).reshape(12, 12, 1)
    offset_height = 11
    offset_width = 11
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

    # Input 10
    image = np.random.randint(-200, 200, size=(5, 20, 30, 3), dtype=np.int32)
    offset_height = 0
    offset_width = 10
    target_height = 10
    target_width = 20
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    image = np.arange(2*5*4, dtype=np.int32).reshape(2, 5, 4)
    offset_height = np.int32(0)
    offset_width = np.int32(1)
    target_height = np.int32(2)
    target_width = np.int32(3)
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    image = np.random.randn(2, 2, 2, 2).astype(np.float32)
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

    return list_of_inputs

generated_inputs["tf.image.crop_to_bounding_box"] = tf_image_crop_to_bounding_box_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_flip_up_down_inputs():
    list_of_inputs = []

    image = np.array(
        [[[1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0]],
         [[7.0, 8.0, 9.0],
          [10.0, 11.0, 12.0]]],
        dtype=np.float32
    )
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.arange(2 * 3 * 4 * 1, dtype=np.float32).reshape(2, 3, 4, 1)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.array(
        [[[-1], [0], [1]],
         [[2], [-3], [4]],
         [[5], [6], [-7]]],
        dtype=np.int32
    )
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.array(
        [[[[255, 0, 0, 255],
           [0, 255, 0, 255],
           [0, 0, 255, 255]],
          [[10, 20, 30, 40],
           [50, 60, 70, 80],
           [90, 100, 110, 120]]]],
        dtype=np.uint8
    )
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.array(
        [[[True, False],
          [False, True]],
         [[True, True],
          [False, False]]],
        dtype=bool
    )
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.linspace(-1.5, 1.5, num=2 * 2 * 3 * 2, dtype=np.float16).reshape(2, 2, 3, 2)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.array(
        [[[0.1], [0.2]],
         [[0.3], [0.4]],
         [[0.5], [0.6]],
         [[0.7], [0.8]]],
        dtype=np.float64
    )
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.arange(-36, -36 + 3 * 2 * 2 * 3, dtype=np.int16).reshape(3, 2, 2, 3)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.array([[[1, -2, 3, -4, 5],
                       [6, -7, 8, -9, 10]]], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = (np.arange(1 * 5 * 2 * 2, dtype=np.float32).reshape(1, 5, 2, 2) / 10.0)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    image = np.arange(2 * 3 * 1, dtype=np.int64).reshape(2, 3, 1)
    list_of_inputs.append(copy.deepcopy({"image": image}))

    return list_of_inputs

generated_inputs["tf.image.flip_up_down"] = tf_image_flip_up_down_inputs()

import numpy as np
import tensorflow as tf
import copy

try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

def tf_image_random_contrast_inputs():
    list_of_inputs = []

    image = np.linspace(0, 1, 4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    lower = np.float32(0.5)
    upper = np.float32(1.5)
    seed = np.int32(42)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.arange(10 * 10, dtype=np.uint8).reshape(10, 10, 1)
    lower = np.float32(0.0)
    upper = np.float32(2.0)
    seed = np.int64(123)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.linspace(-5, 5, 2 * 8 * 8 * 3, dtype=np.float32).reshape(2, 8, 8, 3)
    lower = np.float32(0.2)
    upper = np.float32(0.8)
    seed = np.int32(0)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.linspace(0, 2, 3 * 2 * 16 * 16 * 3, dtype=np.float32).reshape(3, 2, 16, 16, 3)
    lower = np.float32(0.1)
    upper = np.float32(1.0)
    seed = np.int64(999999)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.linspace(0, 255, 7 * 7 * 4, dtype=np.float32).reshape(7, 7, 4)
    lower = np.float32(1.2)
    upper = np.float32(1.3)
    seed = np.int32(31415)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.arange(5 * 32 * 32 * 3, dtype=np.uint8) % 256).reshape(5, 32, 32, 3)
    lower = np.float32(0.01)
    upper = np.float32(0.99)
    seed = np.int64(7)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.linspace(-1, 1, 2 * 3 * 4 * 10 * 10 * 3, dtype=np.float32).reshape(2, 3, 4, 10, 10, 3)
    lower = np.float32(0.75)
    upper = np.float32(1.25)
    seed = np.int64(2024)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)
    lower = np.float32(0.0001)
    upper = np.float32(0.0002)
    seed = np.int32(555)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.linspace(0, 10, 4 * 6 * 6 * 2, dtype=np.float32).reshape(4, 6, 6, 2)
    lower = np.float32(2.0)
    upper = np.float32(3.0)
    seed = np.int64(88)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.linspace(0, 1, 1 * 10 * 10 * 5, dtype=np.float32).reshape(1, 10, 10, 5)
    lower = np.float32(0.3)
    upper = np.float32(0.3001)
    seed = np.int32(999)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.linspace(0, 100, 12 * 9 * 1, dtype=np.float32).reshape(12, 9, 1)
    lower = np.float32(0.4)
    upper = np.float32(1.8)
    seed = np.int64(321)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.linspace(-10, 10, 3 * 4 * 4 * 3, dtype=np.float32).reshape(3, 4, 4, 3)
    lower = np.float32(0.6)
    upper = np.float32(1.4)
    seed = np.int32(1024)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.random_contrast"] = tf_image_random_contrast_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_random_flip_left_right_inputs():
    list_of_inputs = []

    image = np.array(
        [[[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]],
         [[130, 140, 150], [160, 170, 180], [190, 200, 210], [220, 230, 240]],
         [[250, 0, 10], [20, 30, 40], [50, 60, 70], [80, 90, 100]]],
        dtype=np.uint8
    )
    seed = 1
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array(
        [[[-1], [2]],
         [[-3], [4]]],
        dtype=np.int32
    )
    seed = 42
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array(
        [[[np.nan, 1.0], [np.inf, -np.inf], [0.0, -1.5]],
         [[3.2, -4.1], [5.5, 6.6], [-7.7, 8.8]]],
        dtype=np.float32
    )
    seed = 3
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.arange(2*2*2*3, dtype=np.float64).reshape(2, 2, 2, 3)
    seed = 7
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array(
        [[[[1, 2, 3, 4],
           [5, 6, 7, 8],
           [9, 10, 11, 12],
           [13, 14, 15, 16]],
          [[-1, -2, -3, -4],
           [-5, -6, -7, -8],
           [-9, -10, -11, -12],
           [-13, -14, -15, -16]],
          [[21, 22, 23, 24],
           [25, 26, 27, 28],
           [29, 30, 31, 32],
           [33, 34, 35, 36]]]],
        dtype=np.int16
    )
    seed = 9
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array(
        [[[True], [False], [True], [False], [True]],
         [[False], [True], [False], [True], [False]],
         [[True], [True], [False], [False], [True]],
         [[False], [False], [True], [True], [False]]],
        dtype=bool
    )
    seed = 11
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.random.randint(0, 256, size=(3, 5, 2, 1)).astype(np.uint8)
    seed = 13
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array(
        [[[0.1, -0.2, 0.3],
          [0.4, 0.5, -0.6],
          [0.7, -0.8, 0.9],
          [1.0, -1.1, 1.2],
          [-1.3, 1.4, -1.5]]],
        dtype=np.float16
    )
    seed = 15
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array(
        [[[[1, 2], [3, 4]]],
         [[[5, 6], [7, 8]]],
         [[[9, 10], [11, 12]]],
         [[[13, 14], [15, 16]]]],
        dtype=np.float32
    ).reshape(4, 1, 2, 2)  # Just to ensure shape is clear; not necessary but keeps intent
    image = np.array(
        [[[1, 2]],
         [[3, 4]],
         [[5, 6]],
         [[7, 8]]],
        dtype=np.float32
    ).reshape(4, 1, 2)
    seed = 17
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.asfortranarray(np.arange(18, dtype=np.float32).reshape(2, 3, 3))
    seed = 19
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8]],
          [[-1, -2, -3, -4], [-5, -6, -7, -8]]],
         [[[9, 10, 11, 12], [13, 14, 15, 16]],
          [[-9, -10, -11, -12], [-13, -14, -15, -16]]]],
        dtype=np.int8
    )
    seed = 21
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.arange(6*7, dtype=np.int64).reshape(6, 7, 1)
    seed = 23
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.random_flip_left_right"] = tf_image_random_flip_left_right_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_random_hue_inputs():
    list_of_inputs = []

    # Input 1
    image = np.random.rand(4, 4, 3).astype(np.float32)
    max_delta = np.float32(0.2)
    seed = np.int32(123)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 2
    image = np.random.rand(2, 4, 4, 3).astype(np.float32)
    max_delta = np.float64(0.5)
    seed = np.int64(0)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 3
    image = np.random.rand(1, 1, 3).astype(np.float64)
    max_delta = np.float64(0.0)
    seed = np.int32(7)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 4
    image = np.random.rand(8, 5, 3).astype(np.float32)
    max_delta = np.float32(0.15)
    seed = np.int32(9999)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 5
    image = np.linspace(0.0, 1.0, num=3*2*3, dtype=np.float32).reshape(3, 2, 3)
    max_delta = np.float32(0.05)
    seed = np.int32(2021)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 6
    image = np.random.rand(2, 3, 4, 3).astype(np.float32)
    max_delta = np.float32(0.49)
    seed = np.int32(314159)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 7
    image = np.vstack([
        np.zeros((3, 3, 3), dtype=np.float32),
        np.ones((3, 3, 3), dtype=np.float32)
    ])
    max_delta = np.float32(0.25)
    seed = np.int32(1)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 8
    image = np.random.rand(2, 2, 2, 2, 3).astype(np.float64)
    max_delta = np.float64(0.3)
    seed = np.int64(123456789)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 9
    image = np.random.rand(1, 7, 3).astype(np.float32)
    max_delta = np.float32(0.000001)
    seed = np.int32(4242)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 10
    image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    max_delta = np.float32(0.4)
    seed = np.int32(0)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 11
    image = np.random.randint(0, 256, size=(3, 5, 5, 3), dtype=np.uint8)
    max_delta = np.float32(0.12)
    seed = np.int32(555)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 12
    image = np.tile(np.eye(3, dtype=np.float32)[None, :, :], (4, 1, 1))
    max_delta = np.float32(0.35)
    seed = np.int32(8888)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.random_hue"] = tf_image_random_hue_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_image_random_jpeg_quality_inputs():
    list_of_inputs = []
    rng = np.random.default_rng(12345)

    image = np.array([[[1, 2, 3],
                       [4, 5, 6]],
                      [[7, 8, 9],
                       [10, 11, 12]]], dtype=np.uint8)
    min_jpeg_quality = np.int32(75)
    max_jpeg_quality = np.int32(95)
    seed = np.int32(42)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = (np.arange(16, dtype=np.uint8).reshape(4, 4, 1) * 16) % 256
    min_jpeg_quality = np.int32(0)
    max_jpeg_quality = np.int32(100)
    seed = np.int32(0)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.array([[[128, 64, 32]]], dtype=np.uint8)
    min_jpeg_quality = np.int32(1)
    max_jpeg_quality = np.int32(2)
    seed = np.int32(7)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = rng.integers(0, 256, size=(8, 16, 3), dtype=np.uint8)
    min_jpeg_quality = np.int64(50)
    max_jpeg_quality = np.int64(51)
    seed = np.int64(123)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = rng.integers(0, 256, size=(7, 7, 1), dtype=np.uint8)
    min_jpeg_quality = np.int64(10)
    max_jpeg_quality = np.int64(90)
    seed = np.int64(999)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    min_jpeg_quality = np.int32(30)
    max_jpeg_quality = np.int32(31)
    seed = np.int32(12)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.ones((32, 48, 3), dtype=np.uint8) * 255
    min_jpeg_quality = np.int64(5)
    max_jpeg_quality = np.int64(6)
    seed = np.int64(987654321)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.fromfunction(lambda i, j: (i * 17 + j * 29) % 256, (15, 15), dtype=int).astype(np.uint8)
    image = base[..., None]
    min_jpeg_quality = np.int32(60)
    max_jpeg_quality = np.int32(80)
    seed = np.int32(31415)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    min_jpeg_quality = np.int32(95)
    max_jpeg_quality = np.int32(100)
    seed = np.int32(271828)
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = (np.arange(15, dtype=np.uint8).reshape(3, 5, 1) * 17) % 256
    min_jpeg_quality = 20
    max_jpeg_quality = 21
    seed = 2021
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = ((np.indices((10, 10)).sum(axis=0) % 2) * 255).astype(np.uint8)
    ch1 = base
    ch2 = (base // 2).astype(np.uint8)
    ch3 = (255 - base).astype(np.uint8)
    image = np.stack([ch1, ch2, ch3], axis=-1)
    min_jpeg_quality = 2
    max_jpeg_quality = 10
    seed = 444
    input_dict = {
        "image": image,
        "min_jpeg_quality": min_jpeg_quality,
        "max_jpeg_quality": max_jpeg_quality,
        "seed": seed
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    big = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    image = big[2:15, 3:20, :]
    min_jpeg_quality = 40
    max_jpeg_quality = 70
    seed = 8888
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

def tf_image_resize_with_crop_or_pad_inputs():
    list_of_inputs = []

    image = np.arange(75, dtype=np.float32).reshape(5, 5, 3)
    target_height = 3
    target_width = 3
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(1, 28, dtype=np.int32).reshape(3, 3, 3)
    target_height = 5
    target_width = 5
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.linspace(-1.0, 1.0, 7 * 4 * 1, dtype=np.float32).reshape(7, 4, 1)
    target_height = np.int64(5)
    target_width = np.int64(6)
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.randn(2, 6, 9, 3).astype(np.float32)
    target_height = np.int32(8)
    target_width = np.int32(5)
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(1 * 3 * 5 * 2, dtype=np.int32).reshape(1, 3, 5, 2)
    target_height = 2
    target_width = 2
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.ones((4, 4, 1), dtype=np.float32)
    target_height = 6
    target_width = 7
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.zeros((3, 2, 2, 3), dtype=np.float64)
    target_height = np.int64(3)
    target_width = np.int64(5)
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.uniform(-100, 100, size=(9, 7, 5)).astype(np.float32)
    target_height = 9
    target_width = 3
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(-60, 15, dtype=np.int64).reshape(5, 5, 3)
    target_height = 7
    target_width = 6
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.normal(0, 1, size=(10, 8, 4)).astype(np.float32)
    target_height = 10
    target_width = 8
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.zeros((4, 5, 10, 3), dtype=np.int32)
    target_height = np.int32(5)
    target_width = np.int32(12)
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.randint(-50, 50, size=(2, 11, 3, 1)).astype(np.int32)
    target_height = 5
    target_width = 6
    input_dict = {"image": image, "target_height": target_height, "target_width": target_width}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.resize_with_crop_or_pad"] = tf_image_resize_with_crop_or_pad_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_rgb_to_hsv_inputs():
    list_of_inputs = []

    images = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    name = "case1"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    images = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-0.5, 2.0, 0.5],
        [0.3, -1.2, 1.5]
    ], dtype=np.float64)
    name = "case2"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    rng = np.random.RandomState(0)
    images = rng.rand(5, 5, 3).astype(np.float16)
    name = "case3"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    images = np.linspace(0, 1, 2 * 4 * 4 * 3, dtype=np.float32).reshape(2, 4, 4, 3)
    name = "case4"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    rng = np.random.RandomState(1)
    images = rng.randn(2, 1, 3, 4, 3).astype(np.float32)
    name = "case5"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    images = np.array([[1.0, 0.0, 0.0]], dtype=np.float16)
    name = "case6"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    images = np.tile(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float64), (3, 3, 1))
    name = "case7"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    rng = np.random.RandomState(2)
    images = (rng.rand(10, 3).astype(np.float16) * 2.0) - 0.5
    name = "case8"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    images = np.mod(np.arange(7 * 1 * 3).reshape(7, 1, 3), 2).astype(np.float32)
    name = "case9"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    rng = np.random.RandomState(3)
    images = rng.rand(3, 2, 2, 3).astype(np.float32)
    name = "case10"
    list_of_inputs.append(copy.deepcopy({"images": images, "name": name}))

    return list_of_inputs

generated_inputs["tf.image.rgb_to_hsv"] = tf_image_rgb_to_hsv_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_sobel_edges_inputs():
    list_of_inputs = []
    img = np.zeros((1, 2, 2, 1), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    rng = np.random.default_rng(42)
    img = rng.uniform(0, 255, size=(1, 5, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    H, W = 10, 10
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    data = (y - x).astype(np.float64)
    img = data[None, ..., None]
    list_of_inputs.append(copy.deepcopy({"image": img}))

    img = rng.normal(0.0, 1.0, size=(4, 32, 32, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    img = np.full((1, 100, 50, 4), 0.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    img = np.linspace(-1, 1, num=3 * 2 * 5 * 2, dtype=np.float32).reshape(3, 2, 5, 2)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    img = rng.integers(-100, 100, size=(2, 3, 2, 1)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    img = rng.uniform(-1, 1, size=(5, 7, 7, 8)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    H, W = 128, 128
    y, x = np.mgrid[0:H, 0:W]
    data = np.sin(x / 5.0) + np.cos(y / 7.0)
    img = data[None, ..., None].astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    H, W = 13, 21
    y, x = np.mgrid[0:H, 0:W]
    checker = ((x // 2 + y // 2) % 2).astype(np.float32)
    img = np.stack([checker, 1 - checker, checker * 0.25], axis=-1)[None, ...].astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    img = rng.normal(0, 1000, size=(6, 4, 4, 1)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    img = np.ones((1, 2, 3, 4), dtype=np.float32) * np.arange(4, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"image": img}))

    return list_of_inputs

generated_inputs["tf.image.sobel_edges"] = tf_image_sobel_edges_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_contrast_inputs():
    list_of_inputs = []

    image = np.array([[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]],
                      [[7.0, 8.0, 9.0],
                       [10.0, 11.0, 12.0]]], dtype=np.float64)
    lower = np.float64(0.2)
    upper = np.float64(0.5)
    seed = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.arange(1*3*3*1).reshape(1, 3, 3, 1)).astype(np.float64)
    lower = np.float64(0.5)
    upper = np.float64(1.5)
    seed = np.array([12345, 67890], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.random.RandomState(0).randn(2, 1, 4, 4, 3)).astype(np.float64)
    lower = np.float64(0.1)
    upper = np.float64(0.9)
    seed = np.array([7, 11], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.linspace(-5, 5, 4*4*3).reshape(4, 4, 3)).astype(np.float64)
    lower = np.float64(0.8)
    upper = np.float64(1.2)
    seed = np.array([42, 24], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.random.RandomState(1).rand(1, 2, 2, 4) * 255.0).astype(np.float64)
    lower = np.float64(1.0)
    upper = np.float64(2.0)
    seed = np.array([0, 999], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.random.RandomState(2).rand(5, 5, 1) * 100.0 - 50.0).astype(np.float64)
    lower = np.float64(0.2)
    upper = np.float64(0.3)
    seed = np.array([31415, 92653], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.random.RandomState(3).randn(3, 8, 8, 3)).astype(np.float64)
    lower = np.float64(0.0)
    upper = np.float64(3.0)
    seed = np.array([13579, 24680], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = np.array([[[0.5]]], dtype=np.float64)
    lower = np.float64(0.9)
    upper = np.float64(1.1)
    seed = np.array([101, 202], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.random.RandomState(4).rand(2, 1, 1, 2)).astype(np.float64)
    lower = np.float64(0.01)
    upper = np.float64(0.02)
    seed = np.array([1, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    image = (np.random.RandomState(5).randn(1, 2, 3, 3, 3)).astype(np.float64)
    lower = np.float64(0.3)
    upper = np.float64(0.3001)
    seed = np.array([777, 888], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_contrast_1"] = tf_image_stateless_random_contrast_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_contrast_inputs():
    list_of_inputs = []

    # Input 1
    image = np.array([[[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]],
                      [[7.0, 8.0, 9.0],
                       [10.0, 11.0, 12.0]]], dtype=np.float32)
    lower = np.float32(0.2)
    upper = np.float32(0.5)
    seed = (np.int32(1), np.int32(2))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 2
    image = (np.random.RandomState(0).randn(64, 64, 3).astype(np.float32) * 2.0) - 1.0
    lower = np.float32(0.8)
    upper = np.float32(1.2)
    seed = (np.int32(123), np.int32(456))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 3
    image = np.random.RandomState(1).rand(1, 4, 4, 1).astype(np.float64)
    lower = np.float64(0.1)
    upper = np.float64(2.0)
    seed = (np.int32(0), np.int32(0))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 4
    image = np.random.RandomState(2).randn(2, 3, 3, 3).astype(np.float32)
    lower = np.float32(0.5)
    upper = np.float32(1.5)
    seed = (np.int32(42), np.int32(24))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 5
    image = (np.random.RandomState(3).randint(0, 256, size=(5, 5, 1))).astype(np.uint8)
    lower = np.float32(0.0)
    upper = np.float32(0.9)
    seed = (np.int32(7), np.int32(999999))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 6
    image = np.random.RandomState(4).rand(2, 2, 2, 2, 3).astype(np.float32)
    lower = np.float32(1.0)
    upper = np.float32(1.1)
    seed = (np.int32(31415), np.int32(27182))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 7
    image = np.random.RandomState(5).rand(10, 10, 4).astype(np.float32)
    lower = np.float32(0.3)
    upper = np.float32(0.7)
    seed = (np.int32(2021), np.int32(2022))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 8
    image = np.ones((3, 2, 2, 3), dtype=np.float32)
    lower = np.float32(2.0)
    upper = np.float32(3.0)
    seed = (np.int32(111), np.int32(222))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 9
    image = np.arange(1 * 1 * 3 * 3 * 1, dtype=np.float32).reshape((1, 1, 3, 3, 1))
    lower = np.float32(0.05)
    upper = np.float32(0.95)
    seed = (np.int32(8), np.int32(16))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 10
    image = (np.random.RandomState(6).rand(128, 128, 3).astype(np.float32) * 1000.0)
    lower = np.float32(0.0001)
    upper = np.float32(0.01)
    seed = (np.int32(13579), np.int32(24680))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 11
    image = np.random.RandomState(7).rand(2, 8, 8, 3).astype(np.float32)
    lower = np.float32(10.0)
    upper = np.float32(10.5)
    seed = (np.int32(77), np.int32(88))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    # Input 12
    image = np.arange(1, 1 + 1 * 1 * 3, dtype=np.float64).reshape((1, 1, 3))
    lower = np.float64(0.25)
    upper = np.float64(0.75)
    seed = (np.int32(999), np.int32(1001))
    list_of_inputs.append(copy.deepcopy({"image": image, "lower": lower, "upper": upper, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_contrast_2"] = tf_image_stateless_random_contrast_inputs()

import numpy as np
import tensorflow as tf
import torch
import copy

def tf_image_stateless_random_flip_left_right_inputs():
    list_of_inputs = []

    # Input 1: 3D uint8 RGB image
    image = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[100, 110, 120], [130, 140, 150], [160, 170, 180]],
        ],
        dtype=np.uint8,
    )
    seed = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 2: 4D float32 grayscale batch
    image = np.array(
        [
            [[[0.1], [0.2], [0.3]],
             [[0.4], [0.5], [0.6]]],
            [[[1.0], [1.1], [1.2]],
             [[1.3], [1.4], [1.5]]],
        ],
        dtype=np.float32,
    )
    seed = np.array([123, 456], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 3: 3D float64 grayscale with negatives
    image = np.array(
        [
            [[-1.5], [2.0]],
            [[3.5], [-4.0]],
            [[5.25], [0.0]],
        ],
        dtype=np.float64,
    )
    seed = np.array([7, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 4: 4D int32 with negative values
    image = np.array(
        [
            [
                [[-1, 2], [3, -4]],
                [[5, -6], [-7, 8]],
            ]
        ],
        dtype=np.int32,
    )
    seed = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 5: 3D int32 width=1
    image = np.array(
        [
            [[-128]],
            [[-10]],
            [[10]],
            [[127]],
        ],
        dtype=np.int32,
    )
    seed = np.array([2021, 9], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 6: 3D int32 single row, 2 channels
    image = np.array(
        [
            [[1, -1], [2, -2], [3, -3], [4, -4]],
        ],
        dtype=np.int32,
    )
    seed = np.array([5, 6], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 7: 4D uint8 minimal dims
    image = np.array(
        [
            [[[0]]],
            [[[255]]],
        ],
        dtype=np.uint8,
    )  # shape (2,1,1,1)
    seed = np.array([8, 9], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 8: 4D float32 RGBA
    image = np.array(
        [
            [
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
                [[12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
            ],
            [
                [[0.5, 1.5, 2.5, 3.5], [4.5, 5.5, 6.5, 7.5], [8.5, 9.5, 10.5, 11.5]],
                [[12.5, 13.5, 14.5, 15.5], [16.5, 17.5, 18.5, 19.5], [20.5, 21.5, 22.5, 23.5]],
            ],
            [
                [[-1.0, -2.0, -3.0, -4.0], [-5.0, -6.0, -7.0, -8.0], [-9.0, -10.0, -11.0, -12.0]],
                [[-12.0, -13.0, -14.0, -15.0], [-16.0, -17.0, -18.0, -19.0], [-20.0, -21.0, -22.0, -23.0]],
            ],
        ],
        dtype=np.float32,
    )
    seed = np.array([31415, 27182], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 9: 3D float32 with NaN and Inf
    image = np.array(
        [
            [[np.nan, np.inf], [1.0, -1.0]],
            [[-np.inf, 0.0], [2.5, -3.5]],
        ],
        dtype=np.float32,
    )
    seed = np.array([100, 200], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 10: 4D float32 width=1, 3 channels
    image = np.array(
        [
            [
                [[0.0, 0.5, 1.0]],
                [[1.5, 2.0, 2.5]],
                [[-0.5, -1.0, -1.5]],
            ],
            [
                [[3.0, 3.5, 4.0]],
                [[4.5, 5.0, 5.5]],
                [[6.0, 6.5, 7.0]],
            ],
        ],
        dtype=np.float32,
    )
    seed = np.array([12, 34], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 11: 3D uint8 grayscale 5x5x1
    image = np.arange(25, dtype=np.uint8).reshape(5, 5, 1)
    seed = np.array([777, 888], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    # Input 12: 4D float32 1x4x4x3
    image = (np.arange(4 * 4 * 3, dtype=np.float32).reshape(1, 4, 4, 3) / np.float32(10.0))
    seed = np.array([65535, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_left_right"] = tf_image_stateless_random_flip_left_right_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_stateless_random_flip_up_down_inputs():
    list_of_inputs = []

    image = np.arange(2*3*1, dtype=np.int32).reshape(2, 3, 1)
    seed = np.array([0, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.arange(2*3*2*3, dtype=np.float32).reshape(2, 3, 2, 3) - 10.0
    seed = np.array([2, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = (np.arange(4*4*1, dtype=np.int32) - 50).reshape(4, 4, 1)
    seed = np.array([123, 456], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array([[[[1.0, np.nan], [np.inf, -np.inf]],
                        [[0.0, -1.0], [np.nan, 5.5]]]], dtype=np.float64)
    seed = np.array([9, 99], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = (np.arange(5*1*2, dtype=np.int32) % 2).reshape(5, 1, 2)
    seed = np.array([42, 24], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.arange(3*5*4*1, dtype=np.int32).reshape(3, 5, 4, 1)
    seed = np.array([7, 11], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array([[[1234567]]], dtype=np.int32)
    seed = np.array([31415, 92653], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.linspace(-1, 1, 1*1*5*4).astype(np.float32).reshape(1, 1, 5, 4)
    seed = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = (np.arange(7*3*3).reshape(7, 3, 3).astype(np.float32) / 10.0)
    seed = np.array([100, 200], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.arange(2*4*3*2, dtype=np.int32).reshape(2, 4, 3, 2)
    seed = np.array([2147483647, 2147483646], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.arange(10*10*4, dtype=np.float64).reshape(10, 10, 4) / 255.0
    seed = np.array([555, 777], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = ((np.arange(4*2*2*1) % 3) - 1).astype(np.int32).reshape(4, 2, 2, 1)
    seed = np.array([8, 16], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_up_down_1"] = tf_image_stateless_random_flip_up_down_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_flip_up_down_inputs():
    list_of_inputs = []

    image = np.array([[[1], [2]], [[3], [4]]], dtype=np.uint8)
    seed = (np.int32(2), np.int32(3))
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3)
    seed = (0, 0)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(2 * 3 * 4 * 1, dtype=np.int32).reshape(2, 3, 4, 1) - 10
    seed = (123, 456)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(1 * 1 * 5 * 2, dtype=np.float64).reshape(1, 1, 5, 2)
    seed = (999, 1)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = (np.arange(9) % 2 == 0).reshape(3, 3, 1)
    image = base.astype(np.bool_)
    seed = (5, 6)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.random.randn(2, 1, 4, 3).astype(np.float32)
    seed = (7, 8)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = (np.arange(10, dtype=np.int64) - 5).reshape(1, 5, 2)
    seed = (-12, 34)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.empty((0, 3, 3, 3), dtype=np.float32)
    seed = (11, 22)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = np.arange(10 * 10 * 4, dtype=np.float16).reshape(10, 10, 4)
    seed = (np.int64(2147483647), np.int64(-1))
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    image = (np.arange(25, dtype=np.float32) / 10.0).reshape(5, 5, 1)
    seed = (42, 24)
    input_dict = {"image": image, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_up_down_2"] = tf_image_stateless_random_flip_up_down_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_flip_up_down_inputs():
    list_of_inputs = []

    image = np.random.randint(0, 256, size=(4, 4, 3), dtype=np.uint8)
    seed = [123, 456]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array([[[1.0], [-2.5]], [[3.3], [4.4]]], dtype=np.float32)
    seed = [0, 1]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.arange(2 * 3 * 3 * 1).reshape(2, 3, 3, 1).astype(np.float64)
    seed = [42, 24]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array([[[[-1, 2], [3, -4], [5, 6]], [[-7, 8], [9, -10], [11, 12]]]], dtype=np.int32)
    seed = [7, 8]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.arange(3 * 2 * 1).reshape(3, 2, 1).astype(np.uint8)
    seed = [9, 9]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.random.randint(0, 256, size=(3, 1, 5, 3), dtype=np.uint8)
    seed = [100, 200]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    arr = np.linspace(-10, 10, num=5 * 4 * 2).astype(np.float32)
    arr[3] = np.nan
    arr[7] = np.inf
    image = arr.reshape(5, 4, 2)
    seed = [555, 666]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.random.randint(-1000, 1000, size=(2, 4, 4, 4), dtype=np.int64)
    seed = [31415, 92653]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.array([[[5.5]]], dtype=np.float64)
    seed = [11, 12]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.random.randn(1, 10, 7, 3).astype(np.float32)
    seed = [333, 444]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    image = np.random.randint(0, 255, size=(6, 5, 1), dtype=np.uint8)
    seed = [2021, 2022]
    list_of_inputs.append(copy.deepcopy({"image": image, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_flip_up_down_3"] = tf_image_stateless_random_flip_up_down_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_image_stateless_random_hue_inputs():
    list_of_inputs = []
    rs = np.random.RandomState(42)

    # Input 1
    image = np.array([[[0.1, 0.2, 0.3],
                       [0.4, 0.5, 0.6]],
                      [[0.7, 0.8, 0.9],
                       [0.2, 0.3, 0.4]]], dtype=np.float32)
    max_delta = 0.2
    seed = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 2
    image = rs.rand(4, 4, 3).astype(np.float32)
    max_delta = 0.5
    seed = np.array([123456789, 987654321], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 3 (batched)
    image = rs.rand(3, 8, 8, 3).astype(np.float32)
    max_delta = 0.0
    seed = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 4 (float16)
    image = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float16)
    max_delta = 0.3
    seed = np.array([2025, 1108], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 5 (all ones)
    image = np.ones((10, 10, 3), dtype=np.float32)
    max_delta = 0.05
    seed = np.array([42, 24], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 6 (batched with H=1)
    image = rs.rand(2, 1, 5, 3).astype(np.float32)
    max_delta = 0.49
    seed = np.array([31415, 27182], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 7 (all zeros)
    image = np.zeros((224, 224, 3), dtype=np.float32)
    max_delta = 0.4
    seed = np.array([7, 11], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 8 (float32 small image)
    image = np.array([[[0.95, 0.05, 0.5],
                       [0.25, 0.75, 0.5],
                       [0.0, 1.0, 0.5]]], dtype=np.float32)
    max_delta = 0.1
    seed = np.array([1001, 2002], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 9 (batched, small)
    image = rs.rand(5, 3, 3, 3).astype(np.float32)
    max_delta = 0.3
    seed = np.array([123, 456], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 10 (non-square image)
    image = rs.rand(7, 5, 3).astype(np.float32)
    max_delta = 0.2
    seed = np.array([8080, 9090], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 11 (values near 0 and 1)
    image = np.vstack([
        np.zeros((8, 16, 3), dtype=np.float32),
        np.ones((8, 16, 3), dtype=np.float32)
    ])
    max_delta = 0.0001
    seed = np.array([314, 159], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 12 (batched 2 images 3x3x3)
    image = np.array([
        [[[0.2, 0.3, 0.4],
          [0.5, 0.6, 0.7],
          [0.8, 0.9, 1.0]],
         [[0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6],
          [0.7, 0.8, 0.9]],
         [[0.9, 0.8, 0.7],
          [0.6, 0.5, 0.4],
          [0.3, 0.2, 0.1]]],
        [[[1.0, 0.9, 0.8],
          [0.7, 0.6, 0.5],
          [0.4, 0.3, 0.2]],
         [[0.2, 0.1, 0.0],
          [0.3, 0.4, 0.5],
          [0.6, 0.7, 0.8]],
         [[0.05, 0.95, 0.5],
          [0.25, 0.75, 0.5],
          [0.45, 0.55, 0.5]]]
    ], dtype=np.float32)
    max_delta = 0.25
    seed = np.array([999, 111], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_hue_1"] = tf_image_stateless_random_hue_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_image_stateless_random_hue_inputs():
    list_of_inputs = []

    # Input 1
    image = np.array([[[0.1, 0.2, 0.3],
                       [0.4, 0.5, 0.6]],
                      [[0.7, 0.8, 0.9],
                       [1.0, 1.1, 1.2]]], dtype=np.float32)
    max_delta = np.float32(0.2)
    seed = (np.int32(1), np.int32(2))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 2
    image = (np.random.rand(64, 64, 3) * 255.0).astype(np.float32)
    max_delta = np.float64(0.5)
    seed = (np.int64(12345), np.int64(67890))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 3
    image = (np.random.rand(8, 32, 32, 3)).astype(np.float16)
    max_delta = np.float32(0.0)
    seed = (np.int32(0), np.int32(0))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 4
    image = np.array([[[0.2, -0.3, 1.5]]], dtype=np.float64)
    max_delta = np.float64(0.1)
    seed = (np.int64(-7), np.int64(42))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 5
    image = (np.random.rand(10, 20, 3) * 2.0 - 1.0).astype(np.float32)
    max_delta = np.float32(0.05)
    seed = (np.int32(9999), np.int32(1))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 6
    image = np.ones((1, 128, 128, 3), dtype=np.float32)
    max_delta = np.float32(0.3)
    seed = (np.int64(2021), np.int64(2022))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 7
    image = np.arange(27, dtype=np.float16).reshape(3, 3, 3)
    max_delta = np.float32(0.49)
    seed = (np.int32(2147483647), np.int32(-2147483648))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 8
    image = (np.random.rand(4, 5, 5, 3)).astype(np.float64)
    max_delta = np.float32(0.25)
    seed = (np.int64(0), np.int64(1))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 9
    image = np.zeros((3, 4, 3), dtype=np.float32)
    max_delta = np.float32(0.001)
    seed = (np.int32(42), np.int32(24))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 10
    image = np.array([[[1.0, 0.0, 0.5]],
                      [[0.3, 0.7, 0.2]]], dtype=np.float32)
    max_delta = np.float32(0.15)
    seed = (np.int64(555), np.int64(777))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 11
    image = (np.random.rand(16, 16, 3) * 1000.0).astype(np.float16)
    max_delta = np.float32(0.33)
    seed = (np.int32(314159), np.int32(265358))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    # Input 12
    image = (np.random.randn(2, 2, 2, 3)).astype(np.float64)
    max_delta = np.float64(0.5)
    seed = (np.int64(123), np.int64(456))
    list_of_inputs.append(copy.deepcopy({"image": image, "max_delta": max_delta, "seed": seed}))

    return list_of_inputs

generated_inputs["tf.image.stateless_random_hue_2"] = tf_image_stateless_random_hue_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
np.random.seed(42)
tf.random.set_seed(42)

def tf_image_transpose_inputs():
    list_of_inputs = []

    # Input 1: 3D float32 with negatives
    image = np.array(
        [[[-1.0, 0.5], [2.5, -3.2], [4.1, 5.5]],
         [[7.0, -8.0], [9.2, 10.3], [-11.4, 12.6]]],
        dtype=np.float32
    )
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_1"}))

    # Input 2: 4D uint8 RGB image batch
    image = np.random.randint(0, 256, size=(2, 4, 3, 3)).astype(np.uint8)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_2"}))

    # Input 3: 3D int32 non-square dims
    image = (np.arange(5*2*4).reshape(5, 2, 4).astype(np.int32) - 10)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_3"}))

    # Input 4: 4D float64 with batch=1, height=1
    image = np.linspace(-1, 1, 1*1*5*2).reshape(1, 1, 5, 2).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_4"}))

    # Input 5: 3D int32 with larger values
    image = np.array(
        [[[1000, 2000], [3000, 4000]],
         [[5000, 6000], [7000, 8000]],
         [[9000, 10000], [11000, 12000]]],
        dtype=np.int32
    )
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_5"}))

    # Input 6: 4D int32 with negatives
    image = (np.random.randint(-1000, 1000, size=(4, 3, 2, 3))).astype(np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_6"}))

    # Input 7: 3D complex64
    real = (np.arange(3*3*2) / 10.0).reshape(3, 3, 2).astype(np.float32)
    imag = (-np.arange(3*3*2) / 10.0).reshape(3, 3, 2).astype(np.float32)
    image = (real + 1j * imag).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_7"}))

    # Input 8: 4D float32 single channel
    image = np.random.randn(3, 7, 5, 1).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_8"}))

    # Input 9: 3D int8 grayscale single channel
    image = np.random.randint(-128, 127, size=(2, 3, 1)).astype(np.int8)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_9"}))

    # Input 10: 4D float32 with larger batch and channels
    image = (2.0 * np.random.rand(5, 2, 2, 4) - 1.0).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_10"}))

    # Input 11: 3D float16 RGB
    image = np.random.rand(6, 6, 3).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_11"}))

    # Input 12: 4D bool batch
    image = (np.random.rand(2, 2, 2, 2) > 0.5)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_12"}))

    return list_of_inputs

generated_inputs["tf.image.transpose"] = tf_image_transpose_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_linalg_LinearOperatorCirculant2D_inputs():
    rng = np.random.default_rng(123)
    list_of_inputs = []

    def hermitize(arr):
        a = np.array(arr, dtype=np.complex64, copy=True)
        N0, N1 = a.shape[-2], a.shape[-1]
        out = np.empty_like(a)
        it = np.ndindex(a.shape[:-2])
        for idx in it:
            A = a[idx]
            H = np.empty_like(A)
            for n0 in range(N0):
                for n1 in range(N1):
                    m0 = (-n0) % N0
                    m1 = (-n1) % N1
                    H[n0, n1] = 0.5 * (A[n0, n1] + np.conj(A[m0, m1]))
            out[idx] = H
        return out

    spectrum = np.array([[1.5, 2.0, 3.25],
                         [4.5, 5.0, 6.75]], dtype=np.float32)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex64,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "circ2d_real_pos_2x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    spectrum = np.array([[1+2j, -0.5+0.1j, 0.0-1j],
                         [2-3j,  0.3+0.7j, 2.2+0.0j],
                         [-1+0.5j, 0.7-0.4j, -2.1+1.1j]], dtype=np.complex64)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "circ2d_cplx_3x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = rng.normal(size=(2, 3, 3)) + 1j * rng.normal(size=(2, 3, 3))
    spectrum = hermitize(base.astype(np.complex64))
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "circ2d_batched_hermitian_2x3x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    spectrum = np.array([[0., 1., 0., 2.],
                         [3., 0., 4., 0.],
                         [0., 5., 0., 6.],
                         [7., 0., 8., 0.]], dtype=np.float32)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex64,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "circ2d_real_with_zeros_4x4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    spectrum = np.ones((5, 2, 2), dtype=np.float64)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex128,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "circ2d_batched_ones_5x2x2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = rng.normal(size=(2, 4, 5)).astype(np.float32)
    imag = rng.normal(size=(2, 4, 5)).astype(np.float32)
    spectrum = real + 1j * imag
    input_dict = {
        "spectrum": spectrum.astype(np.complex64),
        "input_output_dtype": np.dtype(np.complex64),
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "circ2d_cplx_batched_2x4x5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    kernel = np.array([[1., 2., 1.],
                       [5., -1., 1.]], dtype=np.float32)
    spectrum = np.fft.fft2(kernel).astype(np.complex64)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "circ2d_fft_kernel_2x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    spectrum = np.array([[1., 2., 3.],
                         [4., 5., 6.],
                         [7., 8., 9.]], dtype=np.float64)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex64,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "circ2d_real_pd_3x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    spectrum = np.array([[2.0]], dtype=np.float32)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.dtype('complex64'),
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "circ2d_scalar_1x1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    spectrum = np.abs(rng.integers(low=1, high=5, size=(2, 3, 2, 2))).astype(np.float32)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex128,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "circ2d_multibatch_pos_2x3x2x2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    spectrum = np.array([[-1.0, -2.0],
                         [-3.0, -4.0]], dtype=np.float32)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.dtype(np.complex64),
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "circ2d_real_negative_2x2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    spectrum = rng.normal(size=(4, 3)).astype(np.float64)
    input_dict = {
        "spectrum": spectrum,
        "input_output_dtype": np.complex64,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "circ2d_real_4x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorCirculant2D"] = tf_linalg_LinearOperatorCirculant2D_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_linalg_linearoperatorhouseholder_inputs():
    list_of_inputs = []

    # Input 1
    vec1 = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.float64)
    input_dict = {
        "reflection_axis": vec1,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_vec2_f64_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    vec2 = np.array([[3.0], [-4.0], [0.0]], dtype=np.float32)
    input_dict = {
        "reflection_axis": vec2,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_col3_f32_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    rs3 = np.random.RandomState(0)
    vec3 = rs3.randn(5, 4).astype(np.float32)
    input_dict = {
        "reflection_axis": vec3,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_batch5x4_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    rs4 = np.random.RandomState(1)
    vec4 = rs4.randn(2, 3, 6, 1).astype(np.float64)
    input_dict = {
        "reflection_axis": vec4,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_batch2x3_col6_f64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    vec5 = np.array([1 + 2j, -0.5 + 0.3j, 2 - 1j, 0 - 3j], dtype=np.complex64)
    input_dict = {
        "reflection_axis": vec5,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_vec4_c64_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    rs6a = np.random.RandomState(2)
    rs6b = np.random.RandomState(3)
    vec6 = (rs6a.randn(3, 5) + 1j * rs6b.randn(3, 5)).astype(np.complex128)
    input_dict = {
        "reflection_axis": vec6,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_batch3x5_c128"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    vec7 = np.array([-2.5], dtype=np.float32)
    input_dict = {
        "reflection_axis": vec7,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_scalar1_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    vec8 = np.array([((-1.0) ** i) * (i + 1) / 10.0 for i in range(10)], dtype=np.float16)
    input_dict = {
        "reflection_axis": vec8,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_vec10_f16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    vec9 = np.array([1e-12, -1e-11, 2e-12, -3e-12, 4e-12, -5e-12, 6e-12, -7e-12], dtype=np.float32)
    input_dict = {
        "reflection_axis": vec9,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_small_vec8_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    rs10 = np.random.RandomState(4)
    vec10 = rs10.randn(2, 2, 2, 7).astype(np.float64)
    input_dict = {
        "reflection_axis": vec10,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_batch2x2x2_vec7_f64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    vec11 = np.array([[0 + 1j], [0 + 2j], [0 - 3j], [0 + 4j]], dtype=np.complex64)
    input_dict = {
        "reflection_axis": vec11,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_col4_imag_c64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    vec12 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    input_dict = {
        "reflection_axis": vec12,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_onehot4_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorHouseholder"] = tf_linalg_linearoperatorhouseholder_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_LinearOperatorLowerTriangular_inputs():
    list_of_inputs = []

    # Input 1
    tril = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt1_float32_2x2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tril = np.diag(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "lt2_float64_diag_3x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tril = np.array([
        [0.0, -1.0, 2.0, 3.0],
        [5.0, -1.0, 0.5, 2.0],
        [-7.0, 8.0, 2.0, -3.0],
        [4.0, -6.0, 1.0, 0.5]
    ], dtype=np.float32)
    input_dict = {
        "tril": tril,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt3_float32_4x4_singular_hint"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (batch [2, 2, 2])
    tril = np.array([
        [[2.0, -1.0], [0.3, -4.0]],
        [[-3.0, 5.0], [1.2, 6.0]]
    ], dtype=np.float64)
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt4_float64_batch2_2x2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (batch [3, 4, 4])
    tril = np.random.randn(3, 4, 4).astype(np.float32)
    input_dict = {
        "tril": tril,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt5_float32_batch3_4x4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (batch [2, 3, 4, 4])
    tril = np.random.randn(2, 3, 4, 4).astype(np.float64)
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt6_float64_batch2x3_4x4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (complex64, 2x2 diagonal positive)
    tril = np.array([[2.0 + 0.0j, 0.0 + 0.0j],
                     [0.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex64)
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "lt7_complex64_diag_2x2_pd"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (complex128 batch [2, 2, 2])
    tril = np.array([
        [[1.0 + 2.0j, -3.0 + 0.0j],
         [4.0 - 1.0j, -2.0 + 0.5j]],
        [[-1.0 + 0.0j, 2.0 + 1.0j],
         [3.0 + 4.0j, 1.0 - 2.0j]]
    ], dtype=np.complex128)
    input_dict = {
        "tril": tril,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt8_complex128_batch2_2x2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (float16 5x5)
    tril = np.tril(np.array([
        [0.5, -1.2, 0.0, 2.1, -3.0],
        [1.0, 0.8, -0.7, 0.0, 1.5],
        [2.2, -1.1, 1.3, -0.4, 0.0],
        [0.0, 1.7, -2.5, 0.6, 0.2],
        [-1.4, 0.0, 0.9, -0.8, 2.0]
    ], dtype=np.float16))
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt9_float16_5x5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (batch [1, 2, 3, 3])
    tril = np.arange(1, 1 + 1*2*3*3, dtype=np.float32).reshape(1, 2, 3, 3)
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt10_float32_batch1x2_3x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (1x1 negative)
    tril = np.array([[-2.0]], dtype=np.float64)
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt11_float64_1x1_negative"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (batch [2, 5, 5])
    tril = np.tril(np.random.uniform(-5, 5, size=(2, 5, 5)).astype(np.float32))
    input_dict = {
        "tril": tril,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt12_float32_batch2_5x5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorLowerTriangular"] = tf_linalg_LinearOperatorLowerTriangular_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_linalg_LinearOperatorZeros_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "num_rows": 2,
        "num_columns": 2,
        "batch_shape": [],
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_2x2_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "num_rows": 3,
        "num_columns": 4,
        "batch_shape": [],
        "dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": False,
        "name": "zeros_3x4_f64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "num_rows": 2,
        "num_columns": 2,
        "batch_shape": [2],
        "dtype": np.complex64,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_batch2_2x2_c64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "num_rows": 5,
        "num_columns": 5,
        "batch_shape": [2, 3],
        "dtype": np.complex128,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_b23_5x5_c128"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "num_rows": 7,
        "num_columns": 7,
        "batch_shape": [0],
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_emptybatch_7x7_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "num_rows": 1,
        "num_columns": 3,
        "batch_shape": [4],
        "dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": False,
        "name": "zeros_b4_1x3_f64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "num_rows": 0,
        "num_columns": 0,
        "batch_shape": [],
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_0x0_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "num_rows": 2,
        "num_columns": 2,
        "batch_shape": [1, 0, 2],
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_b102_2x2_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "num_rows": 1000,
        "num_columns": 1000,
        "batch_shape": [],
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_1000x1000_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "num_rows": 8,
        "num_columns": 2,
        "batch_shape": [5, 1],
        "dtype": np.complex64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "zeros_b51_8x2_c64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_dict = {
        "num_rows": 4,
        "num_columns": 4,
        "batch_shape": [3],
        "dtype": np.float16,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_b3_4x4_f16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_dict = {
        "num_rows": 6,
        "num_columns": 6,
        "batch_shape": [1, 1, 1],
        "dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_b111_6x6_f64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorZeros_1"] = tf_linalg_LinearOperatorZeros_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_linalg_LinearOperatorZeros_inputs():
    list_of_inputs = []

    input_dict = {
        "num_rows": np.array(2, dtype=np.int32),
        "num_columns": np.array(2, dtype=np.int32),
        "batch_shape": np.array([], dtype=np.int32),
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_2x2_f32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(3, dtype=np.int64),
        "num_columns": np.array(5, dtype=np.int64),
        "batch_shape": np.array([4], dtype=np.int64),
        "dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "zeros_3x5_batch4_f64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(0, dtype=np.int32),
        "num_columns": np.array(0, dtype=np.int32),
        "batch_shape": np.array([2, 3], dtype=np.int32),
        "dtype": np.float16,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "empty_0x0_b23_f16",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(0, dtype=np.int32),
        "num_columns": np.array(7, dtype=np.int32),
        "batch_shape": np.array([], dtype=np.int32),
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": False,
        "name": "zeros_0x7",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(9, dtype=np.int64),
        "num_columns": np.array(0, dtype=np.int64),
        "batch_shape": np.array([0], dtype=np.int64),
        "dtype": np.complex64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "zeros_9x0_c64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(1, dtype=np.int32),
        "num_columns": np.array(1, dtype=np.int32),
        "batch_shape": np.array([5, 1], dtype=np.int32),
        "dtype": np.complex128,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_1x1_b5x1_c128",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(10, dtype=np.int64),
        "num_columns": np.array(10, dtype=np.int64),
        "batch_shape": np.array([2, 0], dtype=np.int64),
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_10x10_b2x0_f32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(5, dtype=np.int32),
        "num_columns": np.array(3, dtype=np.int32),
        "batch_shape": np.array([3, 2], dtype=np.int32),
        "dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "zeros_5x3_b3x2_f64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(128, dtype=np.int32),
        "num_columns": np.array(256, dtype=np.int32),
        "batch_shape": np.array([2], dtype=np.int32),
        "dtype": np.float32,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": False,
        "name": "zeros_128x256_b2_f32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(4, dtype=np.int64),
        "num_columns": np.array(4, dtype=np.int64),
        "batch_shape": np.array([1, 1, 1], dtype=np.int64),
        "dtype": np.float16,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_4x4_b1x1x1_f16",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(2, dtype=np.int32),
        "num_columns": np.array(3, dtype=np.int32),
        "batch_shape": np.array([2, 1, 3], dtype=np.int32),
        "dtype": np.complex64,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "zeros_2x3_b2x1x3_c64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": np.array(7, dtype=np.int64),
        "num_columns": np.array(7, dtype=np.int64),
        "batch_shape": np.array([], dtype=np.int64),
        "dtype": np.float64,
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_7x7_nobatch_f64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorZeros_2"] = tf_linalg_LinearOperatorZeros_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_linearoperatorzeros_inputs():
    list_of_inputs = []

    input_dict = {
        "num_rows": 2,
        "num_columns": 2,
        "batch_shape": (),
        "dtype": np.dtype('float32'),
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_2x2_f32_no_batch"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 3,
        "num_columns": 5,
        "batch_shape": (4,),
        "dtype": np.dtype('float64'),
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "zeros_3x5_f64_b4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 0,
        "num_columns": 0,
        "batch_shape": (),
        "dtype": np.dtype('float32'),
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_0x0_f32_empty"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 1,
        "num_columns": 1,
        "batch_shape": (2,),
        "dtype": np.dtype('complex64'),
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_1x1_c64_b2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 7,
        "num_columns": 3,
        "batch_shape": (2, 3),
        "dtype": np.dtype('float16'),
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "zeros_7x3_f16_b2x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 10,
        "num_columns": 10,
        "batch_shape": (1,),
        "dtype": np.dtype('complex128'),
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_10x10_c128_b1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 2,
        "num_columns": 0,
        "batch_shape": (3,),
        "dtype": np.dtype('float32'),
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": False,
        "name": "zeros_2x0_f32_b3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 0,
        "num_columns": 5,
        "batch_shape": (0,),
        "dtype": np.dtype('float64'),
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": False,
        "assert_proper_shapes": True,
        "name": "zeros_0x5_f64_b0"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 4,
        "num_columns": 4,
        "batch_shape": (3, 1, 2),
        "dtype": np.dtype('float64'),
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_4x4_f64_b3x1x2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 5,
        "num_columns": 5,
        "batch_shape": (),
        "dtype": np.dtype('complex64'),
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": True,
        "name": "zeros_5x5_c64_assert"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "num_rows": 6,
        "num_columns": 6,
        "batch_shape": (2, 2),
        "dtype": np.dtype('float32'),
        "is_non_singular": False,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "assert_proper_shapes": False,
        "name": "zeros_6x6_f32_b2x2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorZeros_3"] = tf_linalg_linearoperatorzeros_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_trace_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    name = "trace_int32_2x2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[-1, -2, -3],
                  [-4, -5, -6],
                  [-7, -8, -9]], dtype=np.int64)
    name = "trace_int64_neg_3x3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[0.5, -1.2, 3.4, 0.0],
                  [2.1, 4.2, -0.7, 1.1],
                  [9.3, 2.2, -3.3, 4.4],
                  [1.0, -2.0, 3.0, 4.0]], dtype=np.float32)
    name = "trace_float32_4x4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[1.25, 2.5, 3.75],
                  [4.125, 5.625, 6.875],
                  [7.0, 8.25, 9.5]], dtype=np.float64)
    name = "trace_float64_3x3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[1+2j, 3-4j],
                  [-5+0.5j, 2+0j]], dtype=np.complex64)
    name = "trace_complex64_2x2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([[1-1j, 0+2j, 3+0j],
                  [4+4j, -5+5j, 6-6j],
                  [7+0.1j, 8-0.2j, 9+0.3j]], dtype=np.complex128)
    name = "trace_complex128_3x3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
        [[-1, -2, -3],
         [-4, -5, -6],
         [-7, -8, -9]]
    ], dtype=np.int32)
    name = "trace_batch_int32_2x3x3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = (np.arange(5*2*2, dtype=np.float32).reshape(5, 2, 2) - 5.0)
    name = "trace_batch_float32_5x2x2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = (np.arange(2*3*4*4, dtype=np.float32).reshape(2, 3, 4, 4) * 0.1 - 10.0)
    name = "trace_4d_float32_2x3x4x4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.empty((0, 0), dtype=np.float32)
    name = "trace_empty_0x0_float32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    x = np.array([[127, -128],
                  [10, -10]], dtype=np.int8)
    name = "trace_int8_2x2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    x = np.array([[1.0, -2.0, 3.0],
                  [-4.0, 5.0, -6.0],
                  [7.0, -8.0, 9.0]], dtype=np.float16)
    name = "trace_float16_3x3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.trace"] = tf_linalg_trace_inputs()

import numpy as np
import tensorflow as tf
import copy

np.random.seed(42)

def tf_math_bessel_i0_inputs():
    list_of_inputs = []

    x = np.array(0.0, dtype=np.float32)
    name = "i0_scalar_f32_zero"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
    name = "i0_vector_f32_mixed"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[0.1, -0.2], [3.0, -4.0]], dtype=np.float64)
    name = "i0_matrix_f64_small_big"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.random.uniform(-2, 2, size=(2, 3, 4)).astype(np.float16)
    name = "i0_tensor3d_f16_uniform"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([], dtype=np.float32)
    name = "i0_empty_vector_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([20.0, 50.0, -20.0, -50.0], dtype=np.float64)
    name = "i0_large_magnitude_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.arange(-5, 5, dtype=np.float32)[::2]
    name = "i0_noncontiguous_slice_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    x = base.T
    name = "i0_transposed_view_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.zeros((3, 3), dtype=np.float16)
    name = "i0_zeros_matrix_f16"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float64)
    name = "i0_nan_inf_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([1e-8, -1e-8, 1e-12, -1e-12], dtype=np.float32)
    name = "i0_very_small_values_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.linspace(-3, 3, 12, dtype=np.float64).reshape(2, 1, 2, 3)
    name = "i0_tensor4d_f64_linspace"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    return list_of_inputs

generated_inputs["tf.math.bessel_i0"] = tf_math_bessel_i0_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_bessel_i1e_inputs():
    list_of_inputs = []

    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    name = "vec_float32_basic"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[0.1, -2.5, 5.0], [10.0, -7.5, 0.0]], dtype=np.float64)
    name = "matrix_float64_mixed"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array(2.5, dtype=np.float32)
    name = "scalar_float32_positive"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[-3.0, -1.0, 0.0, 1.0, 3.0],
                  [4.0, -4.0, 2.0, -2.0, 0.5]], dtype=np.float16)
    name = "matrix_float16_varied"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    base = np.arange(24, dtype=np.float64).reshape(4, 6) - 12.0
    x = base[:, ::-2]
    name = "noncontiguous_slice_float64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.asfortranarray(np.array([[0.01, 0.1, 1.0],
                                    [2.0, 5.0, 10.0],
                                    [-1.0, -0.1, -0.01]], dtype=np.float32))
    name = "fortran_order_float32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.random.RandomState(42).uniform(-3.0, 3.0, size=(2, 3, 4)).astype(np.float64)
    name = "tensor3d_float64_random"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([-50.0, -20.0, 0.0, 20.0, 50.0, 100.0], dtype=np.float64)
    name = "large_magnitude_float64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.zeros((5,), dtype=np.float16)
    name = "zeros_vector_float16"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.full((2, 2, 2), 1e-6, dtype=np.float32)
    name = "small_values_float32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([np.nan, np.inf, -np.inf, 1.0, -1.0], dtype=np.float64)
    name = "nan_inf_float64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.transpose(np.arange(12, dtype=np.float32).reshape(3, 4))
    name = "transposed_noncontiguous_float32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([], dtype=np.float64)
    name = "empty_array_float64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    return list_of_inputs

generated_inputs["tf.math.bessel_i1e"] = tf_math_bessel_i1e_inputs()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(42)
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_erf_inputs():
    list_of_inputs = []

    x = np.array(0.0, dtype=np.float32)
    name = "erf_scalar_zero_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array(-1.5, dtype=np.float64)
    name = "erf_scalar_neg_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([1.0, 2.0, 3.0, -0.5], dtype=np.float32)
    name = "erf_1d_mixed_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[1.0, -1.0], [2.0, -2.0], [0.0, 0.0]], dtype=np.float16)
    name = "erf_2d_f16"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.random.uniform(-3, 3, size=(2, 2, 3)).astype(np.float32)
    name = "erf_3d_random_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([], dtype=np.float32)
    name = "erf_empty_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([-10.0, -5.0, -2.0, 2.0, 5.0, 10.0], dtype=np.float64)
    name = "erf_large_magnitudes_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    name = "erf_inf_nan_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.arange(0, 10, dtype=np.float32)[::2]
    name = "erf_noncontiguous_stride_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float16).reshape(-1, 1)
    name = "erf_column_vector_f16"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.linspace(-3, 3, 24, dtype=np.float32).reshape(2, 2, 2, 3)
    name = "erf_4d_linspace_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([1e-8, -1e-8, 1e-12, -1e-12], dtype=np.float64)
    name = "erf_tiny_values_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    return list_of_inputs

generated_inputs["tf.math.erf"] = tf_math_erf_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_floor_inputs():
    list_of_inputs = []

    x = np.array([1.3324, -1.5, 5.555, -2.532, 0.99, np.inf], dtype=np.float32)
    name = "basic_float32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-0.1, 0.0, 0.1], [1.999, 2.001, -2.001]], dtype=np.float64)
    name = "matrix_float64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(3.7, dtype=np.float32)
    name = "scalar32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-3.0001, dtype=np.float64)
    name = "neg_scalar64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan, -np.inf, np.inf, -0.0, 0.0], dtype=np.float32)
    name = "nan_inf"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(
        [[[-1.2, -1.8], [1.2, 1.8]], [[0.0, -0.0], [123.5, -123.5]]],
        dtype=np.float16
    )
    name = "tensor3d_fp16"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = (np.arange(12, dtype=np.float32).reshape(2, 1, 2, 3) / 3.0) - 2.5
    name = "tensor4d_float32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = (np.arange(12, dtype=np.float64).reshape(3, 4) / 2.0) - 3.0
    x = base.T
    name = "non_contiguous64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    name = "empty_vec32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.empty((2, 0, 3), dtype=np.float64)
    name = "empty_tensor_float64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-12, -1e-12, 1e-8, -1e-8], dtype=np.float64)
    name = "tiny_values64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e10 + 0.9, -1e10 + 0.1, 3.5e5, -3.5e5], dtype=np.float64)
    name = "large_values64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.floor"] = tf_math_floor_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_in_top_k_inputs():
    list_of_inputs = []

    # Input 1: Example-like
    targets = np.array([0, 1, 3], dtype=np.int32)
    predictions = np.array([
        [1.2, -0.3, 2.8, 5.2],
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.3, 0.3]
    ], dtype=np.float32)
    k = 2
    name = "case_doc_like"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 2: Ties across boundary
    targets = np.array([2, 4], dtype=np.int64)
    predictions = np.array([
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0]
    ], dtype=np.float32)
    k = 2
    name = "ties_across_boundary"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 3: k equals number of classes
    targets = np.array([1, 0], dtype=np.int32)
    predictions = np.array([
        [0.0, 0.0, -1.0],
        [2.0, -2.0, 0.001]
    ], dtype=np.float32)
    k = 3
    name = "k_equals_classes"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 4: Single-class per example with non-finite values
    targets = np.array([0, 0, 0, 0], dtype=np.int64)
    predictions = np.array([
        [-0.1],
        [0.0],
        [np.nan],
        [np.inf]
    ], dtype=np.float32)
    k = 1
    name = "single_class_non_finite"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 5: Negative values and -inf at target class
    targets = np.array([2, 1, 3], dtype=np.int32)
    predictions = np.array([
        [-5.0, -0.2, -0.1, -0.3],
        [0.3, -np.inf, 0.3, 0.3],
        [10.0, 9.0, 8.0, 7.0]
    ], dtype=np.float32)
    k = 2
    name = "negatives_and_inf_target"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 6: Zero batch size
    targets = np.array([], dtype=np.int32)
    predictions = np.empty((0, 3), dtype=np.float32)
    k = 1
    name = "zero_batch"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 7: Larger class count with linspace patterns
    targets = np.array([0, 49], dtype=np.int64)
    row1 = np.linspace(-1.0, 1.0, 50, dtype=np.float32)
    row2 = np.linspace(1.0, -1.0, 50, dtype=np.float32)
    predictions = np.stack([row1, row2], axis=0).astype(np.float32)
    k = 5
    name = "large_class_count"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 8: Extreme float32 values
    targets = np.array([0, 1], dtype=np.int32)
    predictions = np.array([
        [1e38, -1e38, 0.0],
        [-1e-30, 1e-30, 0.0]
    ], dtype=np.float32)
    k = 1
    name = "extreme_float32"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 9: NaNs in non-target columns
    targets = np.array([1, 2], dtype=np.int64)
    predictions = np.array([
        [np.nan, 0.5, 0.5, 0.5],
        [0.1, np.nan, 0.2, 0.3]
    ], dtype=np.float32)
    k = 2
    name = "nans_in_non_target"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 10: All zeros with k=1 (ties)
    targets = np.array([0, 1, 2], dtype=np.int32)
    predictions = np.zeros((3, 4), dtype=np.float32)
    k = 1
    name = "all_zeros_k1"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 11: Mixed positives/negatives with k=3
    targets = np.array([4, 0, 2], dtype=np.int64)
    predictions = np.array([
        [-3.0, 0.2, 0.1, -0.5, 0.7, -0.1],
        [1.5, 0.3, 0.2, 0.1, -0.4, -0.2],
        [0.0, -0.1, 0.9, 0.8, 0.7, 0.6]
    ], dtype=np.float32)
    k = 3
    name = "mixed_values_k3"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    # Input 12: k equals classes with varied patterns
    targets = np.array([5, 3, 1], dtype=np.int32)
    predictions = np.array([
        [0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
        [5.0, 4.0, -1.0, -2.0, -3.0, -4.0],
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ], dtype=np.float32)
    k = 6
    name = "k_equals_classes_varied"
    list_of_inputs.append(copy.deepcopy({
        "targets": targets, "predictions": predictions, "k": k, "name": name
    }))

    return list_of_inputs

generated_inputs["tf.math.in_top_k"] = tf_math_in_top_k_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_invert_permutation_inputs():
    list_of_inputs = []

    x = np.array([0], dtype=np.int32)
    input_dict = {"x": x, "name": "case_len1_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 0], dtype=np.int64)
    input_dict = {"x": x, "name": "case_len2_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0, 1, 2], dtype=np.int32)
    input_dict = {"x": x, "name": "case_identity_len3_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([2, 0, 1], dtype=np.int64)
    input_dict = {"x": x, "name": "case_len3_shuffle_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([3, 4, 0, 2, 1], dtype=np.int32)
    input_dict = {"x": x, "name": "case_len5_example_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([4, 3, 2, 1, 0], dtype=np.int64)
    input_dict = {"x": x, "name": "case_len5_reverse_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3, 4, 5, 0], dtype=np.int32)
    input_dict = {"x": x, "name": "case_len6_rotate_left_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([6, 0, 2, 4, 1, 5, 3], dtype=np.int32)
    input_dict = {"x": x, "name": "case_len7_custom_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([7, 3, 0, 2, 5, 1, 6, 4], dtype=np.int64)
    input_dict = {"x": x, "name": "case_len8_random_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int32)
    input_dict = {"x": x, "name": "case_len9_reverse_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1], dtype=np.int32)
    input_dict = {"x": x, "name": "case_len10_rotate_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.arange(20, dtype=np.int64)[::-1]
    input_dict = {"x": x, "name": "case_len20_reverse_int64"}
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

    x = np.array([5.0, np.inf, 6.8, np.inf], dtype=np.float32)
    input_dict = {"x": x, "name": "simple_1d_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.NINF, -1.0, 0.0, 1.0, np.PINF], dtype=np.float64)
    input_dict = {"x": x, "name": "neg_and_pos_inf_f64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(np.inf, dtype=np.float16)
    input_dict = {"x": x, "name": "scalar_inf_f16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-np.inf, np.nan], [3.4, np.inf]], dtype=np.float16)
    input_dict = {"x": x, "name": "matrix_with_nan_and_inf_f16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(
        [
            [[0.0, 1.0, np.inf], [-np.inf, 5.5, np.nan]],
            [[2.2, 3.3, 4.4], [6.6, -7.7, np.inf]],
        ],
        dtype=np.float32,
    )
    input_dict = {"x": x, "name": "tensor3d_mixed_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float64)
    input_dict = {"x": x, "name": "empty_1d_f64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.empty((0, 3), dtype=np.float32)
    input_dict = {"x": x, "name": "empty_2d_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e308, -1e308, 1e309, -1e309], dtype=np.float64)
    input_dict = {"x": x, "name": "extremal_values_f64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[[np.inf, -np.inf, 0.0]], [[1.0, 2.0, 3.0]]]], dtype=np.float32)
    input_dict = {"x": x, "name": "four_d_shape_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.linspace(-5, 5, 10, dtype=np.float32)
    x = base[::2].copy()
    x[1] = np.inf
    x[3] = -np.inf
    input_dict = {"x": x, "name": "strided_view_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([65504.0, 70000.0, -70000.0, -65504.0], dtype=np.float16)
    input_dict = {"x": x, "name": "f16_overflow_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.5, -3.14, -0.0], dtype=np.float32)
    input_dict = {"x": x, "name": "finite_negatives_only_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.is_inf"] = tf_math_is_inf_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_lgamma_inputs():
    list_of_inputs = []

    x = np.array([0.0, 0.5, 1.0, 4.5, -4.0, -5.6], dtype=np.float32)
    name = "case1_vector32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(5.0, dtype=np.float64)
    name = "case2_scalar64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    name = "case3_matrix32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(
        [
            [[-2.5, -1.2, 0.0], [0.3, 2.0, 3.3]],
            [[-0.5, 1.5, 4.0], [5.5, -3.7, 7.2]],
        ],
        dtype=np.float64,
    )
    name = "case4_tensor3d64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-10.5, -1.5, 0.1, 10.0], dtype=np.float16)
    name = "case5_vector16"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([20.0, 50.0, 100.0], dtype=np.float64)
    name = "case6_large_values64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan, np.inf, -np.inf, 3.14, -3.14], dtype=np.float32)
    name = "case7_special32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[[0.2, 1.2, 2.2]], [[-0.2, -1.2, -2.2]]]], dtype=np.float32)
    name = "case8_tensor4d32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    name = "case9_empty32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-0.999, -1.001, -1.999, -2.001, -10.0001], dtype=np.float64)
    name = "case10_near_neg_integers64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-10, 1e-20, 1e-5, 1e-2], dtype=np.float64)
    name = "case11_small_positive64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.25, 0.75], [1.25, 1.75], [-0.25, -0.75]], dtype=np.float32)
    name = "case12_rect32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.lgamma"] = tf_math_lgamma_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_log1p_inputs():
    list_of_inputs = []

    x = np.array([0.0, 0.5, 1.0, 5.0], dtype=np.float32)
    name = "log1p_float32_1d"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[-0.99999994, -0.5], [0.0, 10.0]], dtype=np.float64)
    name = "log1p_float64_2d_neg"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = (np.arange(-6, 6, dtype=np.float16) / 10).reshape(2, 3, 2)
    name = "log1p_float16_3d"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([1+2j, -0.5+0.3j, -2+5j], dtype=np.complex64)
    name = "log1p_complex64_1d"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[0+0j, -1+1j], [3-4j, -0.999+0.001j]], dtype=np.complex128)
    name = "log1p_complex128_2d"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array(-0.9, dtype=np.float32)
    name = "log1p_scalar_float32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([1e-10, 1e10, -0.9999999999], dtype=np.float64)
    name = "log1p_large_float64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([], dtype=np.float32)
    name = "log1p_empty"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    base = np.arange(-1.0, 5.0, 0.5, dtype=np.float32)
    x = base[::2]
    name = "log1p_noncontig_slice"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([np.nan, np.inf, -np.inf, -1.0, 0.0], dtype=np.float32)
    name = "log1p_nan_inf"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.zeros((2, 1, 3, 4), dtype=np.float32)
    x[0, 0, 1, 2] = -0.5
    x[1, 0, 2, 3] = 10.0
    name = "log1p_4d_float32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([-1.0+1e-12j, -1.0-1e-12j, 0.0+0.0j], dtype=np.complex128)
    name = "log1p_complex_branch"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    return list_of_inputs

generated_inputs["tf.math.log1p"] = tf_math_log1p_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_nextafter_inputs():
    list_of_inputs = []

    x1 = np.array([0.0, -1.0, 3.5], dtype=np.float32)
    x2 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    name = "simple_1d_f32"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array([[1.0, 2.0], [-3.0, 4.0]], dtype=np.float64)
    x2 = np.array([[2.0, 1.0], [-4.0, 3.0]], dtype=np.float64)
    name = "matrix_f64"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array([[0.0], [1.0], [-1.0]], dtype=np.float32)
    x2 = np.array([[1.0, 0.0, -2.0, 3.0]], dtype=np.float32)
    name = "broadcast_f32"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array(-0.0, dtype=np.float64)
    x2 = np.array(0.0, dtype=np.float64)
    name = "negzero_to_poszero_scalar_f64"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array([np.inf, -np.inf, 1.0], dtype=np.float64)
    x2 = np.array([0.0, 0.0, np.inf], dtype=np.float64)
    name = "infinities_f64"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array([np.nan, 1.0, -2.0], dtype=np.float32)
    x2 = np.array([2.0, np.nan, -3.0], dtype=np.float32)
    name = "nans_mixed_f32"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    min_subnormal_f32 = np.nextafter(np.float32(0.0), np.float32(1.0))
    x1 = np.zeros((2, 2, 3), dtype=np.float32)
    x2 = np.full((2, 2, 3), min_subnormal_f32, dtype=np.float32)
    name = "zeros_towards_min_subnormal_f32"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    base = np.linspace(-5.0, 6.0, 12, dtype=np.float64).reshape(3, 4)
    x1 = base[:, ::2]
    x2 = x1 + 0.1
    name = "non_contiguous_f64"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array([3.4e38, -3.4e38], dtype=np.float32)
    x2 = np.array([np.inf, -np.inf], dtype=np.float32)
    name = "near_max_f32"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array([[[[-1.0, 0.0, 1.0]]], [[[2.0, -2.0, 0.5]]]], dtype=np.float64)
    x2 = np.arange(12, dtype=np.float64).reshape(1, 3, 4, 1)
    name = "broadcast_4d_f64"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array([0.1, -0.1, 0.0], dtype=np.float32)
    x2 = np.array([0.1, -0.1, 0.0], dtype=np.float32)
    name = "x1_equals_x2_f32"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array([], dtype=np.float64)
    x2 = np.array([], dtype=np.float64)
    name = "empty_f64"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    min_subnormal_f64 = np.nextafter(np.float64(0.0), np.float64(1.0))
    x1 = np.array([-0.0, 0.0, min_subnormal_f64], dtype=np.float64)
    x2 = np.array([-1.0, 1.0, 0.0], dtype=np.float64)
    name = "signed_zero_and_subnormal_f64"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    x2 = np.array(0.0, dtype=np.float32)
    name = "vector_towards_scalar_f32"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    return list_of_inputs

generated_inputs["tf.math.nextafter"] = tf_math_nextafter_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_pow_inputs():
    list_of_inputs = []

    # Input 1: float32 vector and scalar exponent (0-D)
    x = np.array([1.0, 2.0, 3.0, 4.0, -5.0], dtype=np.float32)
    y = np.array(2.5, dtype=np.float32)
    name = "pow_float32_vec_scalar"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 2: float64 matrix elementwise
    x = np.array([[2.0, 3.0, 4.0], [0.5, 1.5, -2.5]], dtype=np.float64)
    y = np.array([[8.0, 2.0, 0.5], [3.0, 0.0, 1.0]], dtype=np.float64)
    name = "pow_float64_matrix"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 3: int32 2x2 positive exponents
    x = np.array([[2, 3], [4, 5]], dtype=np.int32)
    y = np.array([[3, 2], [1, 0]], dtype=np.int32)
    name = "pow_int32_2x2"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 4: int64 broadcasting (2,1,3) with (1,4,1)
    x = np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.int64)  # shape (2,1,3)
    y = np.array([[[0], [1], [2], [3]]], dtype=np.int64)      # shape (1,4,1)
    name = "pow_int64_broadcast"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 5: complex64 vector elementwise complex exponents
    x = np.array([1+2j, -1+1j, 0+1j, 2-3j], dtype=np.complex64)
    y = np.array([2+0j, 0.5+0.5j, -1+0j, 1-1j], dtype=np.complex64)
    name = "pow_complex64_vector"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 6: complex128 matrix with scalar exponent
    x = np.array([[1+0j, -2+2j], [3-1j, -4-4j]], dtype=np.complex128)
    y = np.array(2-0.5j, dtype=np.complex128)
    name = "pow_complex128_scalar_exp"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 7: float16 3-D arrays
    x = np.array([[[1.0, 2.0], [3.0, 4.0]],
                  [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float16)
    y = np.array([[[0.5, 1.0], [1.5, 2.0]],
                  [[2.5, 3.0], [0.0, -1.0]]], dtype=np.float16)
    name = "pow_float16_3d"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 8: float32 negative bases with fractional exponents
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    y = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    name = "pow_float32_negative_bases_fractional_exp"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 9: float32 broadcasting (2,3,1) with (1,3,4)
    x = np.array([[[1.0], [2.0], [3.0]],
                  [[4.0], [5.0], [6.0]]], dtype=np.float32)  # (2,3,1)
    y = np.array([[[1.0, 2.0, 3.0, 4.0],
                   [0.5, 1.5, 2.5, 3.5],
                   [0.0, -1.0, -2.0, -3.0]]], dtype=np.float32)  # (1,3,4)
    name = "pow_float32_broadcast_2"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 10: float64 zeros and zero exponent
    x = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    y = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    name = "pow_float64_zero_exp"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 11: int32 vector with zero exponents
    x = np.array([10, -3, 0, 2], dtype=np.int32)
    y = np.array([0, 0, 0, 0], dtype=np.int32)
    name = "pow_int32_zero_exp"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 12: float32 scalar base with vector exponents (broadcast)
    x = np.array(1.1, dtype=np.float32)
    y = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    name = "pow_float32_scalar_base_vector_exp"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    return list_of_inputs

generated_inputs["tf.math.pow"] = tf_math_pow_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_special_bessel_j1_inputs():
    list_of_inputs = []

    x = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    name = "case_vec_f32_basic"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.0, -1.5, 3.2], [4.5, -6.7, 8.9]], dtype=np.float64)
    name = "case_matrix_f64_negpos"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(0.0, dtype=np.float32)
    name = "case_scalar_zero_f32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[-0.5, -0.6], [0.7, -0.8]]], dtype=np.float16)
    name = "case_3d_f16_small_vals"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    name = "case_empty_1d_f32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-6, 1e-3, 1e-1, 1.0, 10.0, 100.0, 1e3], dtype=np.float64)
    name = "case_wide_range_f64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-0.1, -1.0, -5.5, -10.0], dtype=np.float32)
    name = "case_negatives_f32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.arange(20, dtype=np.float64)
    x = base[::2]
    name = "case_noncontiguous_view_f64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = (np.arange(6, dtype=np.float32).reshape(1, 2, 1, 3) / 10.0)
    name = "case_4d_f32_arange"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.empty((0, 3), dtype=np.float32)
    name = "case_empty_2d_f32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.linspace(-3.0, 3.0, num=7, dtype=np.float32)
    name = "case_linspace_f32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.arange(4, dtype=np.float64).reshape(1, 1, 2, 1, 2) - 1.5
    name = "case_5d_f64_shifted"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.special.bessel_j1"] = tf_math_special_bessel_j1_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_special_fresnel_cos_inputs():
    list_of_inputs = []

    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    name = "fresnel_cos_vec_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array(0.75, dtype=np.float64)
    name = "fresnel_cos_scalar_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[-2.0, -1.0], [1.0, 2.0]], dtype=np.float64)
    name = "fresnel_cos_matrix_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[[0.1], [0.2], [0.3]], [[-0.1], [-0.2], [-0.3]]], dtype=np.float32)
    name = "fresnel_cos_3d_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.linspace(-10.0, 10.0, 21, dtype=np.float64)
    name = "fresnel_cos_linspace_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.zeros((4,), dtype=np.float32)
    name = "fresnel_cos_zeros_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([1e-8, -1e-8, 1e-4, -1e-4], dtype=np.float64)
    name = "fresnel_cos_small_vals_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[3.14159], [-3.14159], [6.28318]], dtype=np.float32)
    name = "fresnel_cos_column_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[-100.0, 100.0, -1e3, 1e3]], dtype=np.float64)
    name = "fresnel_cos_large_vals_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.arange(-5, 6, dtype=np.float32)
    name = "fresnel_cos_arange_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.empty((0,), dtype=np.float64)
    name = "fresnel_cos_empty_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[[[0.0, 0.5], [1.5, -2.5]]]], dtype=np.float32)
    name = "fresnel_cos_4d_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    return list_of_inputs

generated_inputs["tf.math.special.fresnel_cos"] = tf_math_special_fresnel_cos_inputs()

import numpy as np
import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_unsorted_segment_max_inputs():
    list_of_inputs = []

    data = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [4, 3, 2, 1]], dtype=np.int32)
    segment_ids = np.array([0, 1, 0], dtype=np.int32)
    num_segments = 2
    name = "case1"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = (np.arange(3*2*4).reshape(3, 2, 4).astype(np.float32) - 5.0)
    segment_ids = np.array([0, 1, 0], dtype=np.int64)
    num_segments = 3
    name = "case2"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([10, -2, 7, 0, 5], dtype=np.int64)
    segment_ids = np.array([0, 1, -1, 1, 0], dtype=np.int32)
    num_segments = 2
    name = "case3"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.arange(2*3*2*2, dtype=np.float64).reshape(2, 3, 2, 2) * 0.5
    segment_ids = np.array([[0, 1, 2],
                            [3, 4, 0]], dtype=np.int64)
    num_segments = 5
    name = "case4"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([255, 0, 128, 64, 200, 1], dtype=np.uint8)
    segment_ids = np.array([0, 1, 1, 0, 1, 0], dtype=np.int32)
    num_segments = 3
    name = "case5"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = (np.arange(20).reshape(4, 5).astype(np.float16) - np.float16(10))
    segment_ids = np.array([0, -1, 1, 1], dtype=np.int64)
    num_segments = 3
    name = "case6"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = (np.arange(3*4*2) % 50).astype(np.int16).reshape(3, 4, 2)
    segment_ids = np.array([[0, 1, 2, 3],
                            [4, 5, 0, 1],
                            [2, 3, 4, 5]], dtype=np.int32)
    num_segments = 6
    name = "case7"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[-1, 2, -3],
                     [4, 5, -6],
                     [7, 8, 9],
                     [-10, 11, 12]], dtype=np.int8).reshape(2, 2, 3)
    segment_ids = np.array([[0, 1],
                            [2, 0]], dtype=np.int64)
    num_segments = 3
    name = "case8"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.arange(5, dtype=np.int32).reshape(5, 1)
    segment_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    num_segments = 6
    name = "case9"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.arange(2*3*1*2, dtype=np.int64).reshape(2, 3, 1, 2)
    segment_ids = np.array([[[0], [1], [2]],
                            [[3], [0], [1]]], dtype=np.int32)
    num_segments = 4
    name = "case10"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[-1e9, -2e9],
                     [3.5, -4.2],
                     [0.0, -7.7]], dtype=np.float64)
    segment_ids = np.array([2, 0, 2], dtype=np.int32)
    num_segments = 6
    name = "case11"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.unsorted_segment_max_1"] = tf_math_unsorted_segment_max_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_unsorted_segment_max_2_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([1, 3, -2, 5], dtype=np.int32)
    segment_ids = np.array([0, 1, 0, -1], dtype=np.int32)
    num_segments = np.array(3, dtype=np.int32)
    name = "case1_basic_int32_with_negative_id"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([[1.0, 2.5, -3.0, 4.5],
                     [5.5, -6.0, 7.2, 8.8],
                     [4.0, 3.3, 2.2, 1.1]], dtype=np.float32)
    segment_ids = np.array([0, 1, 0], dtype=np.int64)
    num_segments = np.array(2, dtype=np.int32)
    name = "rows_prefix_float32"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([[1, 200, 3],
                     [4, 5, 6]], dtype=np.uint8)
    segment_ids = np.array([[0, 1, -1],
                            [1, 0, 1]], dtype=np.int64)
    num_segments = np.array(2, dtype=np.int64)
    name = "full_shape_ids_uint8_with_negative"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([[[1.0, -2.0, 3.0, 4.0],
                      [5.0, 6.0, -7.0, 8.0],
                      [9.0, -10.0, 11.0, 12.0]],
                     [[-1.0, 2.0, -3.0, -4.0],
                      [-5.0, -6.0, 7.0, -8.0],
                      [-9.0, 10.0, -11.0, -12.0]]], dtype=np.float64)
    segment_ids = np.array([1, 0], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int64)
    name = "3d_firstdim_float64"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = np.array([[[10, -20],
                      [30, -40]],
                     [[-50, 60],
                      [70, -80]],
                     [[90, -100],
                      [110, -120]]], dtype=np.int16)
    segment_ids = np.array([[0, 1],
                            [1, 2],
                            [2, 0]], dtype=np.int32)
    num_segments = np.array(3, dtype=np.int32)
    name = "2d_prefix_int16"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array([[[-1, 2],
                      [3, -4]],
                     [[5, -6],
                      [-7, 8]]], dtype=np.int8)
    segment_ids = np.array([[[0, 1],
                             [1, 0]],
                            [[1, 0],
                             [0, 1]]], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    name = "elemwise_ids_int8"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = np.array([65530, 1, 500, 60000, 42, 0], dtype=np.int32)
    segment_ids = np.array([2, 2, 7, 2, 7, 7], dtype=np.int32)
    num_segments = np.array(10, dtype=np.int32)
    name = "int32_sparse_segments_with_gaps"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    data = np.array([[[[1.0, -2.0],
                       [3.0, -4.0]],
                      [[-5.0, 6.0],
                       [-7.0, 8.0]]],
                     [[[9.0, -10.0],
                       [11.0, -12.0]],
                      [[-13.0, 14.0],
                       [-15.0, 16.0]]]], dtype=np.float16)
    segment_ids = np.array([0, 1], dtype=np.int64)
    num_segments = np.array(2, dtype=np.int32)
    name = "float16_4d_firstdim_prefix"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    data = np.array([-10, 0, 10], dtype=np.int64)
    segment_ids = np.array([-1, -1, -1], dtype=np.int64)
    num_segments = np.array(3, dtype=np.int64)
    name = "all_dropped_int64"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    data = np.array([[0.1, -0.2, 0.3],
                     [1.5, -1.0, 2.0],
                     [-0.5, 0.7, -0.9],
                     [3.2, -3.3, 3.4]], dtype=np.float64)
    segment_ids = np.array([0, 2, 1, 2], dtype=np.int32)
    num_segments = np.array(4, dtype=np.int32)
    name = "float64_2d_rows_prefix"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    data = np.array([[[4000000000, 123],
                      [987654321, 2222222222],
                      [3333333333, 4294967295]],
                     [[111111111, 222222222],
                      [333333333, 444444444],
                      [555555555, 666666666]]], dtype=np.int64)
    segment_ids = np.array([[0, 4, 1],
                            [2, 4, 0]], dtype=np.int32)
    num_segments = np.array(5, dtype=np.int64)
    name = "int64_3d_2d_prefix_with_gaps"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    data = np.array([[2**62], [2**40], [2**32]], dtype=np.int64)
    segment_ids = np.array([[0], [0], [1]], dtype=np.int64)
    num_segments = np.array(3, dtype=np.int32)
    name = "int64_2d_fullshape_ids"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.unsorted_segment_max_2"] = tf_math_unsorted_segment_max_2_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_math_zero_fraction_inputs():
    list_of_inputs = []

    value = np.array([0.0, 1.0, -2.5, 0.0], dtype=np.float32)
    name = "basic_float_vector"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array([[0, -1, 2], [3, 0, 0]], dtype=np.int32)
    name = "int32_matrix"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.zeros((2, 3, 4), dtype=np.float16)
    value[0, 1, 2] = -1.5
    value[1, 2, 3] = 0.5
    name = "float16_3d_tensor"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array(0, dtype=np.int64)
    name = "scalar_zero_int64"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array(3.14159, dtype=np.float64)
    name = "scalar_nonzero_float64"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array([], dtype=np.float32)
    name = "empty_vector_float32"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.zeros((2, 2, 2, 3), dtype=np.int16)
    value[0, 0, 0, 0] = 10
    value[1, 1, 1, 2] = -5
    name = "int16_4d_tensor"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array([0+0j, 1+0j, 0+2j, 3+4j, 0+0j], dtype=np.complex64)
    name = "complex64_vector"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array([[0+0j, 0+0j], [5+0j, 0+1j]], dtype=np.complex128)
    name = "complex128_matrix"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array([0, 255, 0, 128, 1, 0], dtype=np.uint8)
    name = "uint8_vector"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array([-128, 0, 127, 0, -1], dtype=np.int8)
    name = "int8_vector"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array([0.0, np.nan, np.inf, -np.inf, 1.0, 0.0], dtype=np.float32)
    name = "float_with_nan_inf"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.zeros((0, 5), dtype=np.int64)
    name = "empty_2d_int64"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.zero_fraction"] = tf_math_zero_fraction_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_nest_assert_same_structure_inputs():
    list_of_inputs = []

    nest1 = [np.int32(1), np.int32(2), np.int32(3)]
    nest2 = [np.int64(9), np.int16(-2), np.int8(0)]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": True, "expand_composites": False}))

    nest1 = [np.array([[1.0, -2.5], [3.3, 4.4]], dtype=np.float32),
             np.array([[5.5, 6.6], [7.7, 8.8]], dtype=np.float32)]
    nest2 = [np.array([[0.0, 2.5], [-3.3, -4.4]], dtype=np.float64),
             np.array([[9.9, -6.6], [7.7, -8.8]], dtype=np.float64)]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": True, "expand_composites": False}))

    nest1 = [[np.int32(1), np.int32(2)], [np.int32(3), np.int32(4)]]
    nest2 = [[np.int64(10), np.int64(20)], [np.int64(30), np.int64(40)]]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": False, "expand_composites": False}))

    nest1 = [np.bool_(True), np.bool_(False), np.bool_(True), np.bool_(False)]
    nest2 = [np.bool_(False), np.bool_(True), np.bool_(False), np.bool_(True)]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": True, "expand_composites": False}))

    nest1 = [np.array([1.0, -2.0, 3.5], dtype=np.float32),
             np.array([-4.5, 5.0, -6.1], dtype=np.float32)]
    nest2 = [np.array([0.0, 2.0, -3.5], dtype=np.float32),
             np.array([4.5, -5.0, 6.1], dtype=np.float32)]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": False, "expand_composites": True}))

    nest1 = [[np.bool_(True), np.bool_(True), np.bool_(False)],
             [np.bool_(False), np.bool_(True), np.bool_(False)]]
    nest2 = [[np.bool_(False), np.bool_(False), np.bool_(True)],
             [np.bool_(True), np.bool_(False), np.bool_(True)]]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": True, "expand_composites": True}))

    nest1 = [np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.int32),
             np.array([[[7, 8, 9]], [[10, 11, 12]]], dtype=np.int32)]
    nest2 = [np.array([[[0, -2, -3]], [[-4, -5, -6]]], dtype=np.int64),
             np.array([[[7, -8, 9]], [[-10, 11, -12]]], dtype=np.int64)]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": False, "expand_composites": False}))

    nest1 = [[np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)],
             [np.array([5, 6], dtype=np.int32), np.array([7, 8], dtype=np.int32)]]
    nest2 = [[np.array([-1, -2], dtype=np.int32), np.array([-3, -4], dtype=np.int32)],
             [np.array([-5, -6], dtype=np.int32), np.array([-7, -8], dtype=np.int32)]]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": True, "expand_composites": False}))

    nest1 = [np.array([10, 20, 30], dtype=np.int16),
             np.array([40, 50, 60], dtype=np.int16),
             np.array([70, 80, 90], dtype=np.int16)]
    nest2 = [np.array([-10, -20, -30], dtype=np.int32),
             np.array([-40, -50, -60], dtype=np.int32),
             np.array([-70, -80, -90], dtype=np.int32)]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": False, "expand_composites": True}))

    nest1 = [np.float64(np.nan), np.float64(np.inf), np.float64(-np.inf)]
    nest2 = [np.float32(0.0), np.float32(-1e9), np.float32(1e9)]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": True, "expand_composites": False}))

    nest1 = [np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]], dtype=np.int32),
             np.array([[16, 15, 14, 13],
                       [12, 11, 10, 9],
                       [8, 7, 6, 5],
                       [4, 3, 2, 1]], dtype=np.int32)]
    nest2 = [np.array([[0, -2, -3, -4],
                       [-5, -6, -7, -8],
                       [-9, -10, -11, -12],
                       [-13, -14, -15, -16]], dtype=np.int64),
             np.array([[6, 5, 4, 3],
                       [2, 1, 0, -1],
                       [-2, -3, -4, -5],
                       [-6, -7, -8, -9]], dtype=np.int64)]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": True, "expand_composites": True}))

    nest1 = [[np.float32(1.1), np.float32(-2.2)],
             [np.float32(3.3), np.float32(-4.4)],
             [np.float32(5.5), np.float32(-6.6)]]
    nest2 = [[np.float64(-1.1), np.float64(2.2)],
             [np.float64(-3.3), np.float64(4.4)],
             [np.float64(-5.5), np.float64(6.6)]]
    list_of_inputs.append(copy.deepcopy({"nest1": nest1, "nest2": nest2, "check_types": False, "expand_composites": False}))

    return list_of_inputs

generated_inputs["tf.nest.assert_same_structure"] = tf_nest_assert_same_structure_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_compute_accidental_hits_inputs():
    list_of_inputs = []
    
    true_classes = np.array([[3], [4]], dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3, 5], dtype=np.int64)
    num_true = 1
    seed = 0
    name = "case1_simple_match"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[10, 11], [12, 13], [14, 15]], dtype=np.int64)
    sampled_candidates = np.array([7, 8, 9, 10, 15], dtype=np.int64)
    num_true = 2
    seed = 42
    name = "case2_multi_row_multi_true"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[0, 5, 7]], dtype=np.int64)
    sampled_candidates = np.array([2, 4, 6], dtype=np.int64)
    num_true = 3
    seed = 123
    name = "case3_no_hits"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[0], [1], [2], [3]], dtype=np.int64)
    sampled_candidates = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    num_true = 1
    seed = 999
    name = "case4_all_hit"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[101, 102, 103], [201, 202, 203]], dtype=np.int64)
    sampled_candidates = np.array([0, 102, 200, 202, 300], dtype=np.int64)
    num_true = 3
    seed = 7
    name = "case5_sparse_hits"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[5, 6], [7, 8], [9, 10], [11, 12], [13, 14]], dtype=np.int64)
    sampled_candidates = np.array([1, 3, 5, 8, 10, 12, 14, 16, 18, 20], dtype=np.int64)
    num_true = 2
    seed = 31415
    name = "case6_larger_batch"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[2147483700]], dtype=np.int64)
    sampled_candidates = np.array([2147483700, 9223372000000000000 // 2], dtype=np.int64)
    num_true = 1
    seed = 1
    name = "case7_big_ids"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[2, 4, 6, 8], [1, 3, 5, 7], [0, 9, 10, 11]], dtype=np.int64)
    sampled_candidates = np.array([6, 7, 8, 12, 13, 14], dtype=np.int64)
    num_true = 4
    seed = 12345
    name = "case8_num_true_four"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[100, 200], [300, 400]], dtype=np.int64)
    sampled_candidates = np.array([200, 500, 600, 1000], dtype=np.int64)
    num_true = 2
    seed = 0
    name = "case9_some_hits_with_zero_seed"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    sampled_candidates = np.array([], dtype=np.int64)
    num_true = 2
    seed = 77
    name = "case10_empty_sampled"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.empty((0, 3), dtype=np.int64)
    sampled_candidates = np.array([1, 2, 3], dtype=np.int64)
    num_true = 3
    seed = 222
    name = "case11_empty_batch"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    true_classes = np.array([[9, 10, 11, 12, 13]], dtype=np.int64)
    sampled_candidates = np.array([0, 2, 4, 6, 8, 10, 12, 14], dtype=np.int64)
    num_true = 5
    seed = 555
    name = "case12_single_row_many_true"
    input_dict = {
        "true_classes": true_classes,
        "sampled_candidates": sampled_candidates,
        "num_true": num_true,
        "seed": seed,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.compute_accidental_hits"] = tf_nn_compute_accidental_hits_inputs()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(42)

def tf_nn_ctc_beam_search_decoder_inputs():
    list_of_inputs = []

    # Input 1
    inputs = np.random.randn(4, 2, 3).astype(np.float32)
    sequence_length = np.array([4, 3], dtype=np.int32)
    beam_width = 3
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 2
    inputs = np.array([[[2.0, -1.0]]], dtype=np.float32)  # shape (1,1,2)
    sequence_length = np.array([1], dtype=np.int32)
    beam_width = 1
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 3
    inputs = (np.random.randn(6, 3, 5) * 2.0).astype(np.float32)
    sequence_length = np.array([6, 5, 4], dtype=np.int32)
    beam_width = 5
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 4
    inputs = np.random.uniform(-3, 3, size=(3, 3, 4)).astype(np.float32)
    sequence_length = np.array([3, 2, 2], dtype=np.int32)
    beam_width = 2
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 5
    inputs = (np.random.randn(8, 2, 7) * 5.0 - 2.0).astype(np.float32)
    sequence_length = np.array([5, 8], dtype=np.int32)
    beam_width = 10
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 6
    inputs = (np.random.randn(10, 1, 10)).astype(np.float64)
    sequence_length = np.array([7], dtype=np.int32)
    beam_width = 50
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 7
    inputs = np.random.randn(2, 4, 3).astype(np.float32)
    sequence_length = np.array([2, 1, 2, 1], dtype=np.int32)
    beam_width = 2
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 8
    inputs = (np.random.randn(5, 5, 6) * 0.1).astype(np.float32)
    sequence_length = np.array([5, 5, 5, 5, 5], dtype=np.int32)
    beam_width = 3
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 9
    inputs = np.array([
        [[3.0, -1.0], [1.0, 0.0]],
        [[-2.0, 2.0], [0.5, -0.5]],
        [[0.1, -0.1], [1.5, -1.5]],
        [[-0.3, 0.3], [2.0, -2.0]],
    ], dtype=np.float32)  # shape (4,2,2)
    sequence_length = np.array([2, 4], dtype=np.int32)
    beam_width = 2
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    # Input 10
    inputs = (np.random.randn(7, 3, 9)).astype(np.float64)
    sequence_length = np.array([7, 3, 6], dtype=np.int32)
    beam_width = 7
    top_paths = 1
    list_of_inputs.append(copy.deepcopy({
        "inputs": inputs,
        "sequence_length": sequence_length,
        "beam_width": beam_width,
        "top_paths": top_paths
    }))

    return list_of_inputs

generated_inputs["tf.nn.ctc_beam_search_decoder"] = tf_nn_ctc_beam_search_decoder_inputs()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(123)

def tf_nn_ctc_loss_inputs():
    def actual_blank_index(num_labels, blank_index):
        return num_labels + blank_index if blank_index is not None and blank_index < 0 else blank_index

    def generate_labels(batch_size, max_label_len, num_labels, label_len_arr, blank_index, out_dtype=np.int64):
        bidx = actual_blank_index(num_labels, blank_index)
        allowed_vals = [i for i in range(num_labels) if i != bidx]
        if len(allowed_vals) == 0:
            allowed_vals = [0]
        labels = np.zeros((batch_size, max_label_len), dtype=out_dtype)
        for b in range(batch_size):
            L = int(label_len_arr[b])
            if L > 0:
                labels[b, :L] = np.random.choice(allowed_vals, size=L)
        return labels

    list_of_inputs = []

    # Input 1
    B, Lmax, T, C = 4, 5, 12, 6
    label_length = np.random.randint(1, Lmax + 1, size=(B,), dtype=np.int64)
    logit_length = np.full((B,), T, dtype=np.int64)
    logits_time_major = True
    blank_index = 0
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int64)
    logits = np.random.randn(T, B, C).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case1"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 2
    B, Lmax, T, C = 2, 4, 7, 5
    label_length = np.random.randint(1, Lmax + 1, size=(B,), dtype=np.int32)
    logit_length = np.array([7, 5], dtype=np.int32)
    logits_time_major = False
    blank_index = -1
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int32)
    logits = np.random.randn(B, T, C).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case2"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 3
    B, Lmax, T, C = 4, 6, 10, 7
    label_length = np.random.randint(1, Lmax + 1, size=(B,), dtype=np.int64)
    logit_length = np.array([10, 9, 8, 10], dtype=np.int64)
    logits_time_major = True
    blank_index = C - 1
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int64)
    logits = np.random.randn(T, B, C).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case3"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 4
    B, Lmax, T, C = 3, 4, 6, 5
    label_length = np.array([0, 2, 3], dtype=np.int32)
    logit_length = np.array([6, 6, 5], dtype=np.int32)
    logits_time_major = True
    blank_index = 0
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int32)
    logits = np.random.uniform(-1.0, 1.0, size=(T, B, C)).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case4_zero_label_len"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 5
    B, Lmax, T, C = 1, 3, 4, 4
    label_length = np.array([1], dtype=np.int64)
    logit_length = np.array([4], dtype=np.int64)
    logits_time_major = False
    blank_index = -1
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int64)
    logits = (np.random.randn(B, T, C) * 0.5).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case5_single_sample"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 6
    B, Lmax, T, C = 5, 8, 12, 10
    label_length = np.random.randint(1, Lmax + 1, size=(B,), dtype=np.int32)
    logit_length = np.maximum(label_length + 1, np.random.randint(Lmax, T + 1, size=(B,), dtype=np.int32))
    logits_time_major = True
    blank_index = -2
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int32)
    logits = np.random.randn(T, B, C).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case6_negative_blank_two"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 7
    B, Lmax, T, C = 2, 2, 4, 3
    label_length = np.array([2, 1], dtype=np.int64)
    logit_length = np.array([4, 3], dtype=np.int64)
    logits_time_major = False
    blank_index = 0
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int64)
    logits = np.random.uniform(-2.0, 2.0, size=(B, T, C)).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case7_small_dims"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 8
    B, Lmax, T, C = 6, 7, 15, 8
    label_length = np.random.randint(1, Lmax + 1, size=(B,), dtype=np.int32)
    logit_length = np.random.randint(Lmax, T + 1, size=(B,), dtype=np.int32)
    logits_time_major = True
    blank_index = 0
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int32)
    logits = (np.random.randn(T, B, C) * 1.5).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case8_large_batch"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 9
    B, Lmax, T, C = 2, 3, 5, 4
    label_length = np.array([3, 2], dtype=np.int32)
    logit_length = np.array([5, 5], dtype=np.int32)
    logits_time_major = False
    blank_index = 0
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int32)
    logits = np.random.randn(B, T, C).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case9_float32_logits"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    # Input 10
    B, Lmax, T, C = 3, 5, 9, 5
    label_length = np.random.randint(1, Lmax + 1, size=(B,), dtype=np.int64)
    logit_length = np.array([9, 8, 7], dtype=np.int64)
    logits_time_major = True
    blank_index = 2
    labels = generate_labels(B, Lmax, C, label_length, blank_index, out_dtype=np.int64)
    logits = np.random.randn(T, B, C).astype(np.float32)
    unique = tuple()
    name = "ctc_loss_case10_mid_blank"
    list_of_inputs.append(copy.deepcopy({
        "labels": labels,
        "logits": logits,
        "label_length": label_length,
        "logit_length": logit_length,
        "logits_time_major": logits_time_major,
        "unique": unique,
        "blank_index": blank_index,
        "name": name
    }))

    return list_of_inputs

generated_inputs["tf.nn.ctc_loss"] = tf_nn_ctc_loss_inputs()

import numpy as np
import tensorflow as tf
import copy

np.random.seed(42)

def tf_nn_log_poisson_loss_inputs():
    list_of_inputs = []

    targets = np.array(3.0, dtype=np.float32)
    log_input = np.array(np.log(2.5), dtype=np.float32)
    compute_full_loss = False
    name = "scalar_f32_no_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    log_input = np.array([np.log(0.5), np.log(1.5), np.log(3.0)], dtype=np.float64)
    compute_full_loss = False
    name = "vector_f64_no_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    log_input = np.array([[0.0, -0.7], [1.2, 2.0]], dtype=np.float32)
    compute_full_loss = True
    name = "matrix2x2_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = (np.random.rand(2, 3, 4) * 3.0).astype(np.float64)
    log_input = np.random.randn(2, 3, 4).astype(np.float64)
    compute_full_loss = False
    name = "tensor3d_f64_no_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = (np.random.rand(1, 2, 2, 3).astype(np.float32) + 0.1)
    log_input = np.random.randn(1, 2, 2, 3).astype(np.float32)
    compute_full_loss = True
    name = "tensor4d_f32_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([0.0, 0.0, 1.0, 2.0, 5.0], dtype=np.float32)
    log_input = np.array([-2.0, 0.0, 0.5, 1.0, 2.3], dtype=np.float32)
    compute_full_loss = False
    name = "vector_f32_zeros_no_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([[0.1], [2.0]], dtype=np.float64)
    log_input = np.array([[np.log(0.3)], [np.log(5.0)]], dtype=np.float64)
    compute_full_loss = True
    name = "column_f64_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([], dtype=np.float32)
    log_input = np.array([], dtype=np.float32)
    compute_full_loss = False
    name = "empty_f32_no_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    rates = np.array([[0.8, 1.2, 2.5], [0.3, 4.0, 0.1], [1.5, 2.2, 3.3]], dtype=np.float64)
    targets = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=np.float64)
    log_input = np.log(rates)
    compute_full_loss = False
    name = "matrix3x3_f64_no_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([50.0, 1000.0, 100000.0], dtype=np.float64)
    log_input = np.log(np.array([60.0, 900.0, 100000.0], dtype=np.float64))
    compute_full_loss = True
    name = "large_counts_f64_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array([-1.0, 0.0], dtype=np.float32)
    log_input = np.array([np.log(1.0), np.log(2.0)], dtype=np.float32)
    compute_full_loss = False
    name = "neg_target_f32_no_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    targets = np.array(0.7, dtype=np.float64)
    log_input = np.array(-0.3, dtype=np.float64)
    compute_full_loss = True
    name = "scalar_f64_full"
    input_dict = {"targets": targets, "log_input": log_input, "compute_full_loss": compute_full_loss, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.log_poisson_loss"] = tf_nn_log_poisson_loss_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_softsign_inputs():
    list_of_inputs = []
    rng = np.random.default_rng(42)

    features = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    name = "softsign_case_01"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([[-10.0, -0.5, 0.0], [0.5, 2.5, 100.0]], dtype=np.float64)
    name = "softsign_case_02"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.linspace(-5, 5, num=24, dtype=np.float16).reshape(2, 3, 4)
    name = "softsign_case_3d_float16"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array(3.14159, dtype=np.float32)
    name = "softsign_case_scalar"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = rng.normal(size=(2, 1, 3, 2)).astype(np.float32)
    name = "softsign_case_4d_random_float32"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([], dtype=np.float64)
    name = "softsign_case_empty"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([1e-12, -1e-12, 1e6, -1e6, 1e12, -1e12], dtype=np.float64)
    name = "softsign_case_extremes_float64"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([np.inf, -np.inf, np.nan, 0.0, 1.0, -1.0], dtype=np.float32)
    name = "softsign_case_inf_nan"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    base = np.arange(-20, 20, dtype=np.float16)
    features = base[::3]
    name = "softsign_case_noncontiguous_slice"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([[-3.0], [0.0], [2.5], [10.0], [-7.7]], dtype=np.float32)
    name = "softsign_case_column_vector"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.array([[-2.0, 0.5, 4.0, -9.0, 1e-3]], dtype=np.float64)
    name = "softsign_case_row_vector_float64"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    features = np.asfortranarray(np.linspace(-6, 6, 12, dtype=np.float32).reshape(3, 4, order="F"))
    name = "softsign_case_fortran_order"
    list_of_inputs.append(copy.deepcopy({"features": features, "name": name}))

    return list_of_inputs

generated_inputs["tf.nn.softsign"] = tf_nn_softsign_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_quantization_fake_quant_with_min_max_vars_per_channel_gradient_inputs():
    list_of_inputs = []

    # Input 1: 1D
    gradients = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    inputs = np.array([-0.2, 0.0, 1.1], dtype=np.float32)
    min_arr = np.array([-1.0, -0.5, 0.0], dtype=np.float32)
    max_arr = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 8,
        "narrow_range": False,
        "name": "case1_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D (b=2, d=4)
    gradients = np.array([[0.1, -0.2, 0.3, -0.4],
                          [1.0, -1.0, 0.5, -0.5]], dtype=np.float32)
    inputs = np.array([[0.05, -0.25, 0.35, -0.45],
                       [0.9, -1.2, 0.6, -0.55]], dtype=np.float32)
    min_arr = np.array([-0.5, -0.3, -0.2, -0.1], dtype=np.float32)
    max_arr = np.array([0.5, 0.7, 0.8, 0.9], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 7,
        "narrow_range": False,
        "name": "case2_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D (b=1, h=2, w=2, d=3)
    gradients = np.array([[[[0.1, -0.2, 0.3],
                            [0.2, -0.1, 0.0]],
                           [[-0.3, 0.4, -0.5],
                            [0.6, -0.7, 0.8]]]], dtype=np.float32)
    inputs = np.array([[[[0.05, -0.1, 0.25],
                         [0.15, -0.05, 0.05]],
                        [[-0.25, 0.35, -0.45],
                         [0.55, -0.65, 0.75]]]], dtype=np.float32)
    min_arr = np.array([-0.6, -0.4, -0.2], dtype=np.float32)
    max_arr = np.array([0.6, 0.8, 0.9], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 8,
        "narrow_range": True,
        "name": "case3_4d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D (b=4, h=3, w=3, d=1)
    gradients = np.random.uniform(-1, 1, size=(4, 3, 3, 1)).astype(np.float32)
    inputs = np.random.uniform(-2, 2, size=(4, 3, 3, 1)).astype(np.float32)
    min_arr = np.array([-1.0], dtype=np.float32)
    max_arr = np.array([1.0], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 9,
        "narrow_range": False,
        "name": "case4_4d_d1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D (b=5, d=2) narrow_range True
    gradients = np.array([[0.2, -0.2],
                          [0.3, -0.1],
                          [-0.4, 0.4],
                          [0.0, 0.0],
                          [1.0, -1.0]], dtype=np.float32)
    inputs = np.array([[0.25, -0.25],
                       [0.35, -0.15],
                       [-0.45, 0.45],
                       [0.05, -0.05],
                       [1.2, -1.2]], dtype=np.float32)
    min_arr = np.array([-0.3, -0.6], dtype=np.float32)
    max_arr = np.array([0.7, 0.9], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 8,
        "narrow_range": True,
        "name": "case5_2d_narrow"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D (d=1)
    gradients = np.array([2.5], dtype=np.float32)
    inputs = np.array([-10.0], dtype=np.float32)
    min_arr = np.array([-6.0], dtype=np.float32)
    max_arr = np.array([6.0], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 12,
        "narrow_range": False,
        "name": "case6_1d_single"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D (b=2, h=1, w=5, d=5)
    gradients = np.linspace(-0.5, 0.5, num=2*1*5*5).reshape(2, 1, 5, 5).astype(np.float32)
    inputs = np.linspace(-1.0, 1.0, num=2*1*5*5).reshape(2, 1, 5, 5).astype(np.float32)
    min_arr = np.array([-0.9, -0.5, -0.1, 0.0, 0.2], dtype=np.float32)
    max_arr = np.array([0.9, 0.6, 0.4, 0.7, 1.0], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 6,
        "narrow_range": False,
        "name": "case7_4d_wide_d5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D (b=1, d=8), num_bits=16
    gradients = np.array([[0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4]], dtype=np.float32)
    inputs = np.array([[0.05, -0.15, 0.25, -0.35, 0.45, -0.55, 0.65, -0.75]], dtype=np.float32)
    min_arr = np.array([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2], dtype=np.float32)
    max_arr = np.array([0.8, 0.7, 0.9, 1.0, 1.2, 0.6, 0.5, 0.4], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 16,
        "narrow_range": False,
        "name": "case8_2d_16bits"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D (d=7), fractional min/max
    gradients = np.array([-0.05, 0.1, -0.15, 0.2, -0.25, 0.3, -0.35], dtype=np.float32)
    inputs = np.array([0.01, -0.12, 0.23, -0.34, 0.45, -0.56, 0.67], dtype=np.float32)
    min_arr = np.array([-0.25, -0.2, -0.15, -0.1, -0.05, -0.02, -0.01], dtype=np.float32)
    max_arr = np.array([0.3, 0.25, 0.35, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 10,
        "narrow_range": True,
        "name": "case9_1d_frac"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D (b=3, h=2, w=2, d=2), num_bits=2
    gradients = np.random.randn(3, 2, 2, 2).astype(np.float32)
    inputs = np.random.uniform(-0.8, 0.8, size=(3, 2, 2, 2)).astype(np.float32)
    min_arr = np.array([-0.7, -0.4], dtype=np.float32)
    max_arr = np.array([0.7, 0.9], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 2,
        "narrow_range": True,
        "name": "case10_4d_lowbits"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D (b=3, d=3), negative ranges
    gradients = np.array([[1.0, -1.0, 0.5],
                          [-0.3, 0.2, -0.1],
                          [0.0, 0.0, 0.0]], dtype=np.float32)
    inputs = np.array([[-1.5, -0.8, -0.2],
                       [-0.6, -0.4, -0.05],
                       [-0.9, -0.1, -0.3]], dtype=np.float32)
    min_arr = np.array([-2.0, -1.0, -0.5], dtype=np.float32)
    max_arr = np.array([-0.5, -0.2, -0.1], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 5,
        "narrow_range": False,
        "name": "case11_2d_negative_range"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 4D (b=1, h=4, w=1, d=4), mixed values
    gradients = np.array([[[[0.05, -0.05, 0.1, -0.1]],
                           [[0.2, -0.2, 0.3, -0.3]],
                           [[-0.4, 0.4, -0.5, 0.5]],
                           [[0.6, -0.6, 0.7, -0.7]]]], dtype=np.float32)
    inputs = np.array([[[[0.01, -0.02, 0.03, -0.04]],
                        [[0.25, -0.15, 0.35, -0.45]],
                        [[-0.55, 0.65, -0.75, 0.85]],
                        [[0.95, -1.05, 1.15, -1.25]]]], dtype=np.float32)
    min_arr = np.array([-0.3, -0.2, -0.1, -0.05], dtype=np.float32)
    max_arr = np.array([0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_arr,
        "max": max_arr,
        "num_bits": 11,
        "narrow_range": False,
        "name": "case12_4d_varied"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient"] = tf_quantization_fake_quant_with_min_max_vars_per_channel_gradient_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_random_stateless_gamma_inputs():
    list_of_inputs = []

    shape = np.array([10, 2], dtype=np.int32)
    seed = np.array([12, 34], dtype=np.int64)
    alpha = np.array([0.5, 1.5], dtype=np.float32)
    beta = np.array(1.0, dtype=np.float32)
    dtype = np.float32
    name = "case1_basic_vec"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([7, 5, 2], dtype=np.int32)
    seed = np.array([123, 456], dtype=np.int32)
    alpha = np.array([0.5, 1.5], dtype=np.float64)
    beta = np.array(2.0, dtype=np.float64)
    dtype = np.float64
    name = "case2_broadcast_lastdim"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([30, 3, 2], dtype=np.int32)
    seed = np.array([98765, 43210], dtype=np.int64)
    alpha = np.array([[1.0], [3.0], [5.0]], dtype=np.float32)
    beta = np.array([[3.0, 4.0]], dtype=np.float32)
    dtype = np.float32
    name = "case3_matrix_broadcast"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([100], dtype=np.int32)
    seed = np.array([1, 0], dtype=np.int32)
    alpha = np.array(2.0, dtype=np.float64)
    beta = np.array(0.5, dtype=np.float64)
    dtype = np.float64
    name = "case4_scalar_params"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([6, 8, 4, 3], dtype=np.int32)
    seed = np.array([314159, 265358], dtype=np.int64)
    alpha = np.linspace(0.1, 2.5, num=8 * 4 * 3, dtype=np.float32).reshape(8, 4, 3)
    beta = np.array(1.0, dtype=np.float32)
    dtype = np.float32
    name = "case5_highrank_alpha"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([4, 3, 2, 5], dtype=np.int32)
    seed = np.array([42, 99], dtype=np.int32)
    alpha = np.array([[1.0, 2.0, 1.5, 0.7, 3.0],
                      [2.5, 1.2, 0.8, 2.2, 1.1]], dtype=np.float32)
    beta = np.array([[1.0, 2.0, 1.0, 0.5, 3.0]], dtype=np.float32)
    dtype = np.float32
    name = "case6_partial_broadcast"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([2, 3, 4], dtype=np.int32)
    seed = np.array([7, 13], dtype=np.int64)
    alpha = np.array([[0.8], [1.2], [2.5]], dtype=np.float32)
    beta = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    dtype = np.float32
    name = "case7_mixed_rank_broadcast"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([1000, 1], dtype=np.int64)
    seed = np.array([0, 1], dtype=np.int32)
    alpha = np.array([0.2], dtype=np.float32)
    beta = np.array([50.0], dtype=np.float32)
    dtype = np.float32
    name = "case8_small_alpha_large_beta"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([5, 2], dtype=np.int32)
    seed = np.array([2147483646, 2147483645], dtype=np.int64)
    alpha = np.array([2.0, 3.0], dtype=np.float16)
    beta = np.array([0.5, 2.0], dtype=np.float16)
    dtype = np.float16
    name = "case9_float16_dtype"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([2, 2, 2], dtype=np.int32)
    seed = np.array([888, 777], dtype=np.int32)
    alpha = np.array([[1.0, 2.0],
                      [3.0, 4.0]], dtype=np.float32)
    beta = np.array([[1.0],
                     [0.5]], dtype=np.float32)
    dtype = np.float32
    name = "case10_beta_needs_broadcast"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([9, 3], dtype=np.int64)
    seed = np.array([2025, 1108], dtype=np.int64)
    alpha = np.array([0.7, 1.3, 2.1], dtype=np.float64)
    beta = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    dtype = np.float64
    name = "case11_int64_shape_and_seed"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([3, 4, 1], dtype=np.int32)
    seed = np.array([12, 34], dtype=np.int32)
    alpha = np.array([[0.1],
                      [0.2],
                      [0.3],
                      [0.4]], dtype=np.float32)
    beta = np.array(10.0, dtype=np.float32)
    dtype = np.float32
    name = "case12_trailing_two_dims"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_gamma_1"] = tf_random_stateless_gamma_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_gamma_inputs():
    list_of_inputs = []

    # Input 1
    shape = [np.int32(10), np.int32(2)]
    seed = [np.int32(12), np.int32(34)]
    alpha = [np.float32(0.5), np.float32(1.5)]
    beta = [np.float32(1.0), np.float32(1.0)]
    dtype = np.float32
    name = "gamma_case_1"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = [np.int64(7), np.int64(5), np.int64(2)]
    seed = [np.int64(123), np.int64(456)]
    alpha = [np.float32(0.5), np.float32(1.5)]
    beta = [np.float32(1.0)]
    dtype = np.float32
    name = "gamma_case_2"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = [np.int32(30), np.int32(3), np.int32(2)]
    seed = [np.int32(12), np.int32(34)]
    alpha = [[np.float64(1.0)], [np.float64(3.0)], [np.float64(5.0)]]
    beta = [[np.float64(3.0), np.float64(4.0)]]
    dtype = np.float64
    name = "gamma_case_3"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = [np.int32(4), np.int32(3), np.int32(1)]
    seed = [np.int32(9876), np.int32(5432)]
    alpha = [[np.float32(0.7)], [np.float32(2.3)], [np.float32(5.1)]]
    beta = [np.float32(2.0)]
    dtype = np.float32
    name = "gamma_case_4"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = [np.int32(5), np.int32(2), np.int32(1), np.int32(3)]
    seed = [np.int32(42), np.int32(24)]
    alpha = [
        [[np.float32(0.5), np.float32(1.0), np.float32(1.5)]],
        [[np.float32(2.0), np.float32(2.5), np.float32(3.0)]]
    ]  # shape (2,1,3)
    beta = [np.float32(0.8), np.float32(1.2), np.float32(2.0)]  # shape (3,)
    dtype = np.float32
    name = "gamma_case_5"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    shape = [np.int32(8), np.int32(1)]
    seed = [np.int32(7), np.int32(11)]
    alpha = [np.float32(2.0)]
    beta = [np.float32(0.5)]
    dtype = np.float32
    name = "gamma_case_6"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    shape = [np.int32(2), np.int32(4), np.int32(2)]
    seed = [np.int32(111), np.int32(222)]
    alpha = [np.float16(0.2), np.float16(5.0)]
    beta = [np.float16(10.0), np.float16(0.8)]
    dtype = np.float16
    name = "gamma_case_7"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    shape = [np.int64(4), np.int64(3), np.int64(2)]
    seed = [np.int64(13579), np.int64(24680)]
    alpha = [
        [np.float64(1.1), np.float64(2.2)],
        [np.float64(3.3), np.float64(4.4)],
        [np.float64(5.5), np.float64(6.6)]
    ]  # shape (3,2)
    beta = [
        [np.float64(0.5)],
        [np.float64(1.5)],
        [np.float64(2.5)]
    ]  # shape (3,1)
    dtype = np.float64
    name = "gamma_case_8"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    shape = [np.int32(2), np.int32(2)]
    seed = [np.int32(333), np.int32(444)]
    alpha = [
        [np.float32(0.9)],
        [np.float32(1.8)]
    ]  # shape (2,1)
    beta = [
        [np.float32(1.0), np.float32(2.0)]
    ]  # shape (1,2)
    dtype = np.float32
    name = "gamma_case_9"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    shape = [np.int32(2), np.int32(2), np.int32(2), np.int32(1)]
    seed = [np.int32(555), np.int32(666)]
    alpha = [np.float32(3.0)]
    beta = [np.float32(4.0)]
    dtype = np.float32
    name = "gamma_case_10"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    shape = [np.int32(3), np.int32(1)]
    seed = [np.int32(777), np.int32(888)]
    alpha = [np.float64(0.05)]
    beta = [np.float64(50.0)]
    dtype = np.float64
    name = "gamma_case_11"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    shape = [np.int32(9), np.int32(4)]
    seed = [np.int32(1010), np.int32(2020)]
    alpha = [np.float32(1.0), np.float32(2.0), np.float32(3.0), np.float32(4.0)]  # shape (4,)
    beta = [np.float32(0.5), np.float32(1.5), np.float32(2.5), np.float32(3.5)]  # shape (4,)
    dtype = np.float32
    name = "gamma_case_12"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_gamma_2"] = tf_random_stateless_gamma_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_random_stateless_parameterized_truncated_normal_inputs():
    list_of_inputs = []

    shape = np.array([10], dtype=np.int32)
    seed = np.array([123, 456], dtype=np.int32)
    means = np.array(0.0, dtype=np.float32)
    stddevs = np.array(1.0, dtype=np.float32)
    minvals = np.array(-2.0, dtype=np.float32)
    maxvals = np.array(2.0, dtype=np.float32)
    name = "basic_float32_scalar_params"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([5, 4], dtype=np.int64)
    seed = np.array([7, 17], dtype=np.int64)
    means = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    stddevs = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    minvals = np.array([-1.0, -1.0, -2.0, -3.0], dtype=np.float32)
    maxvals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    name = "per_column_params_float32"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([3, 2, 3], dtype=np.int32)
    seed = np.array([42, 24], dtype=np.int32)
    means = np.array([[0.0, 0.5, -0.5], [1.0, -1.0, 2.0]], dtype=np.float32)
    stddevs = np.array([[1.0, 0.2, 0.3], [0.5, 2.0, 1.5]], dtype=np.float32)
    minvals = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    maxvals = np.array([[3.0], [1.0]], dtype=np.float32)
    name = "broadcasting_example_float32"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([5, 2, 1], dtype=np.int64)
    seed = np.array([3, 999999], dtype=np.int64)
    means = np.array(-0.5, dtype=np.float64)
    stddevs = np.array([[0.1], [2.0]], dtype=np.float64)
    minvals = np.array(-1.5, dtype=np.float64)
    maxvals = np.array(1.5, dtype=np.float64)
    name = "float64_mixed_rank_params"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([4, 1, 3, 1], dtype=np.int32)
    seed = np.array([31415, 27182], dtype=np.int32)
    means = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32).reshape(1, 3, 1)
    stddevs = np.array([0.5, 1.0, 1.5], dtype=np.float32).reshape(1, 3, 1)
    minvals = np.array(-0.75, dtype=np.float32)
    maxvals = np.array(0.75, dtype=np.float32)
    name = "higher_rank_broadcast_float32"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([0, 7], dtype=np.int32)
    seed = np.array([2021, 2022], dtype=np.int64)
    means = np.array([0.0, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6], dtype=np.float32)
    stddevs = np.array([0.3] * 7, dtype=np.float32)
    minvals = np.array([-0.4] * 7, dtype=np.float32)
    maxvals = np.array([0.4] * 7, dtype=np.float32)
    name = "zero_sized_dim_float32"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([6, 5, 4], dtype=np.int64)
    seed = np.array([1, 2], dtype=np.int32)
    means = np.array(0.0, dtype=np.float32)
    stddevs = np.array(0.5, dtype=np.float32)
    minvals = np.array(-1.0, dtype=np.float32)
    maxvals = np.array(1.0, dtype=np.float32)
    name = "float32_scalar_params_large_shape"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([8], dtype=np.int32)
    seed = np.array([123456789, 987654321], dtype=np.int64)
    means = np.linspace(-2.0, 2.0, 8, dtype=np.float32)
    stddevs = np.linspace(0.1, 1.0, 8, dtype=np.float32)
    minvals = np.full((8,), -5.0, dtype=np.float32)
    maxvals = np.full((8,), 5.0, dtype=np.float32)
    name = "vectorized_params_float32"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([2, 3, 1, 5], dtype=np.int32)
    seed = np.array([0, 1], dtype=np.int32)
    means = (np.arange(3 * 1 * 5, dtype=np.float32).reshape(3, 1, 5) - 7.0) / 3.0
    stddevs = np.array(1.25, dtype=np.float32)
    minvals = np.full((1, 5), -0.5, dtype=np.float32)
    maxvals = np.full((3, 1, 5), 2.0, dtype=np.float32)
    name = "complex_broadcasting_float32"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([2, 2], dtype=np.int64)
    seed = np.array([1234, 5678], dtype=np.int64)
    means = np.array([[-1.0, 1.0], [2.0, -2.0]], dtype=np.float64)
    stddevs = np.array([[0.1, 0.2], [3.0, 4.0]], dtype=np.float64)
    minvals = np.array(-0.2, dtype=np.float64)
    maxvals = np.array(0.2, dtype=np.float64)
    name = "float64_matrix_params"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([3], dtype=np.int32)
    seed = np.array([13579, 24680], dtype=np.int32)
    means = np.array([0.0, -0.5, 0.5], dtype=np.float32)
    stddevs = np.array([1e-6, 1e-3, 1e-2], dtype=np.float32)
    minvals = np.array(-1e-3, dtype=np.float32)
    maxvals = np.array(1e-3, dtype=np.float32)
    name = "tiny_stddevs_float32"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    shape = np.array([1, 2, 1, 3], dtype=np.int64)
    seed = np.array([101, 202], dtype=np.int64)
    means = np.array([[[0.0, 1.0, 2.0]], [[-1.0, -2.0, -3.0]]], dtype=np.float32)
    stddevs = np.array(0.7, dtype=np.float32)
    minvals = np.array([-0.5, -1.0, -1.5], dtype=np.float32)
    maxvals = np.array([[[0.5, 1.5, 2.5]], [[1.0, 2.0, 3.0]]], dtype=np.float32)
    name = "mixed_rank_tail_broadcast_float32"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_parameterized_truncated_normal_1"] = tf_random_stateless_parameterized_truncated_normal_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_parameterized_truncated_normal_inputs():
    list_of_inputs = []

    # Input 1
    shape = [10]
    seed = np.array([123, 456], dtype=np.int32)
    means = np.array(0.0, dtype=np.float32)
    stddevs = np.array(1.0, dtype=np.float32)
    minvals = np.array(-2.0, dtype=np.float32)
    maxvals = np.array(2.0, dtype=np.float32)
    name = "case1"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    shape = [5, 3]
    seed = np.array([2021, 7], dtype=np.int64)
    means = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    stddevs = np.array([0.5, 2.0, 1.5], dtype=np.float64)
    minvals = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    maxvals = np.array([1.0, 4.0, 2.0], dtype=np.float64)
    name = "case2"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    shape = [4, 2, 3]
    seed = np.array([11, 22], dtype=np.int32)
    means = np.array(0.5, dtype=np.float16)
    stddevs = np.array(0.3, dtype=np.float16)
    minvals = np.array([[-1.0, -0.5, -2.0], [-1.5, -0.1, -1.0]], dtype=np.float16)
    maxvals = np.array([[1.0, 0.7, 2.0], [1.2, 0.8, 1.5]], dtype=np.float16)
    name = "case3"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    shape = [2, 3, 1, 4]
    seed = np.array([314, 159], dtype=np.int64)
    means = np.array(np.reshape(np.linspace(-1.0, 1.0, 12, dtype=np.float32), (3, 1, 4)), dtype=np.float32)
    stddevs = np.array(np.reshape(np.full(12, 0.8, dtype=np.float32), (3, 1, 4)), dtype=np.float32)
    minvals = np.array(-5.0, dtype=np.float32)
    maxvals = np.array(5.0, dtype=np.float32)
    name = "case4"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    shape = [7, 1]
    seed = np.array([100, 200], dtype=np.int32)
    means = np.array([2.0], dtype=np.float32)
    stddevs = np.array(0.75, dtype=np.float32)
    minvals = np.array([0.0], dtype=np.float32)
    maxvals = np.array([3.0], dtype=np.float32)
    name = "case5"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (empty first dimension)
    shape = [0, 3]
    seed = np.array([77, 88], dtype=np.int32)
    means = np.array([0.0, -1.0, 2.0], dtype=np.float32)
    stddevs = np.array(1.2, dtype=np.float32)
    minvals = np.array(-1.0, dtype=np.float32)
    maxvals = np.array(1.0, dtype=np.float32)
    name = "case6"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (broadcast across 2 and 4)
    shape = [6, 2, 4]
    seed = np.array([555, 999], dtype=np.int64)
    means = np.array([[0.0], [1.5]], dtype=np.float64)
    stddevs = np.array([[0.4, 0.6, 0.8, 1.0]], dtype=np.float64)
    minvals = np.array([[-3.0], [-2.0]], dtype=np.float64)
    maxvals = np.array([[3.0, 2.5, 2.0, 1.5]], dtype=np.float64)
    name = "case7"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (full matrix parameters)
    shape = [3, 3]
    seed = np.array([42, 24], dtype=np.int32)
    means = np.array([[0.0, 1.0, 2.0],
                      [-1.0, -2.0, -3.0],
                      [0.5, -0.5, 1.5]], dtype=np.float32)
    stddevs = np.array([[0.5, 0.6, 0.7],
                        [1.0, 1.2, 0.8],
                        [0.3, 0.4, 0.9]], dtype=np.float32)
    minvals = np.array([[-2.0, -2.0, -2.0],
                        [-3.0, -4.0, -5.0],
                        [-1.0, -1.0, -1.0]], dtype=np.float32)
    maxvals = np.array([[2.0, 2.0, 2.0],
                        [3.0, 4.0, 5.0],
                        [1.0, 1.0, 1.0]], dtype=np.float32)
    name = "case8"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (negative mean window)
    shape = [1]
    seed = np.array([1, 2], dtype=np.int32)
    means = np.array(-5.0, dtype=np.float32)
    stddevs = np.array(0.1, dtype=np.float32)
    minvals = np.array(-5.2, dtype=np.float32)
    maxvals = np.array(-4.8, dtype=np.float32)
    name = "case9"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (float16 vector params)
    shape = [8, 5]
    seed = np.array([8, 16], dtype=np.int64)
    means = np.array([0.0, -1.0, 1.0, 2.0, -2.0], dtype=np.float16)
    stddevs = np.array([0.5, 0.25, 0.75, 1.0, 0.6], dtype=np.float16)
    minvals = np.array([-1.5, -2.0, -0.5, 0.0, -3.0], dtype=np.float16)
    maxvals = np.array([1.5, 0.0, 2.0, 3.0, -1.0 + 4.0], dtype=np.float16)
    name = "case10"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (mixed broadcast across 3,4,5)
    shape = [2, 3, 4, 5]
    seed = np.array([1010, 2020], dtype=np.int32)
    means = np.array(np.random.uniform(-1.0, 1.0, size=(3, 1, 5)).astype(np.float32))
    stddevs = np.array(np.full((1, 4, 1), 0.9, dtype=np.float32))
    minvals = np.array(-3.0, dtype=np.float32)
    maxvals = np.array(3.0, dtype=np.float32)
    name = "case11"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (vector params matching shape)
    shape = [9]
    seed = np.array([31415, 92653], dtype=np.int64)
    means = np.array(np.linspace(-2.0, 2.0, 9), dtype=np.float64)
    stddevs = np.array(np.linspace(0.2, 1.0, 9), dtype=np.float64)
    minvals = np.array(-4.0, dtype=np.float64)
    maxvals = np.array(4.0, dtype=np.float64)
    name = "case12"
    input_dict = {"shape": shape, "seed": seed, "means": means, "stddevs": stddevs, "minvals": minvals, "maxvals": maxvals, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.random.stateless_parameterized_truncated_normal_2"] = tf_random_stateless_parameterized_truncated_normal_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_raw_ops_Bucketize_inputs():
    list_of_inputs = []

    # Input 1
    name = "basic_int32_2d"
    input_arr = np.array([[-5, 10000], [150, 10], [5, 100]], dtype=np.int32)
    boundaries = [0.0, 10.0, 100.0]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 2
    name = "float32_1d_neg_pos"
    input_arr = np.array([-2.5, 0.5, 1.1, 3.0, -1.0], dtype=np.float32)
    boundaries = [-1.0, 0.5, 2.5]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 3
    name = "int64_3d_random"
    input_arr = np.random.randint(-200, 1200, size=(2, 2, 3)).astype(np.int64)
    boundaries = [-100.0, 0.0, 100.0, 1000.0]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 4
    name = "scalar_float64_single_boundary"
    input_arr = np.array(3.14, dtype=np.float64)
    boundaries = [0.0]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 5
    name = "empty_boundaries_2d"
    input_arr = np.array([[1.2, -3.4], [5.6, 0.0]], dtype=np.float32)
    boundaries = []
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 6
    name = "int32_4d"
    input_arr = np.arange(12, dtype=np.int32).reshape(2, 1, 3, 2)
    boundaries = [-10.0, 10.0]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 7
    name = "int64_mixed_2d"
    input_arr = np.array([[-10, -3, 0], [2, 4, 9], [11, -6, 3]], dtype=np.int64)
    boundaries = [-5.5, -2.0, 3.3]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 8
    name = "empty_input_1d"
    input_arr = np.array([], dtype=np.float32)
    boundaries = [0.0, 1.0]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 9
    name = "large_ints_2d"
    input_arr = np.array([[-2000000000, 0], [2000000000, -1]], dtype=np.int32)
    boundaries = [-1000000000.0, 0.0, 1000000000.0]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 10
    name = "float64_5d_small_vals"
    input_arr = np.random.randn(1, 2, 1, 2, 2).astype(np.float64)
    boundaries = [-0.1, 0.1]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 11
    name = "int32_many_boundaries_1d"
    input_arr = np.arange(-12, 13, 3, dtype=np.int32)
    boundaries = [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 12
    name = "float32_decimals_2d"
    input_arr = np.array([[-4.2, -1.2, -0.5, 2.7, 10.1],
                          [0.0, 1.5, -3.5, 9.9, 2.6]], dtype=np.float32)
    boundaries = [-3.5, -1.2, 0.0, 2.7, 9.9]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Bucketize"] = tf_raw_ops_Bucketize_inputs()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(42)

def tf_raw_ops_Cross_inputs():
    list_of_inputs = []

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"name": "basic_float32_vec", "a": a, "b": b}))

    a = np.array([-1, 0, 1], dtype=np.int32)
    b = np.array([2, -3, 4], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "int32_vec", "a": a, "b": b}))

    a = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]], dtype=np.float64)
    b = np.array([[0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"name": "float64_matrix", "a": a, "b": b}))

    a = np.array([[3, -2, 5],
                  [-7, 4, 1],
                  [0, -1, -3]], dtype=np.int16)
    b = np.array([[-1, 6, -4],
                  [5, -2, -8],
                  [2, 2, 2]], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"name": "int16_matrix_neg", "a": a, "b": b}))

    a = np.array([[[10, 20, 30], [40, 50, 60]],
                  [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8)
    b = np.array([[[5, 4, 3], [2, 1, 0]],
                  [[255, 254, 253], [10, 20, 30]]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"name": "uint8_3d", "a": a, "b": b}))

    a = (np.random.randn(4, 5, 3) * 0.1).astype(np.float16)
    b = (np.random.randn(4, 5, 3) * 0.1).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"name": "float16_3d_random", "a": a, "b": b}))

    a = np.array([
        [[[1, -2, 3], [4, -5, 6]],
         [[-7, 8, -9], [10, -11, 12]]],
        [[[13, -14, 15], [-16, 17, -18]],
         [[19, -20, 21], [-22, 23, -24]]]
    ], dtype=np.int64)
    b = np.array([
        [[[6, 5, 4], [3, 2, 1]],
         [[-1, -2, -3], [-4, -5, -6]]],
        [[[7, 8, 9], [10, 11, 12]],
         [[-13, -14, -15], [-16, -17, -18]]]
    ], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "int64_4d", "a": a, "b": b}))

    a = (np.random.uniform(-1.0, 1.0, size=(1, 2, 1, 2, 3))).astype(np.float32)
    b = (np.random.uniform(-1.0, 1.0, size=(1, 2, 1, 2, 3))).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"name": "float32_5d", "a": a, "b": b}))

    a = np.array([[-128, 0, 127],
                  [50, -50, 25],
                  [-1, -1, -1]], dtype=np.int8)
    b = np.array([[127, 0, -128],
                  [-25, 50, -50],
                  [1, 2, 3]], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"name": "int8_extremes", "a": a, "b": b}))

    base_a = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    base_b = (np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3) + 1.5)
    a = base_a[:, ::-1, :]
    b = base_b[:, ::-1, :]
    list_of_inputs.append(copy.deepcopy({"name": "sliced_float32_3d", "a": a, "b": b}))

    a = np.empty((0, 3), dtype=np.float32)
    b = np.empty((0, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"name": "empty_batch_float32", "a": a, "b": b}))

    a = np.random.randint(-1000, 1000, size=(2, 1, 3, 4, 5, 3), dtype=np.int32)
    b = np.random.randint(-1000, 1000, size=(2, 1, 3, 4, 5, 3), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "int32_6d_random", "a": a, "b": b}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Cross"] = tf_raw_ops_Cross_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_EnsureShape_inputs():
    list_of_inputs = []

    arr = np.array(7, dtype=np.int32)
    shape = []
    input_dict = {"name": "scalar_int32_exact", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([0.1, -0.2, 3.3, 4.4, -5.5], dtype=np.float32)
    shape = [-1]
    input_dict = {"name": "vector_float32_dynamic", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[1, 2, 3], [-4, -5, -6]], dtype=np.int64)
    shape = [2, 3]
    input_dict = {"name": "matrix_int64_exact", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[1, -1, 2, -2], [3, -3, 4, -4]], dtype=np.int32)
    shape = [2, -1]
    input_dict = {"name": "matrix_partial_unknown", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.arange(2*3*4, dtype=np.float64).reshape(2, 3, 4)
    shape = [-1, 3, 4]
    input_dict = {"name": "tensor3d_partial", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([True, False, True], dtype=bool)
    shape = [3]
    input_dict = {"name": "vector_bool_exact", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[1+2j, -3+4j], [5-6j, -7-8j]], dtype=np.complex64)
    shape = [2, 2]
    input_dict = {"name": "matrix_complex64_exact", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([], dtype=np.int32)
    shape = [0]
    input_dict = {"name": "empty_vector_int32", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.zeros((0, 3), dtype=np.float32)
    shape = [0, 3]
    input_dict = {"name": "zero_rows_matrix_float32", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.arange(1*2*3*4, dtype=np.float16).reshape(1, 2, 3, 4)
    shape = [1, 2, 3, 4]
    input_dict = {"name": "tensor4d_float16_exact", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([b"hello", b"world"], dtype=np.object_)
    shape = [2]
    input_dict = {"name": "vector_string_bytes", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.zeros((1, 0, 5), dtype=np.int8)
    shape = [1, 0, 5]
    input_dict = {"name": "tensor3d_zero_mid_dim", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([1+0j, -2+3j, 4-5j, -6-7j], dtype=np.complex128)
    shape = [-1]
    input_dict = {"name": "vector_complex128_dynamic", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.EnsureShape"] = tf_raw_ops_EnsureShape_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_erf_inputs():
    list_of_inputs = []

    x = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, -2.0]], dtype=np.float32)
    input_dict = {"name": "erf_matrix_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-5.0, -2.5, 0.0], [1.25, 3.75, 10.0]], dtype=np.float64)
    input_dict = {"name": "erf_matrix_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(0.5, dtype=np.float16)
    input_dict = {"name": "erf_scalar_f16", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    input_dict = {"name": "erf_vector_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    rs = np.random.RandomState(0)
    x = rs.normal(size=(2, 2, 3)).astype(np.float64)
    input_dict = {"name": "erf_3d_random_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[[-10.0, 0.0, 10.0]]], [[[20.0, -20.0, 0.0]]]], dtype=np.float32)
    input_dict = {"name": "erf_4d_large_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    input_dict = {"name": "erf_empty_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0], dtype=np.float64)
    input_dict = {"name": "erf_specials_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.arange(0, 8, dtype=np.float16).reshape(1, 2, 2, 2, 1)
    input_dict = {"name": "erf_5d_f16", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.linspace(-2.0, 2.0, num=24, dtype=np.float32).reshape(2, 3, 4)
    input_dict = {"name": "erf_3d_linspace_f32", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-7.5, -0.0, 0.0, 7.5], dtype=np.float64)
    input_dict = {"name": "erf_vector_precise_f64", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-8.0, -4.0, -2.0], [2.0, 4.0, 8.0]], dtype=np.float16)
    input_dict = {"name": "erf_matrix_f16", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Erf"] = tf_raw_ops_erf_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_floormod_inputs():
    list_of_inputs = []

    x = np.array(7, dtype=np.int32)
    y = np.array(3, dtype=np.int32)
    input_dict = {"name": "case_int32_scalar_pos", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-7, dtype=np.int32)
    y = np.array(3, dtype=np.int32)
    input_dict = {"name": "case_int32_scalar_neg", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-7, -1, 0, 1, 7], dtype=np.int32)
    y = np.array([3, -3, 2, -2, 5], dtype=np.int32)
    input_dict = {"name": "case_int32_vector_mixed", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[10, -10], [5, -5]], dtype=np.int64)
    y = np.array(6, dtype=np.int64)
    input_dict = {"name": "case_int64_matrix_scalar_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([255, 128, 0], dtype=np.uint8)
    y = np.array([10, 7, 13], dtype=np.uint8)
    input_dict = {"name": "case_uint8_vector", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1000, -2000, 3000], [4000, -5000, 6000]], dtype=np.int16)
    y = np.array(-4, dtype=np.int16)
    input_dict = {"name": "case_int16_matrix_scalar_neg_divisor", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-7.5, 7.5, 0.0, -0.1], dtype=np.float32)
    y = np.array([3.0, -3.0, 2.0, 0.2], dtype=np.float32)
    input_dict = {"name": "case_float32_vector_mixed", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-5.0], [0.5], [8.0]], dtype=np.float64)
    y = np.array([[2.0, -2.0, 3.0, -3.0]], dtype=np.float64)
    input_dict = {"name": "case_float64_broadcast_2d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[ -1], [ -8], [  7]], [[  4], [-12], [ 15]]], dtype=np.int8)
    y = np.array([3, -5, 10], dtype=np.int8)
    input_dict = {"name": "case_int8_broadcast_3d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    y = np.array(-1.5, dtype=np.float32)
    input_dict = {"name": "case_float32_vector_scalar_neg_divisor", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = (np.arange(24, dtype=np.int32).reshape(1, 2, 3, 4) - 12).astype(np.int32)
    y = np.array([2, -2, 5, -7], dtype=np.int32)
    input_dict = {"name": "case_int32_4d_broadcast_lastdim", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.int64(2**62), np.int64(-2**62), np.int64(2**61 + 7)], dtype=np.int64)
    y = np.array(97, dtype=np.int64)
    input_dict = {"name": "case_int64_large_values_scalar_divisor", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FloorMod"] = tf_raw_ops_floormod_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_fractional_max_pool_inputs():
    list_of_inputs = []

    # 1
    value = np.random.randn(1, 8, 8, 3).astype(np.float32)
    pooling_ratio = [1.0, 1.5, 1.5, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": False,
        "deterministic": False,
        "seed": 0,
        "seed2": 0,
        "name": "case1",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    value = (np.arange(2*7*5*1).reshape(2, 7, 5, 1) - 10).astype(np.int32)
    pooling_ratio = [1.0, 1.8, 1.3, 1.0]
    input_dict = {
        "pseudo_random": True,
        "overlapping": False,
        "deterministic": False,
        "seed": 0,
        "seed2": 0,
        "name": "case2",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    value = (np.random.randn(1, 9, 4, 2) * 5.0).astype(np.float64)
    pooling_ratio = [1.0, 2.2, 1.7, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": True,
        "deterministic": True,
        "seed": 42,
        "seed2": 77,
        "name": "case3",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    value = (np.arange(3*10*10*1).reshape(3, 10, 10, 1) - 50).astype(np.int64)
    pooling_ratio = [1.0, 1.33, 1.67, 1.0]
    input_dict = {
        "pseudo_random": True,
        "overlapping": True,
        "deterministic": True,
        "seed": 999,
        "seed2": 1,
        "name": "case4",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    value = np.array([[[[1.0], [-2.0], [3.0]],
                       [[-4.0], [5.0], [-6.0]],
                       [[7.0], [-8.0], [9.0]]]], dtype=np.float32)
    pooling_ratio = [1.0, 1.2, 1.6, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": False,
        "deterministic": True,
        "seed": 0,
        "seed2": 0,
        "name": "case5",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    value = (np.random.randn(4, 5, 6, 2)).astype(np.float32)
    pooling_ratio = [1.0, 1.99, 1.01, 1.0]
    input_dict = {
        "pseudo_random": True,
        "overlapping": False,
        "deterministic": True,
        "seed": 0,
        "seed2": 0,
        "name": "case6",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    value = (np.random.randn(2, 12, 7, 4) * 2.0).astype(np.float64)
    pooling_ratio = [1.0, 3.0, 1.2, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": True,
        "deterministic": True,
        "seed": 321,
        "seed2": 654,
        "name": "case7",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    value = np.array([[[[1]], [[2]]],
                      [[[3]], [[4]]]], dtype=np.int32)
    pooling_ratio = [1.0, 1.0, 1.0, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": False,
        "deterministic": False,
        "seed": 0,
        "seed2": 0,
        "name": "case8",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    value = (np.random.randn(1, 15, 9, 1) * 10 - 5).astype(np.float32)
    pooling_ratio = [1.0, 1.3, 2.7, 1.0]
    input_dict = {
        "pseudo_random": True,
        "overlapping": True,
        "deterministic": False,
        "seed": 0,
        "seed2": 0,
        "name": "case9",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    value = (np.arange(5*13*13*3).reshape(5, 13, 13, 3) - 1000).astype(np.int64)
    pooling_ratio = [1.0, 2.5, 2.5, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": False,
        "deterministic": True,
        "seed": 7,
        "seed2": 11,
        "name": "case10",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    value = (np.random.randn(2, 14, 5, 2) * 0.5).astype(np.float32)
    pooling_ratio = [1.0, 1.01, 4.0, 1.0]
    input_dict = {
        "pseudo_random": True,
        "overlapping": False,
        "deterministic": False,
        "seed": 0,
        "seed2": 0,
        "name": "case11",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12
    value = (np.random.randn(1, 20, 20, 1) + 1.0).astype(np.float64)
    pooling_ratio = [1.0, 1.9, 1.9, 1.0]
    input_dict = {
        "pseudo_random": False,
        "overlapping": True,
        "deterministic": False,
        "seed": 0,
        "seed2": 0,
        "name": "case12",
        "value": value,
        "pooling_ratio": pooling_ratio
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FractionalMaxPool"] = tf_fractional_max_pool_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_gather_inputs():
    list_of_inputs = []

    params = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    indices = np.array(3, dtype=np.int32)
    input_dict = {
        "validate_indices": True,
        "name": "gather_scalar_idx",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    params = np.array([-5, -1, 0, 7, 9], dtype=np.int32)
    indices = np.array([0, 4, 2], dtype=np.int64)
    input_dict = {
        "validate_indices": False,
        "name": "gather_vector_idx_int64",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    params = np.arange(12, dtype=np.float64).reshape(3, 4)
    indices = np.array([2, 0], dtype=np.int32)
    input_dict = {
        "validate_indices": True,
        "name": "gather_2d_from_rows",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    params = np.arange(2 * 3 * 4, dtype=np.int64).reshape(2, 3, 4)
    indices = np.array([[1, 0], [0, 1]], dtype=np.int32)
    input_dict = {
        "validate_indices": True,
        "name": "gather_matrix_indices",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.arange(3 * 2 * 2 * 2).reshape(3, 2, 2, 2)
    params = (base % 2 == 0)
    indices = np.array([1, 2], dtype=np.int64)
    input_dict = {
        "validate_indices": False,
        "name": "gather_bool",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = np.arange(6, dtype=np.float32).reshape(2, 3)
    imag = np.arange(6, dtype=np.float32).reshape(2, 3)
    params = (real + 1j * imag).astype(np.complex64)
    indices = np.array(1, dtype=np.int64)
    input_dict = {
        "validate_indices": True,
        "name": "gather_complex",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    params = np.arange(10, dtype=np.float16).reshape(5, 1, 2)
    indices = np.array([4, 3, 2, 1, 0], dtype=np.int32)
    input_dict = {
        "validate_indices": True,
        "name": "gather_permute",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    params = np.empty((2, 0, 3), dtype=np.float32)
    indices = np.array([1, 0], dtype=np.int32)
    input_dict = {
        "validate_indices": True,
        "name": "gather_zero_len_inner",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    params = np.array([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]], dtype=np.uint8)
    indices = np.array([0, 0, 0], dtype=np.int64)
    input_dict = {
        "validate_indices": False,
        "name": "gather_repeat",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    params = np.array([[1.0, -1.0],
                       [2.0, -2.0],
                       [3.0, -3.0]], dtype=np.float32)
    indices = np.array([[[0, 1]], [[2, 0]]], dtype=np.int64)
    input_dict = {
        "validate_indices": True,
        "name": "gather_3d_indices",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    params = np.arange(8, dtype=np.int8).reshape(4, 2)
    indices = np.array(2, dtype=np.int32)
    input_dict = {
        "validate_indices": True,
        "name": "gather_int8_scalar_idx",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Gather"] = tf_raw_ops_gather_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_Imag_inputs():
    list_of_inputs = []

    arr1 = np.array([-2.25+4.75j, 3.25+5.75j, -0.0-7.125j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr1, "Tout": np.float32, "name": "imag_vec_c64_f32"}))

    arr2 = np.array([1+2j, -3-4.5j, 0+0j, 6-1e-3j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr2, "Tout": np.float64, "name": "imag_vec_c128_f64"}))

    arr3 = np.array([[-1+0.5j, 2-3j], [4+0j, -5-6j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr3, "Tout": np.float32, "name": "imag_mat_c64_f32"}))

    arr4 = np.array([[[1+1j, 2+2j], [3+3j, 4-4j]], [[-1-1j, -2+2j], [0+0j, 5-0.25j]]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr4, "Tout": np.float64, "name": "imag_3d_c128_f64"}))

    arr5 = np.array(3-7j, dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr5, "Tout": np.float32, "name": "imag_scalar_c64_f32"}))

    arr6 = np.array([], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr6, "Tout": np.float32, "name": "imag_empty1d_c64_f32"}))

    arr7 = np.empty((2, 0), dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr7, "Tout": np.float64, "name": "imag_2x0_c128_f64"}))

    arr8 = np.array([np.nan + 1j, 2 + np.inf*1j, -np.inf + (-np.inf)*1j, np.nan + np.nan*1j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr8, "Tout": np.float64, "name": "imag_specials_c128_f64"}))

    arr9 = np.zeros((2, 1, 1, 3), dtype=np.complex64)
    arr9[0, 0, 0, :] = np.array([1+10j, -2-20j, 0+0j], dtype=np.complex64)
    arr9[1, 0, 0, :] = np.array([3-30j, -4+40j, 5-50j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"input": arr9, "Tout": np.float32, "name": "imag_4d_c64_f32"}))

    arr10 = np.array([1+0j, -2+0j, 3+0j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr10, "Tout": np.float64, "name": "imag_zeroimag_c128_f64"}))

    base = np.array([0+1j, 1+2j, 2+3j, 3+4j, 4+5j, 5+6j], dtype=np.complex64)
    arr11 = base[::-2]
    list_of_inputs.append(copy.deepcopy({"input": arr11, "Tout": np.float32, "name": "imag_view_c64_f32"}))

    arr12 = np.array([1e-30+1e-30j, -1e30-1e30j, 3.141592653589793+2.718281828459045j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"input": arr12, "Tout": np.float64, "name": "imag_mixed_mags_c128_f64"}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Imag"] = tf_raw_ops_Imag_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_LRN_inputs():
    rs = np.random.RandomState(42)
    list_of_inputs = []

    x1 = np.array([[[[1.0]]]], dtype=np.float32)
    input_dict = {
        "depth_radius": 0,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_case1",
        "input": copy.deepcopy(x1),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x2 = rs.randn(2, 2, 2, 3).astype(np.float32)
    input_dict = {
        "depth_radius": 1,
        "bias": 2.0,
        "alpha": 1e-4,
        "beta": 0.75,
        "name": "lrn_case2",
        "input": copy.deepcopy(x2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x3 = (rs.randn(1, 3, 3, 8) * 2 - 1).astype(np.float16)
    input_dict = {
        "depth_radius": 2,
        "bias": 1.0,
        "alpha": 0.5,
        "beta": 1.0,
        "name": "lrn_case3",
        "input": copy.deepcopy(x3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x4 = rs.randn(4, 4, 1, 16).astype(np.float32)
    input_dict = {
        "depth_radius": 5,
        "bias": 1e-4,
        "alpha": 1.5,
        "beta": 0.5,
        "name": "lrn_case4",
        "input": copy.deepcopy(x4),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x5 = rs.randn(1, 5, 5, 32).astype(np.float16)
    input_dict = {
        "depth_radius": 0,
        "bias": 10.0,
        "alpha": 1e-3,
        "beta": 0.5,
        "name": "lrn_case5",
        "input": copy.deepcopy(x5),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x6 = rs.randn(2, 1, 7, 4).astype(np.float32)
    input_dict = {
        "depth_radius": 3,
        "bias": 0.75,
        "alpha": 2.0,
        "beta": 0.25,
        "name": "lrn_case6",
        "input": copy.deepcopy(x6),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x7 = rs.randn(3, 2, 2, 2).astype(np.float16)
    input_dict = {
        "depth_radius": 1,
        "bias": 1.0,
        "alpha": 0.1,
        "beta": 2.0,
        "name": "lrn_case7",
        "input": copy.deepcopy(x7),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x8 = rs.uniform(-3, 3, size=(1, 2, 3, 5)).astype(np.float32)
    input_dict = {
        "depth_radius": 4,
        "bias": 1.0,
        "alpha": 1.0,
        "beta": 0.5,
        "name": "lrn_case8",
        "input": copy.deepcopy(x8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x9 = rs.randn(2, 3, 1, 10).astype(np.float32)
    input_dict = {
        "depth_radius": 9,
        "bias": 0.9,
        "alpha": 0.01,
        "beta": 0.5,
        "name": "lrn_case9",
        "input": copy.deepcopy(x9),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x10 = rs.randn(1, 1, 10, 3).astype(np.float16)
    input_dict = {
        "depth_radius": 1,
        "bias": 1.0,
        "alpha": 0.75,
        "beta": 0.0,
        "name": "lrn_case10",
        "input": copy.deepcopy(x10),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x11 = (rs.randn(5, 5, 5, 6) * 0.1).astype(np.float32)
    input_dict = {
        "depth_radius": 2,
        "bias": 1e-6,
        "alpha": 1e-3,
        "beta": 0.5,
        "name": "lrn_case11",
        "input": copy.deepcopy(x11),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x12 = rs.randn(2, 2, 2, 64).astype(np.float32)
    input_dict = {
        "depth_radius": 7,
        "bias": 3.0,
        "alpha": 5e-4,
        "beta": 1.5,
        "name": "lrn_case12",
        "input": copy.deepcopy(x12),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.LRN"] = tf_raw_ops_LRN_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_logsoftmax_inputs():
    list_of_inputs = []

    logits = np.array([[1.0, 2.0, 3.0], [0.5, -1.5, 2.5]], dtype=np.float32)
    name = "logsoftmax_input_1"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.array([[-1.0, -2.0, -3.0, -4.0], [4.0, 3.0, 2.0, 1.0]], dtype=np.float64)
    name = "logsoftmax_input_2"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.array([[100.0, -100.0], [0.0, 0.0], [1.0, -1.0]], dtype=np.float16)
    name = "logsoftmax_input_3"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.linspace(-5, 5, num=20, dtype=np.float32).reshape(1, -1)
    name = "logsoftmax_input_4"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.empty((0, 5), dtype=np.float32)
    name = "logsoftmax_input_5"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.array([[3.14], [-2.71], [0.0]], dtype=np.float64)
    name = "logsoftmax_input_6"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.array([[np.inf, 0.0, -np.inf], [np.nan, 1.0, -1.0]], dtype=np.float64)
    name = "logsoftmax_input_7"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -999.5, -1001.0]], dtype=np.float32)
    name = "logsoftmax_input_8"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    rng = np.random.default_rng(42)
    logits = rng.normal(loc=0.0, scale=5.0, size=(4, 4)).astype(np.float16)
    name = "logsoftmax_input_9"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.array([[-0.0, 0.0], [5.0, 5.0]], dtype=np.float32)
    name = "logsoftmax_input_10"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                       [-0.5, -0.4, -0.3, -0.2, -0.1]], dtype=np.float64)
    name = "logsoftmax_input_11"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    logits = rng.uniform(low=-10.0, high=10.0, size=(2, 50)).astype(np.float32)
    name = "logsoftmax_input_12"
    list_of_inputs.append(copy.deepcopy({"name": name, "logits": logits}))

    return list_of_inputs

generated_inputs["tf.raw_ops.LogSoftmax"] = tf_raw_ops_logsoftmax_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_MatrixSetDiag_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    diagonal_arr = np.array([9, 8], dtype=np.int32)
    input_dict = {"name": "msd_int32_square_2x2", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = np.arange(15, dtype=np.float32).reshape(3, 5)
    diagonal_arr = np.array([10.5, -2.0, 3.3], dtype=np.float32)
    input_dict = {"name": "msd_float32_rect_3x5", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = np.arange(18, dtype=np.int64).reshape(2, 3, 3)
    diagonal_arr = np.array([[1, -1, 1], [2, -2, 2]], dtype=np.int64)
    input_dict = {"name": "msd_int64_batch_2_3x3", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = (np.arange(2 * 1 * 3 * 2, dtype=np.float16).reshape(2, 1, 3, 2) / 10.0).astype(np.float16)
    diagonal_arr = np.array([[[0.5, -1.5]], [[2.0, 3.0]]], dtype=np.float16)
    input_dict = {"name": "msd_float16_batch_2_1_3x2", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = (np.arange(16, dtype=np.float32).reshape(4, 4) + 1j * np.arange(16, dtype=np.float32).reshape(4, 4)).astype(np.complex64)
    diagonal_arr = np.array([1 + 1j, -2 + 0.5j, 3 - 3j, -4 + 2j], dtype=np.complex64)
    input_dict = {"name": "msd_complex64_4x4", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = (np.arange(3 * 2 * 4, dtype=np.float64).reshape(3, 2, 4) - 5.0).astype(np.float64)
    diagonal_arr = np.array([[1.1, -2.2], [3.3, -4.4], [5.5, -6.6]], dtype=np.float64)
    input_dict = {"name": "msd_float64_batch_3_2x4", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = (np.arange(35, dtype=np.int32).reshape(5, 7) * -1).astype(np.int32)
    diagonal_arr = np.array([-9, -8, -7, -6, -5], dtype=np.int32)
    input_dict = {"name": "msd_int32_rect_5x7", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = (np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4) / 3.0).astype(np.float32)
    diagonal_arr = np.random.uniform(-1.0, 1.0, size=(2, 3, 4)).astype(np.float32)
    input_dict = {"name": "msd_float32_large_batch_2_3_4x4", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = np.zeros((3, 3), dtype=np.float32)
    diagonal_arr = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    input_dict = {"name": "msd_float32_with_nans_infs", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = (np.arange(2 * 2 * 2, dtype=np.float64).reshape(2, 2, 2) + 1j * (np.arange(2 * 2 * 2, dtype=np.float64).reshape(2, 2, 2) + 1)).astype(np.complex128)
    diagonal_arr = np.array([[1 + 0j, 2 - 1j], [-1 + 2j, 0 + 0j]], dtype=np.complex128)
    input_dict = {"name": "msd_complex128_batch_2_2x2", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = np.array([[3.14, 2.71, -1.0, 0.0]], dtype=np.float32)
    diagonal_arr = np.array([42.0], dtype=np.float32)
    input_dict = {"name": "msd_float32_min_diag_len1_row", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    diagonal_arr = np.array([7.0], dtype=np.float64)
    input_dict = {"name": "msd_float64_min_diag_len1_col", "input": input_arr, "diagonal": diagonal_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.MatrixSetDiag"] = tf_raw_ops_MatrixSetDiag_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_maximum_inputs():
    list_of_inputs = []

    x = np.array([0., -1., 2., -3., 4.], dtype=np.float32)
    y = np.array([-2., 0.5, 1.5, -5., 4.], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"name": "max_f32_vec", "x": x, "y": y}))

    x = np.array([[-5.0, 0.0, 7.5], [1.2, -3.4, 9.9]], dtype=np.float64)
    y = np.array(-3.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"name": "max_f64_mat_scalar", "x": x, "y": y}))

    x = np.arange(-12, 12, dtype=np.float16).reshape(2, 3, 4)
    y = np.array([[[0.5], [-1.0], [3.0]]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"name": "max_f16_3d_bcast", "x": x, "y": y}))

    x = np.array([[-10, 0, 5], [7, -8, 2]], dtype=np.int32)
    y = np.array([[-5, -1, 6], [7, 8, -3]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "max_i32_2d_same", "x": x, "y": y}))

    x = np.array([-128, -1, 0, 127], dtype=np.int8)
    y = np.array([-2], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"name": "max_i8_vec_scalar", "x": x, "y": y}))

    x = np.array([[[0], [5], [255]], [[100], [150], [200]]], dtype=np.uint8)
    y = np.array([[[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"name": "max_u8_3d_bcast", "x": x, "y": y}))

    x = np.array(
        [
            [[-32768, -123, 0], [32767, 1000, -1]],
            [[-500, 500, -200], [200, -100, 100]],
        ],
        dtype=np.int16,
    ).reshape(2, 1, 2, 3)
    y = np.array(-1000, dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"name": "max_i16_4d_scalar", "x": x, "y": y}))

    x = np.array(-1234567890123456789, dtype=np.int64)
    y = np.array([10, -20], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "max_i64_scalar_vec", "x": x, "y": y}))

    x = np.array([[np.nan, np.inf], [-np.inf, 3.0]], dtype=np.float32)
    y = np.array([[1.0, -np.inf], [np.inf, np.nan]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"name": "max_f32_nans_infs", "x": x, "y": y}))

    x = np.arange(6, dtype=np.float32).reshape(2, 1, 3)
    y = np.array([[[0.0], [1.5], [2.5], [3.5]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"name": "max_f32_bcast_mixed", "x": x, "y": y}))

    x = np.array([np.iinfo(np.int32).min, -1000000000, 0, 1000000000, np.iinfo(np.int32).max], dtype=np.int32)
    y = np.array([-1, -2000000000, 1, 2000000000, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "max_i32_large_values", "x": x, "y": y}))

    x = np.array([[1.0, -2.0, 3.0, -4.0], [5.0, -6.0, 7.0, -8.0]], dtype=np.float16)
    y = np.array([0.0, -3.0, 2.0, -9.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"name": "max_f16_matrix_vector", "x": x, "y": y}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Maximum"] = tf_raw_ops_maximum_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_pow_inputs():
    list_of_inputs = []

    x = np.array([2.0, 3.0, -4.0], dtype=np.float32)
    y = np.array([3.0, 2.0, 1.5], dtype=np.float32)
    input_dict = {"name": "pow_float32_vector", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.5, 2.0], [-3.0, 0.5]], dtype=np.float64)
    y = np.array(2.0, dtype=np.float64)
    input_dict = {"name": "pow_float64_broadcast_scalar", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[-2, -1], [0, 1]], [[2, 3], [4, 5]]], dtype=np.int32)
    y = np.array([[[0, 1], [2, 3]], [[4, 0], [1, 2]]], dtype=np.int32)
    input_dict = {"name": "pow_int32_3d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-2, dtype=np.int8)
    y = np.array([0, 1, 2, 3], dtype=np.int8)
    input_dict = {"name": "pow_int8_scalar_vector_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10**6, -2, 0, 1], dtype=np.int64)
    y = np.array([0, 1, 5, 63], dtype=np.int64)
    input_dict = {"name": "pow_int64_vector", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0],
                  [9.0, 10.0, 11.0, 12.0]], dtype=np.float16)
    y = np.array([0.5, 1.0, 2.0, -1.0], dtype=np.float16)
    input_dict = {"name": "pow_float16_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1+1j, -2+0j], [0+2j, -3-4j]], dtype=np.complex64)
    y = np.array([[2+0j, 0.5+0j], [3+0j, -1+0j]], dtype=np.complex64)
    input_dict = {"name": "pow_complex64_matrix", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1-1j, 2+3j, -1+2j],
                   [0+1j, -2-2j, 3-1j]]], dtype=np.complex128)
    y = np.array([[[1+0j]], [[-0.5+0.5j]]], dtype=np.complex128)
    input_dict = {"name": "pow_complex128_broadcast_nd", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-32768, -1, 0, 12345], dtype=np.int16)
    y = np.zeros((4,), dtype=np.int16)
    input_dict = {"name": "pow_int16_zeros_exp", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(3.5, dtype=np.float32)
    y = np.array(-2.0, dtype=np.float32)
    input_dict = {"name": "pow_float32_scalar_negexp", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[2.0], [3.0], [4.0]]], dtype=np.float64)
    y = np.array([[[0.0, 1.0, 2.0, -1.0]],
                  [[1.5, -0.5, 0.25, 3.0]]], dtype=np.float64)
    input_dict = {"name": "pow_float64_broadcast_nd", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-10, -1, 1, 2, 10], dtype=np.int32)
    y = np.ones((5,), dtype=np.int32)
    input_dict = {"name": "pow_int32_ones_exp", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(0+1j, dtype=np.complex64)
    y = np.array(10+0j, dtype=np.complex64)
    input_dict = {"name": "pow_complex64_scalar", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.0, -1.0, 2.5]], dtype=np.float64)
    y = np.array([[3.0, -0.5, 0.0]], dtype=np.float64)
    input_dict = {"name": "pow_float64_edge_cases", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Pow"] = tf_raw_ops_pow_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_randomshuffle_inputs():
    list_of_inputs = []

    # Input 1: 1D int32
    value = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    seed = np.int32(1)
    seed2 = np.int32(2)
    name = "shuffle_1d_int32"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 2: 2D float32 with negatives
    value = np.array([[1.5, -2.3], [3.0, 4.1], [0.0, 9.9]], dtype=np.float32)
    seed = np.int64(42)
    seed2 = np.int64(7)
    name = "shuffle_2d_float32"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 3: 2D int64 with negatives
    value = np.array([[-1, -2, -3], [4, 5, 6]], dtype=np.int64)
    seed = np.int64(7)
    seed2 = np.int64(99)
    name = "shuffle_2d_int64_neg"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 4: 2D bool
    value = np.array([[True, False], [False, True], [True, True], [False, False]], dtype=np.bool_)
    seed = np.int32(123)
    seed2 = np.int32(456)
    name = "shuffle_2d_bool"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 5: 3D float16
    value = (np.arange(2 * 3 * 4, dtype=np.float16).reshape(2, 3, 4) - np.float16(10.5))
    seed = np.int32(11)
    seed2 = np.int32(22)
    name = "shuffle_3d_float16"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 6: 2D float64 single row
    value = np.array([[-1.0, -2.0, 0.0, 2.5, 10.75]], dtype=np.float64)
    seed = np.int32(1)
    seed2 = np.int32(3)
    name = "shuffle_2d_float64_row"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 7: 2D uint8
    value = np.arange(24, dtype=np.uint8).reshape(6, 4)
    seed = np.int32(2024)
    seed2 = np.int32(8)
    name = "shuffle_2d_uint8"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 8: 4D int16
    value = np.random.randint(-100, 100, size=(5, 2, 3, 4)).astype(np.int16)
    seed = np.int64(8080)
    seed2 = np.int64(9090)
    name = "shuffle_4d_int16"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 9: 5D float32
    value = np.arange(12, dtype=np.float32).reshape(2, 1, 3, 1, 2)
    seed = np.int64(314159)
    seed2 = np.int64(2653)
    name = "shuffle_5d_float32"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    # Input 10: Empty along first dim, float64
    value = np.zeros((0, 5), dtype=np.float64)
    seed = np.int64(100)
    seed2 = np.int64(200)
    name = "shuffle_empty_first_dim"
    list_of_inputs.append(copy.deepcopy({"seed": seed, "seed2": seed2, "name": name, "value": value}))

    return list_of_inputs

generated_inputs["tf.raw_ops.RandomShuffle"] = tf_raw_ops_randomshuffle_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_randomuniform_inputs():
    list_of_inputs = []

    shape = np.array([2, 3], dtype=np.int32)
    dtype = np.float32
    seed = 7
    seed2 = 11
    name = "ru_2x3_f32"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([], dtype=np.int32)
    dtype = np.float64
    seed = 123
    seed2 = 1
    name = "ru_scalar_f64"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([5], dtype=np.int64)
    dtype = np.float16
    seed = 456
    seed2 = 789
    name = "ru_len5_f16"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([2, 0, 4], dtype=np.int32)
    dtype = np.float32
    seed = 999
    seed2 = 1001
    name = "ru_zero_dim_mid_f32"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([1, 1, 1], dtype=np.int64)
    dtype = np.float64
    seed = 42
    seed2 = 24
    name = "ru_ones_f64"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([10, 10], dtype=np.int32)
    dtype = np.float16
    seed = 2021
    seed2 = 2022
    name = "ru_10x10_f16_bothseed"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([3, 4, 5, 6], dtype=np.int64)
    dtype = np.float32
    seed = 31415
    seed2 = 27182
    name = "ru_3x4x5x6_f32"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([7], dtype=np.int32)
    dtype = np.float64
    seed = 707
    seed2 = 808
    name = "ru_len7_f64"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([2, 3, 1, 0, 4], dtype=np.int64)
    dtype = np.float32
    seed = 8080
    seed2 = 9090
    name = "ru_5d_with_zero_f32"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([1000], dtype=np.int32)
    dtype = np.float32
    seed = 1
    seed2 = 2
    name = "ru_len1000_f32"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([8, 8, 8], dtype=np.int32)
    dtype = np.float64
    seed = 123456
    seed2 = 654321
    name = "ru_8cube_f64"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    shape = np.array([0], dtype=np.int64)
    dtype = np.float16
    seed = 42
    seed2 = 43
    name = "ru_emptyvec_f16"
    list_of_inputs.append(copy.deepcopy({
        "seed": seed, "seed2": seed2, "name": name, "shape": shape, "dtype": dtype
    }))

    return list_of_inputs

generated_inputs["tf.raw_ops.RandomUniform"] = tf_raw_ops_randomuniform_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_realdiv_inputs():
    list_of_inputs = []

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    input_dict = {"name": "case_float32_vector", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, -2.0], [3.5, -4.5]], dtype=np.float64)
    y = np.array(2.0, dtype=np.float64)
    input_dict = {"name": "case_scalar_divisor_float64", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[-5.0, -6.0], [7.0, 8.0]]], dtype=np.float16)
    y = np.array([[[1.0, -1.0], [2.0, -2.0]], [[-1.0, 1.0], [0.5, -0.5]]], dtype=np.float16)
    input_dict = {"name": "case_float16_3d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([[1.0, -1.0, 2.0, -2.0]], dtype=np.float32)
    input_dict = {"name": "case_broadcast_2d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1 + 2j, -3 + 4j, -1 - 1j], dtype=np.complex64)
    y = np.array([1 - 1j, 2 + 0j, -0.5 + 0.5j], dtype=np.complex64)
    input_dict = {"name": "case_complex64_vector", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1 + 0j, 0 + 1j, -1 - 1j], [2 - 2j, -3 + 0j, 4 + 4j]], dtype=np.complex128)
    y = np.array([1 + 0j, -1 + 1j, 2 - 1j], dtype=np.complex128)
    input_dict = {"name": "case_complex128_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1.0, -2.0, 0.0], dtype=np.float32)
    y = np.array([0.0, 2.0, -0.0], dtype=np.float32)
    input_dict = {"name": "case_zero_division", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = (np.arange(6, dtype=np.float64).reshape(2, 1, 3, 1) - 2.5)
    y = np.array([[1.0], [-2.0], [3.0]], dtype=np.float64)
    input_dict = {"name": "case_float64_4d_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(10.0, dtype=np.float32)
    y = np.array([1.0, -2.0, 5.0, -10.0], dtype=np.float32)
    input_dict = {"name": "case_scalar_numerator", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1000.0, 0.125, 65504.0, -0.0005, 1.5], dtype=np.float16)
    y = np.array([2.0, -0.5, 256.0, 2.0, -3.0], dtype=np.float16)
    input_dict = {"name": "case_float16_edge_values", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(3 + 4j, dtype=np.complex128)
    y = np.array([[1 - 1j, 2 + 2j], [-3 + 0j, 4 - 4j]], dtype=np.complex128)
    input_dict = {"name": "case_complex128_scalar_over_matrix", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[np.inf, -np.inf], [np.nan, 1.0]], dtype=np.float32)
    y = np.array([[1.0, -2.0], [3.0, np.nan]], dtype=np.float32)
    input_dict = {"name": "case_nan_inf", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.RealDiv"] = tf_raw_ops_realdiv_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SegmentSum_inputs():
    list_of_inputs = []

    data = np.array([[1.0, 2.0, 3.0, 4.0],
                     [4.0, 3.0, 2.0, 1.0],
                     [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    input_dict = {"name": "seg_sum_case_1", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([-3, 0, 2, -1, 5, -2], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    input_dict = {"name": "seg_sum_case_2", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.arange(24, dtype=np.float64).reshape(4, 2, 3)
    segment_ids = np.array([0, 0, 1, 2], dtype=np.int32)
    input_dict = {"name": "seg_sum_case_3", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([10, 20, 30, 40, 50], dtype=np.uint8)
    segment_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    input_dict = {"name": "seg_sum_case_4", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[-10, 20, -30],
                     [40, -50, 60],
                     [70, -80, 90],
                     [-100, 110, -120],
                     [130, -140, 150],
                     [160, -170, 180]], dtype=np.int16)
    segment_ids = np.array([0, 1, 1, 3, 3, 3], dtype=np.int64)
    input_dict = {"name": "seg_sum_case_5", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[[1, -2], [3, -4]],
                     [[5, 6], [-7, 8]],
                     [[-9, 10], [11, -12]],
                     [[13, -14], [15, 16]]], dtype=np.int8)
    segment_ids = np.array([0, 2, 2, 2], dtype=np.int32)
    input_dict = {"name": "seg_sum_case_6", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[1+2j, -3+4j, 5-6j],
                     [7+0j, -1-1j, 2+2j],
                     [0+3j, -4-5j, 6+6j]], dtype=np.complex64)
    segment_ids = np.array([0, 1, 1], dtype=np.int32)
    input_dict = {"name": "seg_sum_case_7", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16],
                     [17, 18, 19, 20]], dtype=np.int64)
    segment_ids = np.array([0, 0, 0, 0, 0], dtype=np.int64)
    input_dict = {"name": "seg_sum_case_8", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[[1, 2], [3, 4]],
                     [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=np.float16)
    segment_ids = np.array([0, 0, 2], dtype=np.int32)
    input_dict = {"name": "seg_sum_case_9", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([1+0j, 0+1j, 3+3j, -2-2j], dtype=np.complex128)
    segment_ids = np.array([0, 1, 1, 3], dtype=np.int64)
    input_dict = {"name": "seg_sum_case_10", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.arange(35, dtype=np.float32).reshape(7, 5)
    segment_ids = np.array([0, 0, 1, 1, 1, 2, 3], dtype=np.int32)
    input_dict = {"name": "seg_sum_case_11", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[100, -50, 25],
                     [0, 0, 0],
                     [1, 2, 3],
                     [9, 9, 9]], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 5], dtype=np.int64)
    input_dict = {"name": "seg_sum_case_12", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SegmentSum"] = tf_raw_ops_SegmentSum_inputs()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(0)

def tf_sparse_segment_mean_inputs():
    list_of_inputs = []

    data = np.array([1.0, -2.5, 3.5, 4.0, 5.5, -6.0], dtype=np.float32)
    indices = np.array([0, 2, 5, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "case1",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([
        [1.0, 2.0, -1.0],
        [3.0, -4.0, 5.0],
        [6.0, 7.0, 8.0],
        [-9.0, 10.0, 11.0],
        [12.0, -13.0, 14.0]
    ], dtype=np.float64)
    indices = np.array([0, 1, 3, 4, 2], dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    input_dict = {
        "sparse_gradient": True,
        "name": "case2",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([
        [[1.0, -1.0], [2.0, -2.0]],
        [[3.0, -3.0], [4.0, -4.0]],
        [[5.0, -5.0], [6.0, -6.0]],
        [[7.0, -7.0], [8.0, -8.0]]
    ], dtype=np.float16)
    indices = np.array([3, 1, 0], dtype=np.int32)
    segment_ids = np.array([0, 0, 2], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "case3",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = (np.arange(8, dtype=np.float32).reshape(8, 1) * 0.5) - 2.0
    indices = np.array([7, 0, 2, 5, 3], dtype=np.int64)
    segment_ids = np.array([0, 1, 1, 3, 3], dtype=np.int64)
    input_dict = {
        "sparse_gradient": True,
        "name": "case4",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([10.0, -10.0, 20.0, -20.0], dtype=np.float64)
    indices = np.array([1, 1, 2, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "case5",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.random.randn(10, 4, 5).astype(np.float32)
    indices = np.arange(10, dtype=np.int64)
    segment_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    input_dict = {
        "sparse_gradient": True,
        "name": "case6",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float16)
    indices = np.array([2, 2, 2], dtype=np.int32)
    segment_ids = np.array([0, 1, 2], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "case7",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = (np.arange(24, dtype=np.float32).reshape(6, 2, 1, 2) - 12.0) / 3.0
    indices = np.array([1, 1, 2, 4, 0], dtype=np.int64)
    segment_ids = np.array([0, 0, 2, 2, 5], dtype=np.int64)
    input_dict = {
        "sparse_gradient": True,
        "name": "case8",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([1.5, -2.5, 3.5, 4.5, -5.5], dtype=np.float16)
    indices = np.array([3], dtype=np.int32)
    segment_ids = np.array([3], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "case9",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[0.0, -1.0, 2.0, -3.0, 4.0]], dtype=np.float64)
    indices = np.array([0, 0, 0, 0], dtype=np.int64)
    segment_ids = np.array([0, 0, 0, 0], dtype=np.int64)
    input_dict = {
        "sparse_gradient": True,
        "name": "case10",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([
        [0.0, 1.0],
        [2.0, 3.0],
        [4.0, 5.0],
        [6.0, 7.0],
        [8.0, 9.0],
        [10.0, 11.0],
        [12.0, 13.0]
    ], dtype=np.float32)
    indices = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
    segment_ids = np.array([0, 0, 2, 2, 2, 5, 5], dtype=np.int32)
    input_dict = {
        "sparse_gradient": False,
        "name": "case11",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([
        [[-1.0, 1.0], [-2.0, 2.0]],
        [[-3.0, 3.0], [-4.0, 4.0]]
    ], dtype=np.float32)
    indices = np.array([1, 1, 0], dtype=np.int64)
    segment_ids = np.array([0, 1, 1], dtype=np.int64)
    input_dict = {
        "sparse_gradient": True,
        "name": "case12",
        "data": data,
        "indices": indices,
        "segment_ids": segment_ids
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSegmentMean"] = tf_sparse_segment_mean_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseSoftmaxCrossEntropyWithLogits_inputs():
    list_of_inputs = []

    # Input 1: float32, moderate values
    features = np.array([[1.0, -1.0, 0.5, 2.0, -0.3],
                         [0.0, 0.0, 0.0, 0.0, 0.0],
                         [-2.0, 3.0, 0.1, -0.5, 1.5]], dtype=np.float32)
    labels = np.array([3, 0, 1], dtype=np.int32)
    input_dict = {"name": "case_float32_basic", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, single sample
    features = np.array([[10.0, -5.0, 0.0, 2.5]], dtype=np.float64)
    labels = np.array([2], dtype=np.int64)
    input_dict = {"name": "case_float64_single_sample", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16, small matrix
    features = np.array([[0.5, -0.5],
                         [-1.2, 1.2]], dtype=np.float16)
    labels = np.array([0, 1], dtype=np.int32)
    input_dict = {"name": "case_float16_small", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, 4x3 with mix of negatives/positives
    features = np.array([[2.0, -3.0, 0.1],
                         [-1.0, 4.0, -0.2],
                         [0.0, 0.0, 0.0],
                         [5.0, -5.0, 1.0]], dtype=np.float32)
    labels = np.array([1, 1, 2, 0], dtype=np.int64)
    input_dict = {"name": "case_float32_mixed_values", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64, wider class set
    features = np.array([[1.5, -2.5, 3.0, 0.0, -1.0, 2.2],
                         [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6]], dtype=np.float64)
    labels = np.array([2, 5], dtype=np.int32)
    input_dict = {"name": "case_float64_wide_classes", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32, single class (degenerate softmax)
    features = np.array([[0.0],
                         [1.0],
                         [-1.0],
                         [3.14],
                         [-2.71]], dtype=np.float32)
    labels = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    input_dict = {"name": "case_float32_single_class", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float16, 3x4 with varied magnitudes
    features = np.array([[8.0, -8.0, 1.0, -1.0],
                         [0.25, -0.25, 0.5, -0.5],
                         [4.0, 3.0, -2.0, 1.0]], dtype=np.float16)
    labels = np.array([0, 3, 1], dtype=np.int32)
    input_dict = {"name": "case_float16_varied", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, extreme logits for stability
    features = np.array([[-1000.0, 0.0, 1000.0],
                         [1000.0, -1000.0, 0.0]], dtype=np.float32)
    labels = np.array([2, 0], dtype=np.int64)
    input_dict = {"name": "case_float32_extreme_logits", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64, binary classification, larger batch
    features = np.array([[0.1, -0.1],
                         [2.0, -2.0],
                         [-3.0, 3.0],
                         [4.5, -4.5],
                         [0.0, 0.0],
                         [1.1, -1.1],
                         [-0.7, 0.7],
                         [5.0, -5.0],
                         [-2.2, 2.2],
                         [3.3, -3.3]], dtype=np.float64)
    labels = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32)
    input_dict = {"name": "case_float64_binary_large_batch", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, 2x10 multi-class
    features = np.array([[0.5, -0.2, 1.5, -1.2, 0.3, 2.0, -0.7, 0.8, -0.1, 1.0],
                         [1.2, 0.0, -0.5, 0.7, -1.0, 0.9, 0.4, -0.3, 2.5, -2.0]], dtype=np.float32)
    labels = np.array([5, 8], dtype=np.int64)
    input_dict = {"name": "case_float32_multiclass_10", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float64, 1x1 trivial
    features = np.array([[0.0]], dtype=np.float64)
    labels = np.array([0], dtype=np.int32)
    input_dict = {"name": "case_float64_trivial_1x1", "features": features, "labels": labels}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: bfloat16 if available, else float32 fallback
    if hasattr(np, "bfloat16"):
        features = np.array([[1.0, -1.0, 0.0],
                             [0.5, 0.5, -1.0],
                             [-2.0, 1.0, 3.0]], dtype=np.bfloat16)
        labels = np.array([0, 2, 1], dtype=np.int32)
        input_dict = {"name": "case_bfloat16_available", "features": features, "labels": labels}
        list_of_inputs.append(copy.deepcopy(input_dict))
    else:
        features = np.array([[1.0, -1.0, 0.0],
                             [0.5, 0.5, -1.0],
                             [-2.0, 1.0, 3.0]], dtype=np.float32)
        labels = np.array([0, 2, 1], dtype=np.int32)
        input_dict = {"name": "case_float32_bfloat16_fallback", "features": features, "labels": labels}
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits"] = tf_raw_ops_SparseSoftmaxCrossEntropyWithLogits_inputs()

import tensorflow as tf
import numpy as np
import copy

np.random.seed(42)

def tf_raw_ops_squared_difference_inputs():
    list_of_inputs = []

    x = np.array([1.0, -2.5, 3.3], dtype=np.float32)
    y = np.array([0.5, 2.0, -3.3], dtype=np.float32)
    input_dict = {"name": "case1_f32_1d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.int32)
    y = np.array([[0, 2, -3], [5, -7, 9]], dtype=np.int32)
    input_dict = {"name": "case2_i32_2d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(3.14, dtype=np.float64)
    y = np.array([[[1.0, -2.0, 0.0], [4.0, -5.5, 6.6]]], dtype=np.float64)
    input_dict = {"name": "case3_f64_scalar_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, -1.5, 2.0, -2.5],
                  [3.0, -3.5, 4.0, -4.5],
                  [5.0, -5.5, 6.0, -6.5]], dtype=np.float16)
    y = np.array([0.5, -0.5, 1.5, -1.5], dtype=np.float16)
    input_dict = {"name": "case4_f16_broadcast_2d_1d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1]], [[-2]]], dtype=np.int64)
    y = np.array([[[3, -4, 5, -6],
                   [7, -8, 9, -10],
                   [11, -12, 13, -14]]], dtype=np.int64)
    input_dict = {"name": "case5_i64_broadcast_3d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1+2j, -3+0.5j, 0-1j, 2-2j], dtype=np.complex64)
    y = np.array(1-1j, dtype=np.complex64)
    input_dict = {"name": "case6_c64_scalar_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1+1j, -2-3j], [4-5j, -6+7j]], dtype=np.complex128)
    y = np.array([[0-1j, 2+3j], [-4+5j, 6-7j]], dtype=np.complex128)
    input_dict = {"name": "case7_c128_2d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.empty((0, 3), dtype=np.float32)
    y = np.empty((0, 3), dtype=np.float32)
    input_dict = {"name": "case8_f32_zerosized", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.arange(1*2*3*1, dtype=np.float32).reshape(1, 2, 3, 1)
    y = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    input_dict = {"name": "case9_f32_4d_2d_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-7, dtype=np.int32)
    y = np.array(5, dtype=np.int32)
    input_dict = {"name": "case10_i32_scalar", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.inf, -np.inf, np.nan], dtype=np.float64)
    y = np.array([1.0, -2.0, 0.0], dtype=np.float64)
    input_dict = {"name": "case11_f64_nan_inf", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex64)
    y = np.array([1+0j, 0+1j, -1+0j, 0-1j], dtype=np.complex64)
    input_dict = {"name": "case12_c64_3d_1d_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SquaredDifference"] = tf_raw_ops_squared_difference_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_reverse_inputs():
    list_of_inputs = []

    tensor = np.array([1, 2, 3, 4], dtype=np.int32)
    axis = np.array([0], dtype=np.int32)
    name = "rev_int32_1d_axis0"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    axis = np.array([-1], dtype=np.int64)
    name = "rev_float32_2d_last"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    axis = np.array([0, 2], dtype=np.int32)
    name = "rev_int64_3d_axes0_2"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.array([[[True, False], [False, True]]], dtype=bool)
    axis = np.array([-2], dtype=np.int64)
    name = "rev_bool_3d_axis_minus2"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = (np.arange(6).reshape(2, 3).astype(np.float32) + 1j * np.arange(6).reshape(2, 3).astype(np.float32)).astype(np.complex64)
    axis = np.array([], dtype=np.int32)
    name = "rev_complex64_2d_noaxis"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    axis = np.array([1], dtype=np.int32)
    name = "rev_uint8_3d_axis1"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(12, dtype=np.float16).reshape(1, 1, 2, 2, 1, 3)
    axis = np.array([5], dtype=np.int64)
    name = "rev_float16_6d_axis5"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(8, dtype=np.int8).reshape(1, 1, 1, 2, 2, 1, 2)
    axis = np.array([0, -3, 6], dtype=np.int64)
    name = "rev_int8_7d_axes0_minus3_6"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    shape = (1, 1, 1, 2, 1, 2, 1, 2)
    tensor = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    axis = np.array([-8, 2, -1], dtype=np.int64)
    name = "rev_float64_8d_axes_minus8_2_minus1"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    axis = np.array([1], dtype=np.int32)
    name = "rev_int16_3d_axis1"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    return list_of_inputs

generated_inputs["tf.reverse_1"] = tf_reverse_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_reverse_2_inputs():
    list_of_inputs = []

    tensor = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    axis = [0]
    name = "rev_1d_int32"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=np.float32)
    axis = [1]
    name = "rev_2d_float32_axis1"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.array([[[True, False, True],
                        [False, False, True]],
                       [[True, True, False],
                        [False, True, False]]], dtype=np.bool_)
    axis = [-1]
    name = "rev_3d_bool_last"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(1*2*3*4, dtype=np.int32).reshape(1, 2, 3, 4)
    axis = [1, 3]
    name = "rev_4d_int32_multi"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(1*1*2*2*3, dtype=np.float32).reshape(1, 1, 2, 2, 3)
    axis = [-3]
    name = "rev_5d_float32_neg"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.array(42, dtype=np.int64)
    axis = []
    name = "rev_scalar_int64_empty_axis"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(12, dtype=np.int64).reshape(2, 2, 3)
    axis = [2, 0]
    name = "rev_3d_int64_multi"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.empty((3, 0, 4), dtype=np.float64)
    axis = [1]
    name = "rev_empty_dim_float64"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.array([10, -3, 7, 0, 5, -8, 2, 9], dtype=np.int32)
    axis = [-1]
    name = "rev_1d_int32_negaxis"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(2*3*4*1*2*1, dtype=np.int32).reshape(2, 3, 4, 1, 2, 1)
    axis = [0, 2, 5]
    name = "rev_6d_int32_multi"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(1*1*1*2*1*2*1, dtype=np.int32).reshape(1, 1, 1, 2, 1, 2, 1)
    axis = [3, 5]
    name = "rev_7d_int32_axes"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.array([[1.0, 2.0],
                       [3.0, 4.0]], dtype=np.float64)
    axis = []
    name = "rev_2d_float64_noop"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(1*2*1*2*1*2*1*2, dtype=np.int64).reshape(1, 2, 1, 2, 1, 2, 1, 2)
    axis = [0, 2, 4, 6]
    name = "rev_8d_int64_even_axes"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(2*3*1*5, dtype=np.float32).reshape(2, 3, 1, 5)
    axis = [-4, -1]
    name = "rev_4d_float32_negaxes"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    return list_of_inputs

generated_inputs["tf.reverse_2"] = tf_reverse_2_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_reverse_inputs():
    list_of_inputs = []

    tensor = np.arange(10, dtype=np.int32)
    axis = (0,)
    name = "rev_case_1_int32_1d"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    axis = (-1,)
    name = "rev_case_2_float32_2d_last_axis"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    axis = (1, 2)
    name = "rev_case_3_int64_3d_axes_1_2"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(24, dtype=np.float64).reshape(1, 2, 3, 4)
    axis = (3,)
    name = "rev_case_4_float64_4d_axis3"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.array(42.5, dtype=np.float64)
    axis = ()
    name = "rev_case_5_scalar_noop"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = (np.arange(2 * 1 * 2 * 1 * 2) % 2 == 0).reshape(2, 1, 2, 1, 2)
    axis = (0, 3)
    name = "rev_case_6_bool_5d_axes_0_3"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(2 * 1 * 2 * 1 * 2 * 1, dtype=np.float32).reshape(2, 1, 2, 1, 2, 1)
    axis = (0, 2, 4)
    name = "rev_case_7_float32_6d_axes_0_2_4"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(1 * 1 * 2 * 1 * 2 * 1 * 3, dtype=np.int32).reshape(1, 1, 2, 1, 2, 1, 3)
    axis = (-7, -1)
    name = "rev_case_8_int32_7d_neg_axes"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(2, dtype=np.int64).reshape(1, 1, 1, 1, 1, 1, 1, 2)
    axis = (-1,)
    name = "rev_case_9_int64_8d_last_axis"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = (np.arange(24) % 3 == 0).reshape(2, 3, 4)
    axis = (-3, -2, -1)
    name = "rev_case_10_bool_3d_all_axes"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(20, dtype=np.float64).reshape(5, 4)
    axis = (0, 1)
    name = "rev_case_11_float64_2d_both_axes"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    tensor = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    axis = (1,)
    name = "rev_case_12_float32_3d_axis1"
    list_of_inputs.append(copy.deepcopy({"tensor": tensor, "axis": axis, "name": name}))

    return list_of_inputs

generated_inputs["tf.reverse_3"] = tf_reverse_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sets_intersection_inputs():
    list_of_inputs = []

    a = np.array([[1, 2, 3], [-1, -2, -3]], dtype=np.int32)
    b = np.array([[3, 4, 2, 2], [-3, -4, -2, -2]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": True}))

    a = np.array([[[1, 2, 3], [4, 5, 6]],
                  [[7, 8, 9], [10, 11, 12]]], dtype=np.int64)
    b = np.array([[[2, 4], [4, 1]],
                  [[9, 7], [12, 0]]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": False}))

    a = np.empty((2, 0), dtype=np.int32)
    b = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": True}))

    a = np.array([[1, 1, 1, 2, 2, 3],
                  [4, 4, 4, 4, 5, 5]], dtype=np.int32)
    b = np.array([[1, 2, 2, 2, 5],
                  [4, 6, 4, 5, 4]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": False}))

    a = np.array([[[[1, 2, 3], [4, 5, 6]]],
                  [[[7, 8, 9], [10, 11, 12]]]], dtype=np.int32)
    b = np.array([[[[3, 1], [6, 0]]],
                  [[[9, 0], [12, 10]]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": True}))

    a = np.array([[10],
                  [20],
                  [30],
                  [40]], dtype=np.int32)
    b = np.array([[5, 10],
                  [20, 21],
                  [0, 1],
                  [40, 40]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": False}))

    a = np.empty((1, 3, 0), dtype=np.int64)
    b = np.array([[[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": True}))

    a = np.array([[[-5, -1, 0, 3, 9],
                   [2, -2, -2, 8, 8],
                   [100, -100, 50, 0, 1]]], dtype=np.int64)
    b = np.array([[[0, -1, 7],
                   [-2, 2, 3],
                   [50, 1, -999]]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": True}))

    a = np.array([[2147483647, -2147483647, 123, 456],
                  [0, -1, 2147483646, -2147483647]], dtype=np.int32)
    b = np.array([[2147483647, 0, 999, -2147483647],
                  [-1, 2147483646, -2147483646, 42]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": False}))

    a = np.array([[[1, 2, 2, 3, 4],
                   [5, 6, 6, 7, 8]],
                  [[9, 10, 10, 11, 12],
                   [13, 14, 14, 15, 16]],
                  [[-1, -2, -2, -3, -4],
                   [0, 1, 1, 2, 3]]], dtype=np.int32)
    b = np.array([[[2, 3, 9],
                   [6, 5, 10]],
                  [[10, 11, 0],
                   [14, 15, 100]],
                  [[-2, -4, -5],
                   [1, 3, 4]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": True}))

    a = np.random.randint(-50, 50, size=(5, 5, 10), dtype=np.int32)
    b = np.random.randint(-50, 50, size=(5, 5, 8), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b, "validate_indices": False}))

    return list_of_inputs

generated_inputs["tf.sets.intersection"] = tf_sets_intersection_inputs()

import numpy as np
import tensorflow as tf
import copy

def tf_sparse_SparseTensor_inputs():
    list_of_inputs = []

    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([1, 2], dtype=np.int32)
    dense_shape = np.array([3, 4], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[3], [1], [5]], dtype=np.int64)
    values = np.array([10.0, -2.0, 5.0], dtype=np.float32)
    dense_shape = np.array([7], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 0, 0], [1, 2, 3], [1, 0, 2]], dtype=np.int64)
    values = np.array([0.5, -1.25, 3.0], dtype=np.float64)
    dense_shape = np.array([2, 3, 4], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 1]], dtype=np.int64)
    values = np.array([True], dtype=bool)
    dense_shape = np.array([2, 2], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0], [2]], dtype=np.int64)
    values = np.array([1 + 2j, -3 + 0.5j], dtype=np.complex64)
    dense_shape = np.array([4], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.empty((0, 2), dtype=np.int64)
    values = np.empty((0,), dtype=np.float32)
    dense_shape = np.array([0, 5], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[1, 1], [1, 1], [0, 2]], dtype=np.int64)
    values = np.array([5, -3, 9], dtype=np.int64)
    dense_shape = np.array([3, 3], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 0, 0, 1], [1, 1, 1, 1]], dtype=np.int64)
    values = np.array([1.5, -2.5], dtype=np.float16)
    dense_shape = np.array([2, 2, 2, 2], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[999999]], dtype=np.int64)
    values = np.array([7], dtype=np.int8)
    dense_shape = np.array([1000000], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 0], [1, 2]], dtype=np.int64)
    values = np.array([255, 128], dtype=np.uint8)
    dense_shape = np.array([2, 3], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = np.array([[0, 0, 0, 0, 0]], dtype=np.int64)
    values = np.array([42], dtype=np.int16)
    dense_shape = np.array([1, 1, 1, 1, 1], dtype=np.int64)
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.SparseTensor_1"] = tf_sparse_SparseTensor_inputs()

import numpy as np
import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_SparseTensor_2_inputs():
    list_of_inputs = []

    indices = [[0, 0], [1, 2], [2, 3]]
    values = np.array([1.5, -2.0, 3.25], dtype=np.float32)
    dense_shape = [3, 4]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0], [3]]
    values = np.array([True, True], dtype=bool)
    dense_shape = [5]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0, 0, 1], [1, 2, 3]]
    values = np.array([-10, 20], dtype=np.int32)
    dense_shape = [2, 3, 4]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0, 1, 2], [1, 0, 0], [1, 2, 1], [0, 0, 0]]
    values = np.array([0.0, -3.14, 2.71828, 1.0], dtype=np.float64)
    dense_shape = [2, 3, 3]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0, 0], [2, 2]]
    values = np.array([2**33, -(2**33)], dtype=np.int64)
    dense_shape = [3, 3]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0]]
    values = np.array([np.nan], dtype=np.float32)
    dense_shape = [1]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 2, 1], [2, 0, 2]]
    values = np.array([np.inf, -np.inf, 0.0, 7.0, -1.0], dtype=np.float32)
    dense_shape = [3, 3, 3]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0, 1], [1, 2]]
    values = np.array([255, 128], dtype=np.uint8)
    dense_shape = [2, 3]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0, 0], [1, 1], [2, 2]]
    values = np.array([1.5, -2.5, 3.5], dtype=np.float16)
    dense_shape = [3, 3]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    indices = [[0, 0, 0, 0], [1, 2, 0, 3], [1, 0, 0, 1], [0, 1, 0, 2]]
    values = np.array([-1000, 2000, 15, -7], dtype=np.int16)
    dense_shape = [2, 3, 1, 4]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.SparseTensor_2"] = tf_sparse_SparseTensor_2_inputs()

import tensorflow as tf
import numpy as np
import copy
import torch  # ensuring availability if required by environment

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_sparse_SparseTensor_inputs():
    list_of_inputs = []

    # Input 1: 2D int values
    indices = [[np.int64(0), np.int64(0)], [np.int64(1), np.int64(2)]]
    values = [np.int32(1), np.int32(2)]
    dense_shape = [np.int64(3), np.int64(4)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 values with negatives
    indices = [[np.int64(0)], [np.int64(4)], [np.int64(9)]]
    values = [np.float32(1.5), np.float32(-2.0), np.float32(3.14)]
    dense_shape = [np.int64(10)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D boolean values
    indices = [
        [np.int64(0), np.int64(1), np.int64(2)],
        [np.int64(1), np.int64(0), np.int64(3)],
        [np.int64(1), np.int64(2), np.int64(1)]
    ]
    values = [np.bool_(True), np.bool_(False), np.bool_(True)]
    dense_shape = [np.int64(2), np.int64(3), np.int64(4)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D float32 values
    indices = [
        [np.int64(0), np.int64(0), np.int64(0), np.int64(0)],
        [np.int64(1), np.int64(1), np.int64(1), np.int64(1)],
        [np.int64(0), np.int64(1), np.int64(1), np.int64(0)]
    ]
    values = [np.float32(2.5), np.float32(-3.0), np.float32(0.75)]
    dense_shape = [np.int64(2), np.int64(2), np.int64(2), np.int64(2)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float16 values, unsorted indices
    indices = [
        [np.int64(2), np.int64(2)],
        [np.int64(0), np.int64(1)],
        [np.int64(1), np.int64(0)]
    ]
    values = [np.float16(0.1), np.float16(-0.2), np.float16(1.5)]
    dense_shape = [np.int64(3), np.int64(3)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D int64 values with negatives
    indices = [
        [np.int64(0), np.int64(0), np.int64(0)],
        [np.int64(2), np.int64(2), np.int64(2)],
        [np.int64(1), np.int64(1), np.int64(1)],
        [np.int64(0), np.int64(2), np.int64(1)]
    ]
    values = [np.int64(-1), np.int64(0), np.int64(2), np.int64(-5)]
    dense_shape = [np.int64(3), np.int64(3), np.int64(3)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D boolean values
    indices = [[np.int64(1)], [np.int64(3)]]
    values = [np.bool_(True), np.bool_(False)]
    dense_shape = [np.int64(5)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float64 values
    indices = [
        [np.int64(0), np.int64(0), np.int64(0)],
        [np.int64(0), np.int64(1), np.int64(2)],
        [np.int64(1), np.int64(1), np.int64(1)]
    ]
    values = [np.float64(0.0), np.float64(4.2), np.float64(5.5)]
    dense_shape = [np.int64(2), np.int64(2), np.int64(3)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D int16 values
    indices = [
        [np.int64(3), np.int64(2), np.int64(1), np.int64(4)],
        [np.int64(0), np.int64(0), np.int64(0), np.int64(0)],
        [np.int64(2), np.int64(1), np.int64(0), np.int64(3)]
    ]
    values = [np.int16(7), np.int16(-8), np.int16(15)]
    dense_shape = [np.int64(4), np.int64(3), np.int64(2), np.int64(5)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D uint8 values
    indices = [
        [np.int64(0), np.int64(0)],
        [np.int64(0), np.int64(1)],
        [np.int64(0), np.int64(3)],
        [np.int64(1), np.int64(2)]
    ]
    values = [np.uint8(255), np.uint8(0), np.uint8(128), np.uint8(64)]
    dense_shape = [np.int64(2), np.int64(4)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 5D float32 values
    indices = [
        [np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0)],
        [np.int64(0), np.int64(1), np.int64(0), np.int64(1), np.int64(0)]
    ]
    values = [np.float32(0.25), np.float32(-1.75)]
    dense_shape = [np.int64(1), np.int64(2), np.int64(1), np.int64(2), np.int64(1)]
    input_dict = {"indices": indices, "values": values, "dense_shape": dense_shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.SparseTensor_3"] = tf_sparse_SparseTensor_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_sparse_eye_inputs():
    list_of_inputs = []

    # Input 1
    num_rows = np.int32(0)
    num_columns = np.int32(0)
    dtype = np.float32
    name = "eye_zero_zero"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    num_rows = np.int32(1)
    num_columns = np.int32(1)
    dtype = np.int32
    name = "eye_one"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    num_rows = np.int64(3)
    num_columns = np.int64(5)
    dtype = np.float64
    name = "eye_rect64"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    num_rows = np.int32(5)
    num_columns = np.int32(3)
    dtype = np.float32
    name = "eye_rect"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    num_rows = np.int64(10)
    num_columns = np.int64(10)
    dtype = np.int64
    name = "eye_int64"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    num_rows = np.int32(7)
    num_columns = np.int32(7)
    dtype = np.complex64
    name = "eye_c64"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    num_rows = np.int32(4)
    num_columns = np.int32(4)
    dtype = np.complex128
    name = "eye_c128"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    num_rows = np.int32(2)
    num_columns = np.int32(8)
    dtype = np.bool_
    name = "eye_bool"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    num_rows = np.int32(9)
    num_columns = np.int32(9)
    dtype = np.float16
    name = "eye_f16"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    num_rows = np.int64(12)
    num_columns = np.int64(15)
    dtype = np.float32
    name = "eye_12_15"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    num_rows = np.int32(100)
    num_columns = np.int32(100)
    dtype = np.float32
    name = "eye_100"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    num_rows = np.int32(3)
    num_columns = np.int32(3)
    dtype = np.uint8
    name = "eye_uint8"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.eye_1"] = tf_sparse_eye_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_eye_inputs():
    list_of_inputs = []

    # Input 1
    num_rows = 0
    num_columns = np.array(0, dtype=np.int32)
    dtype = np.float32
    name = "eye_zero_square"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    num_rows = 1
    num_columns = np.array(1, dtype=np.int32)
    dtype = np.int32
    name = "eye_one_int"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    num_rows = 3
    num_columns = np.array(3, dtype=np.int32)
    dtype = np.float64
    name = "eye_three_float64"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    num_rows = 4
    num_columns = np.array(2, dtype=np.int32)
    dtype = np.float16
    name = "eye_rect_tall_f16"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    num_rows = 2
    num_columns = np.array(5, dtype=np.int32)
    dtype = np.bool_
    name = "eye_rect_wide_bool"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    num_rows = 6
    num_columns = np.array(6, dtype=np.int32)
    dtype = np.complex64
    name = "eye_complex64"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    num_rows = 7
    num_columns = np.array(10, dtype=np.int32)
    dtype = np.int8
    name = "eye_int8"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    num_rows = 10
    num_columns = np.array(7, dtype=np.int32)
    dtype = np.int64
    name = "eye_int64_rect"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    num_rows = 5
    num_columns = np.array(5, dtype=np.int32)
    dtype = np.complex128
    name = "eye_complex128"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    num_rows = 8
    num_columns = np.array(8, dtype=np.int32)
    dtype = np.uint8
    name = "eye_uint8"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    num_rows = 12
    num_columns = np.array(0, dtype=np.int32)
    dtype = np.float32
    name = "eye_zero_cols"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    num_rows = 0
    num_columns = np.array(9, dtype=np.int32)
    dtype = np.float32
    name = "eye_zero_rows"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.eye_2"] = tf_sparse_eye_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_eye_inputs():
    list_of_inputs = []

    # Input 1
    num_rows = np.array(5, dtype=np.int32)
    num_columns = 5
    dtype = np.float32
    name = "eye_square_f32"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    num_rows = np.array(0, dtype=np.int32)
    num_columns = 0
    dtype = np.float64
    name = "eye_empty"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    num_rows = np.array(7, dtype=np.int64)
    num_columns = 3
    dtype = np.int32
    name = "eye_tall_int32"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    num_rows = np.array(3, dtype=np.int64)
    num_columns = 7
    dtype = np.int64
    name = "eye_wide_int64"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    num_rows = np.array(1, dtype=np.int32)
    num_columns = 1
    dtype = np.bool_
    name = "eye_bool_1x1"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    num_rows = np.array(4, dtype=np.int64)
    num_columns = 4
    dtype = np.complex64
    name = "eye_c64_4x4"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    num_rows = np.array(2, dtype=np.int32)
    num_columns = 5
    dtype = np.complex128
    name = "eye_c128_2x5"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    num_rows = np.array(10, dtype=np.int64)
    num_columns = 10
    dtype = np.float16
    name = "eye_f16_10x10"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    num_rows = np.array(6, dtype=np.int32)
    num_columns = 4
    dtype = np.uint8
    name = "eye_uint8_6x4"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    num_rows = np.array(8, dtype=np.int64)
    num_columns = 8
    dtype = np.float64
    name = "eye_double"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    num_rows = np.array(12, dtype=np.int64)
    num_columns = 0
    dtype = np.int16
    name = "eye_zero_cols"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    num_rows = np.array(0, dtype=np.int32)
    num_columns = 5
    dtype = np.float32
    name = "eye_zero_rows"
    input_dict = {"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sparse.eye_3"] = tf_sparse_eye_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_sparse_eye_inputs():
    list_of_inputs = []

    num_rows = np.array(3, dtype=np.int32)
    num_columns = np.array(3, dtype=np.int32)
    dtype = np.float32
    name = "eye_f32_3x3"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(5, dtype=np.int64)
    num_columns = np.array(7, dtype=np.int64)
    dtype = np.float64
    name = "eye_f64_5x7"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(0, dtype=np.int32)
    num_columns = np.array(0, dtype=np.int32)
    dtype = np.float32
    name = "eye_zero_zero"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(4, dtype=np.int64)
    num_columns = np.array(0, dtype=np.int64)
    dtype = np.int32
    name = "eye_i32_4x0"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(0, dtype=np.int32)
    num_columns = np.array(6, dtype=np.int32)
    dtype = np.int64
    name = "eye_i64_0x6"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(1, dtype=np.int64)
    num_columns = np.array(1, dtype=np.int64)
    dtype = np.complex64
    name = "eye_c64_1x1"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(2, dtype=np.int32)
    num_columns = np.array(3, dtype=np.int32)
    dtype = np.float16
    name = "eye_f16_2x3"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(3, dtype=np.int64)
    num_columns = np.array(2, dtype=np.int64)
    dtype = np.complex128
    name = "eye_c128_3x2"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(10, dtype=np.int32)
    num_columns = np.array(10, dtype=np.int32)
    dtype = np.uint8
    name = "eye_u8_10x10"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(8, dtype=np.int64)
    num_columns = np.array(8, dtype=np.int64)
    dtype = np.float32
    name = "éye_unicode_8x8"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(12, dtype=np.int32)
    num_columns = np.array(15, dtype=np.int32)
    dtype = np.float64
    name = "eye_rect_f64_12x15"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    num_rows = np.array(100, dtype=np.int64)
    num_columns = np.array(100, dtype=np.int64)
    dtype = np.float32
    name = "eye_large_100x100"
    list_of_inputs.append(copy.deepcopy({"num_rows": num_rows, "num_columns": num_columns, "dtype": dtype, "name": name}))

    return list_of_inputs

generated_inputs["tf.sparse.eye_4"] = tf_sparse_eye_inputs()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_strided_slice_inputs():
    list_of_inputs = []

    # Input 1: basic 1D stride
    input_ = np.arange(10, dtype=np.int32)
    begin = np.array([2], dtype=np.int32)
    end = np.array([8], dtype=np.int32)
    strides = np.array([2], dtype=np.int32)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "basic_1d_stride2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D with column step
    input_ = np.arange(20, dtype=np.float32).reshape(4, 5)
    begin = np.array([1, 0], dtype=np.int64)
    end = np.array([4, 5], dtype=np.int64)
    strides = np.array([1, 2], dtype=np.int64)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "2d_step2_cols"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: reverse 1D using negative stride
    input_ = np.arange(8, dtype=np.int64)
    begin = np.array([-1], dtype=np.int64)
    end = np.array([-9], dtype=np.int64)
    strides = np.array([-1], dtype=np.int64)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "reverse_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D with begin_mask and end_mask
    input_ = np.arange(3 * 4 * 5, dtype=np.int32).reshape(3, 4, 5)
    begin = np.array([0, 1, 0], dtype=np.int32)
    end = np.array([0, 4, 5], dtype=np.int32)
    strides = np.array([1, 1, 2], dtype=np.int32)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 1,  # ignore begin[0]
        "end_mask": 1,    # ignore end[0]
        "ellipsis_mask": 0,
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": np.zeros_like(input_),
        "name": "3d_begin_end_mask"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D with ellipsis in the middle
    input_ = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)
    begin = np.array([0, 1, 0], dtype=np.int32)
    end = np.array([2, 3, 5], dtype=np.int32)
    strides = np.array([1, 1, 1], dtype=np.int32)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 1 << 1,  # ellipsis at spec index 1
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "4d_ellipsis_middle"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: new axis inserted between dims
    input_ = np.arange(12, dtype=np.int16).reshape(3, 4)
    begin = np.array([0, 0, 0], dtype=np.int64)
    end = np.array([3, 0, 4], dtype=np.int64)
    strides = np.array([1, 1, 1], dtype=np.int64)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": 1 << 1,  # add new axis at spec index 1
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "new_axis_between"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: shrink axis in the middle
    input_ = (np.arange(3 * 4 * 5, dtype=np.float16).reshape(3, 4, 5))
    begin = np.array([0, 2, 0], dtype=np.int32)
    end = np.array([0, 0, 0], dtype=np.int32)
    strides = np.array([1, 1, 1], dtype=np.int32)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": 0,
        "shrink_axis_mask": 1 << 1,  # shrink spec index 1
        "var": input_.copy(),
        "name": "shrink_axis_mid"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D with negative stride on middle dim
    input_ = np.arange(5 * 6 * 7, dtype=np.int32).reshape(5, 6, 7)
    begin = np.array([1, 5, 0], dtype=np.int64)
    end = np.array([4, 1, 7], dtype=np.int64)
    strides = np.array([1, -2, 1], dtype=np.int64)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "3d_negative_stride_middle"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: shrink first spec and add two new axes
    input_ = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    begin = np.array([2, 0, 0], dtype=np.int32)
    end = np.array([0, 0, 0], dtype=np.int32)
    strides = np.array([1, 1, 1], dtype=np.int32)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": (1 << 1) | (1 << 2),  # new axes at specs 1 and 2
        "shrink_axis_mask": 1 << 0,            # shrink spec 0
        "var": input_.copy(),
        "name": "shrink_then_newaxes"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: ellipsis at first spec on 5D tensor
    input_ = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.int8).reshape(2, 3, 4, 5, 6)
    begin = np.array([0, 1], dtype=np.int64)
    end = np.array([0, 5], dtype=np.int64)
    strides = np.array([1, 2], dtype=np.int64)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 1 << 0,  # ellipsis at spec index 0
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "ellipsis_first"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: boolean 2D with strides
    input_ = (np.arange(9).reshape(3, 3) % 2 == 0)
    begin = np.array([0, 0], dtype=np.int32)
    end = np.array([3, 3], dtype=np.int32)
    strides = np.array([2, 1], dtype=np.int32)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "bool_2d_stride"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 2D with negative indices
    input_ = np.arange(7 * 8, dtype=np.int32).reshape(7, 8)
    begin = np.array([5, -8], dtype=np.int64)
    end = np.array([7, -3], dtype=np.int64)
    strides = np.array([1, 1], dtype=np.int64)
    input_dict = {
        "input_": input_,
        "begin": begin,
        "end": end,
        "strides": strides,
        "begin_mask": 0,
        "end_mask": 0,
        "ellipsis_mask": 0,
        "new_axis_mask": 0,
        "shrink_axis_mask": 0,
        "var": input_.copy(),
        "name": "negative_indices_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.strided_slice"] = tf_strided_slice_inputs()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_unicode_script_inputs():
    list_of_inputs = []

    arr = np.array([1, 31, 38], dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[65, 97, 48], [90, 122, 57]], dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array(
        [
            [[233, 1046, 937], [1569, 1488, 2309]],
            [[4352, 12354, 20013], [12452, 65313, 0x1F600]],
        ],
        dtype=np.int32,
    )
    input_dict = {"input": arr, "name": "unicode_script_case_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([-1, -100, 1114112, 2000000], dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_4_invalids"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([0, 127, 128512, 1114111], dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_5_boundaries"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array(128169, dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_6_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([], dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_7_empty_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.empty((0, 3), dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_8_empty_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([55296, 56320, 57343, 57344], dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_9_surrogates_pua"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[[[9, 945, 1040, 44032]], [[19968, 20108, 20225, 20320]]]], dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_10_4d_mixed"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[769, 8205, 1632], [8212, 8226, 8364]], dtype=np.int32)
    input_dict = {"input": arr, "name": "unicode_script_case_11_combining_symbols"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array(
        [
            [[-5, 65], [0, 55297], [1114111, 128175]],
            [[173, 180], [156, 711], [701, 7424]],
        ],
        dtype=np.int32,
    )
    input_dict = {"input": arr, "name": "unicode_script_case_12_3d_varied"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.strings.unicode_script"] = tf_strings_unicode_script_inputs()

