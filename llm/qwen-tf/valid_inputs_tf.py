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

