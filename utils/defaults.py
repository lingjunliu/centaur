import numpy as np
############### default values ################

# Currently supported: tensor, tensor_list (which is just tensor for now), integer, float, boolean, string, tuple, list, dtype
supported_paramtypes = [
    "tensor",
    "tensor_list",
    "integer",
    "float",
    "boolean",
    "string",
    "tuple",
    "list",
    "dtype"
]

MAX_N_DIM=6
MAX_SZ_DIM=100
# MAX_SZ_NUM=10000 # old
MAX_SZ_NUM=np.iinfo(np.int64).max # new, experimental
# MAX_SZ_FLT=10000000.0 # old
MAX_SZ_FLT=MAX_SZ_NUM
MAX_SZ_LST=10
MAX_SZ_TENSOR=256 # MB

list_of_available_dtypes = [bool, np.int8, np.int16, np.int32, np.int64, np.uint8, np.float16, np.float32, np.float64, np.complex64, np.complex128, str, np.dtype]
int_buckets = [0, MAX_N_DIM, MAX_SZ_DIM, np.iinfo(np.int8).max, np.iinfo(np.int16).max, np.iinfo(np.int32).max, np.iinfo(np.int64).max]
float_buckets = [0, 1.0, MAX_N_DIM, MAX_SZ_DIM, np.finfo(np.float16).max, np.finfo(np.float32).max, np.finfo(np.float64).max]

# For PyTorch
list_of_string_values_torch = [
    "ii", "ii->i", "i,j->ij", "bij,bjk->bik", "...ij->...ji", "bn,anm,bm->ba",
    "none", "mean", "sum", "max", "min",
    "relu", "tanh", "sigmoid", "softmax", "elu", "selu", "gelu", "swish", "softplus", "linear",
    "constant", "reflect", "replicate", "circular",
    "nearest", "bilinear", "bicubic", "trilinear", "area", "linear"
]

# For TensorFlow
list_of_string_values_tf = [
    "ii", "ii->i", "i,j->ij", "bij,bjk->bik", "...ij->...ji", "bn,anm,bm->ba",
    "none", "sum", "max", "min", "prod",
    "relu", "tanh", "sigmoid", "softmax", "elu", "selu", "gelu", "swish", "softplus", "linear",
    "valid", "same", "causal",
    "channels_last", "channels_first",
    "channels_last", "channels_first",
    "VALID", "SAME",       # padding (conv ops) — TF kernel requires uppercase
    "EXPLICIT",
    "NWC", "NCW",          # data_format (conv1d)
    "NHWC", "NCHW",        # data_format (conv2d/conv3d)
    "NDHWC", "NCDHW",      # data_format (conv3d volumetric)
    "NCHW_VECT_C",
    "REFLECT", "SYMMETRIC",
    "UTF-8", "UTF-16-BE", "UTF-32-BE",
    "replace", "ignore", "strict",
    "ab,bc->ac", "i,i->", "ij->ji", "ii->", "ij->j", "ijk->", "...ij,...jk->...ik",
    "LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT",
    "quicksort", "stable",
]

domain_limits_torch = {
    'tensor': [0, MAX_SZ_DIM, 1, MAX_N_DIM],
    'tensor_dtype': [0, len(list_of_available_dtypes)-3, 1, 1], # except str
    'tensor_value_range': [-MAX_SZ_NUM, MAX_SZ_NUM, 2, 2],
    'integer': [-MAX_SZ_NUM, MAX_SZ_NUM, 1, 1],
    'integer_dtype': [1, 5, 1, 1], # only integer dtypes
    'integer_value_range': [-MAX_SZ_NUM, MAX_SZ_NUM, 2, 2], # the range should be equal to the range of the "integer" limits. we only need this for tensors actually
    'float': [-MAX_SZ_FLT, MAX_SZ_FLT, 1, 1],
    'float_dtype': [6, 8, 1, 1], # only float dtypes
    'float_value_range': [-MAX_SZ_FLT, MAX_SZ_FLT, 2, 2],
    'boolean': [False, True, 1, 1],
    'boolean_dtype': [0, 0, 1, 1], # only boolean
    'boolean_value_range': [False, True, 2, 2],
    'string': [0, len(list_of_string_values_torch)-2, 1, 1],
    'string_dtype': [len(list_of_available_dtypes)-2, len(list_of_available_dtypes)-2, 1, 1], # only string
    'string_value_range': [0, len(list_of_string_values_torch)-1, 2, 2],
    'tuple': [-MAX_SZ_DIM, MAX_SZ_DIM-1, 1, MAX_N_DIM],
    'tuple_dtype': [1, 5, 1, 1], # only integer dtypes
    'tuple_value_range': [-MAX_SZ_DIM, MAX_SZ_DIM-1, 2, 2],
    'list': [-MAX_SZ_NUM, MAX_SZ_NUM, 1, MAX_SZ_LST],
    'list_dtype': [1, 5, 1, 1], # only integer dtypes
    'list_value_range': [-MAX_SZ_NUM, MAX_SZ_NUM, 2, 2],
    'dtype': [0, len(list_of_available_dtypes)-2, 1, 1],
    'dtype_dtype': [len(list_of_available_dtypes)-1, len(list_of_available_dtypes)-1, 1, 1],   # only dtype
    'dtype_value_range': [0, len(list_of_available_dtypes)-2, 2, 2],
}

domain_limits_tf = {
    'tensor': [0, MAX_SZ_DIM, 1, MAX_N_DIM],
    'tensor_dtype': [0, len(list_of_available_dtypes)-3, 1, 1], # except str
    'tensor_value_range': [-MAX_SZ_NUM, MAX_SZ_NUM, 2, 2],
    'integer': [-MAX_SZ_NUM, MAX_SZ_NUM, 1, 1],
    'integer_dtype': [1, 5, 1, 1], # only integer dtypes
    'integer_value_range': [-MAX_SZ_NUM, MAX_SZ_NUM, 2, 2], # the range should be equal to the range of the "integer" limits. we only need this for tensors actually
    'float': [-MAX_SZ_FLT, MAX_SZ_FLT, 1, 1],
    'float_dtype': [6, 8, 1, 1], # only float dtypes
    'float_value_range': [-MAX_SZ_FLT, MAX_SZ_FLT, 2, 2],
    'boolean': [False, True, 1, 1],
    'boolean_dtype': [0, 0, 1, 1], # only boolean
    'boolean_value_range': [False, True, 2, 2],
    'string': [0, len(list_of_string_values_tf)-2, 1, 1],
    'string_dtype': [len(list_of_available_dtypes)-2, len(list_of_available_dtypes)-2, 1, 1], # only string
    'string_value_range': [0, len(list_of_string_values_tf)-1, 2, 2],
    'tuple': [-MAX_SZ_DIM, MAX_SZ_DIM-1, 1, MAX_N_DIM],
    'tuple_dtype': [1, 5, 1, 1], # only integer dtypes
    'tuple_value_range': [-MAX_SZ_DIM, MAX_SZ_DIM-1, 2, 2],
    'list': [-MAX_SZ_NUM, MAX_SZ_NUM, 1, MAX_SZ_LST],
    'list_dtype': [1, 5, 1, 1], # only integer dtypes
    'list_value_range': [-MAX_SZ_NUM, MAX_SZ_NUM, 2, 2],
    'dtype': [0, len(list_of_available_dtypes)-2, 1, 1],
    'dtype_dtype': [len(list_of_available_dtypes)-1, len(list_of_available_dtypes)-1, 1, 1],   # only dtype
    'dtype_value_range': [0, len(list_of_available_dtypes)-2, 2, 2],
}

def np_dtype(dtype):
    try:
        import torch
        torch_map = {
            torch.bool: bool,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.uint8: np.uint8,
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.complex64: np.complex64,
            torch.complex128: np.complex128,
        }
        if isinstance(dtype, torch.dtype):
            return torch_map.get(dtype, None)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf_map = {
            tf.bool: bool,
            tf.int8: np.int8,
            tf.int16: np.int16,
            tf.int32: np.int32,
            tf.int64: np.int64,
            tf.uint8: np.uint8,
            tf.float16: np.float16,
            tf.float32: np.float32,
            tf.float64: np.float64,
            tf.complex64: np.complex64,
            tf.complex128: np.complex128,
        }
        if isinstance(dtype, tf.dtypes.DType):
            return tf_map.get(dtype, None)
    except ImportError:
        pass

    return None
