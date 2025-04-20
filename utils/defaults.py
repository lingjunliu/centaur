import numpy as np
############### default values ################

MAX_N_DIM=6
MAX_SZ_DIM=100
MAX_SZ_NUM=10000
MAX_SZ_FLT=10000000.0
MAX_SZ_LST=10
MAX_SZ_TENSOR=256 # MB

list_of_available_dtypes = [bool, np.int8, np.int16, np.int32, np.int64, np.uint8, np.float16, np.float32, np.float64, np.complex64, np.complex128, str, np.dtype]

list_of_string_values = [
    "ii",
    "ii->i",
    "i,j->ij",
    "bij,bjk->bik",
    "...ij->...ji",
    "bn,anm,bm->ba",
    "none",
    "mean",
    "sum",
    "max",
    "constant",
    "tanh",
]

domain_limits = {
    'tensor': [0, MAX_SZ_DIM, 0, MAX_N_DIM],
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
    'string': [0, len(list_of_string_values)-2, 1, 1],
    'string_dtype': [len(list_of_available_dtypes)-2, len(list_of_available_dtypes)-2, 1, 1], # only string
    'string_value_range': [0, len(list_of_string_values)-1, 2, 2],
    'tuple': [-MAX_SZ_DIM, MAX_SZ_DIM-1, 0, MAX_N_DIM],
    'tuple_dtype': [1, 5, 1, 1], # only integer dtypes
    'tuple_value_range': [-MAX_SZ_DIM, MAX_SZ_DIM-1, 2, 2],
    'list': [-MAX_SZ_NUM, MAX_SZ_NUM, 0, MAX_SZ_LST],
    'list_dtype': [1, 5, 1, 1], # only integer dtypes
    'list_value_range': [-MAX_SZ_NUM, MAX_SZ_NUM, 2, 2],
    'dtype': [0, len(list_of_available_dtypes)-2, 1, 1],
    'dtype_dtype': [len(list_of_available_dtypes)-1, len(list_of_available_dtypes)-1, 1, 1],   # only dtype
    'dtype_value_range': [0, len(list_of_available_dtypes)-2, 2, 2],
}