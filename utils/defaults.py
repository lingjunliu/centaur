import numpy as np
############### default values ################

MAX_N_DIM=32
MAX_SZ_DIM=100
MAX_SZ_NUM=10000

list_of_available_dtypes = [bool, np.int8, np.int16, np.int32, np.int64, np.uint8, np.float16, np.float32, np.float64, np.complex64, np.complex128]

domain_limits = {
    'tensor': [0, MAX_SZ_DIM, 0, MAX_N_DIM],
    'tensor_dtype': [0, len(list_of_available_dtypes)-1, 1, 1],
    'tensor_value_range': [-MAX_SZ_NUM, MAX_SZ_NUM, 2, 2],
    'integer': [-MAX_SZ_NUM, MAX_SZ_NUM, 1, 1],
    'integer_dtype': [1, 5, 1, 1], # only integer dtypes
    'integer_value_range': [-MAX_SZ_NUM, MAX_SZ_NUM, 2, 2], # the range should be equal to the range of the "integer" limits. we only need this for tensors actually
}