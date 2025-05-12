import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    dtype = input_dict["input"]
    
    if not cpu:
        pass
    
    if dtype == np.int8:
        torch_dtype = torch.int8
    elif dtype == np.int16:
        torch_dtype = torch.int16
    elif dtype == np.int32:
        torch_dtype = torch.int32
    elif dtype == np.int64:
        torch_dtype = torch.int64
    elif dtype == np.uint8:
        torch_dtype = torch.uint8
    elif dtype == np.uint16:
        torch_dtype = torch.uint16
    elif dtype == np.uint32:
        torch_dtype = torch.uint32
    elif dtype == np.uint64:
        torch_dtype = torch.uint64
    else:
        raise ValueError("Unsupported numpy dtype: {}".format(dtype))

    result = torch.iinfo(torch_dtype)
    
    if not cpu:
        pass
    
    return {
        "bits": result.bits,
        "max": result.max,
        "min": result.min
    }

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    dtype = input_dict["input"]

    if dtype == np.int8:
        tf_dtype = tf.int8
    elif dtype == np.int16:
        tf_dtype = tf.int16
    elif dtype == np.int32:
        tf_dtype = tf.int32
    elif dtype == np.int64:
        tf_dtype = tf.int64
    elif dtype == np.uint8:
        tf_dtype = tf.uint8
    elif dtype == np.uint16:
        tf_dtype = tf.uint16
    elif dtype == np.uint32:
        if hasattr(tf, 'uint32'):
            tf_dtype = tf.uint32
        else:
            info = np.iinfo(dtype)
            return {
                "bits": info.bits,
                "max": info.max,
                "min": info.min
            }
    elif dtype == np.uint64:
        if hasattr(tf, 'uint64'):
            tf_dtype = tf.uint64
        else:
            info = np.iinfo(dtype)
            return {
                "bits": info.bits,
                "max": info.max,
                "min": info.min
            }
    else:
        raise ValueError("Unsupported numpy dtype: {}".format(dtype))

    try:
        info = tf.experimental.numpy.iinfo(tf_dtype)
    except (AttributeError, ValueError):
        info = np.iinfo(dtype)
        return {
            "bits": info.bits,
            "max": info.max,
            "min": info.min
        }

    return {
        "bits": info.bits,
        "max": info.max,
        "min": info.min
    }

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.int32
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["bits"] == tf_result["bits"], "Bits do not match"
    assert torch_result["max"] == tf_result["max"], "Max values do not match"
    assert torch_result["min"] == tf_result["min"], "Min values do not match"

    print("Success")

if __name__ == "__main__":
    main()