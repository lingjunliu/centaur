import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    dtype = input_dict.get("dtype", np.float16)

    if not cpu:
        torch.cuda.set_device(0)

    if isinstance(dtype, type(np.float16)):
        if dtype == np.float16:
            dtype = torch.float16
        elif dtype == np.float32:
            dtype = torch.float32
        else:
            raise ValueError("Unsupported numpy dtype for torch conversion")

    torch.set_autocast_dtype("cuda", dtype)

    return {"result": torch.get_autocast_dtype("cuda")}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    dtype = input_dict.get("dtype", np.float16)

    if dtype == np.float16:
        tf_dtype = tf.float16
        policy = "mixed_float16"
    elif dtype == np.float32:
        tf_dtype = tf.float32
        policy = "float32"
    else:
        raise ValueError("Unsupported numpy dtype for tensorflow conversion")

    class DummyModel(tf.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.dtype = tf_dtype

        @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
        def __call__(self, x):
            tf.keras.mixed_precision.set_global_policy(policy)
            return tf.cast(x, dtype=self.dtype)

    model = DummyModel()
    dummy_input = tf.constant(1.0, dtype=tf.float32)
    casted_input = model(dummy_input)
    return {"result": tf.keras.mixed_precision.global_policy().compute_dtype}

def main():
    A_TOL = 0.01
    
    input_data = {
        "dtype": np.float16
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_result_str = str(torch_result["result"])
    tf_result_str = str(tf_result["result"])

    if "float16" in torch_result_str and "float16" in tf_result_str:
        pass
    elif "bfloat16" in torch_result_str and "bfloat16" in tf_result_str:
        pass
    elif "float32" in torch_result_str and "float32" in tf_result_str:
        pass
    else:
        raise AssertionError("Results do not match")
    
    input_data = {
        "dtype": np.float32
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_result_str = str(torch_result["result"])
    tf_result_str = str(tf_result["result"])

    if "float32" in torch_result_str and "float32" in tf_result_str:
        pass
    else:
        raise AssertionError("Results do not match")

    print("Success")

if __name__ == "__main__":
    main()