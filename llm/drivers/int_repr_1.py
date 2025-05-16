import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    if input_tensor.dtype == torch.float32:
        result = input_tensor.view(torch.int32)
    elif input_tensor.dtype == torch.float64:
        result = input_tensor.view(torch.int64)
    elif input_tensor.dtype == torch.int8:
        result = input_tensor.view(torch.int8)
    elif input_tensor.dtype == torch.int16:
        result = input_tensor.view(torch.int16)
    elif input_tensor.dtype == torch.int32:
        result = input_tensor.view(torch.int32)
    elif input_tensor.dtype == torch.int64:
        result = input_tensor.view(torch.int64)
    elif input_tensor.dtype == torch.uint8:
        result = input_tensor.view(torch.uint8)
    else:
        raise ValueError(f"Unsupported dtype: {input_tensor.dtype}")
    
    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        
        if input_tensor.dtype == tf.float32:
            result = tf.bitcast(input_tensor, tf.int32)
        elif input_tensor.dtype == tf.float64:
            result = tf.bitcast(input_tensor, tf.int64)
        elif input_tensor.dtype == tf.int8:
            result = tf.bitcast(input_tensor, tf.int8)
        elif input_tensor.dtype == tf.int16:
            result = tf.bitcast(input_tensor, tf.int16)
        elif input_tensor.dtype == tf.int32:
            result = tf.bitcast(input_tensor, tf.int32)
        elif input_tensor.dtype == tf.int64:
            result = tf.bitcast(input_tensor, tf.int64)
        elif input_tensor.dtype == tf.uint8:
            result = tf.bitcast(input_tensor, tf.uint8)
        else:
            raise ValueError(f"Unsupported dtype: {input_tensor.dtype}")

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()