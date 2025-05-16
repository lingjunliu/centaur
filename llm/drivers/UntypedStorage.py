import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    dtype = input_dict.get("dtype", torch.float64)
    device = input_dict.get("device", torch.device('cpu') if cpu else torch.device('cuda'))
    pin_memory = input_dict.get("pin_memory", False)
    size = input_dict["size"]
    
    storage = torch.UntypedStorage(size, device=device)
    
    data = []
    for i in range(size):
        data.append(float(0))
    
    tensor = torch.tensor(data, dtype=dtype)

    return {"result": tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import numpy as np

    size = input_dict["size"]
    dtype = input_dict.get("dtype", tf.float64)
    
    if dtype == tf.float64:
        fill_value = 0.0
    elif dtype == tf.float32:
        fill_value = 0.0
    elif dtype == tf.float16:
        fill_value = 0.0
    elif dtype == tf.int64:
        fill_value = 0
    elif dtype == tf.int32:
        fill_value = 0
    elif dtype == tf.int16:
        fill_value = 0
    elif dtype == tf.int8:
        fill_value = 0
    elif dtype == tf.bool:
        fill_value = False
    else:
      fill_value = 0.0
    
    if not cpu:
        with tf.device('/GPU:0'):
            tensor = tf.fill([size], tf.cast(fill_value, dtype=dtype))
    else:
        tensor = tf.fill([size], tf.cast(fill_value, dtype=dtype))

    return {"result": tensor.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "size": 5,
        "dtype": np.float32
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()