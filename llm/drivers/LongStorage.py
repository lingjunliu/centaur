import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    long_tensor = input_tensor.long()
    buffer = long_tensor.numpy().tobytes()
    result = torch.LongStorage.from_buffer(buffer, byte_order='native')

    if not cpu:
        result = result.cpu()

    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        result = tf.cast(tf.cast(input_tensor, tf.int64), tf.int64).numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.2, 2.8, 3.1, 4.9, 5.5], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()