import numpy as np
import ctypes

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    data = input_dict["data_ptr"].astype(np.float16)
    data_ptr = torch.tensor(data)
    size = input_dict["size"]

    if not cpu:
        data_ptr = data_ptr.cuda()

    address = data_ptr.untyped_storage().data_ptr()
    storage = torch.HalfStorage.from_buffer(ctypes.cast(address, ctypes.c_void_p), int(size))

    if not cpu:
        storage = storage.cpu()

    return {"result": np.array(storage)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    data = input_dict["data_ptr"].astype(np.float16)
    data_ptr = tf.convert_to_tensor(data, dtype=tf.float16)
    size = input_dict["size"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        result = data_ptr.numpy()[:size]

    return {"result": result}

def main():
    A_TOL = 0.01

    data = np.array([0.0202, 1.0985, 1.3506, -0.6056, 0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float16)
    input_data = {
        "data_ptr": data,
        "size": 4
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()