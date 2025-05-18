import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    storage = input_tensor.storage()
    
    result = np.frombuffer(storage.data_ptr().to_bytes(storage.nbytes(), 'little'), dtype=np.complex128, count=storage.size())

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])

    if not cpu:
        with tf.device('/GPU:0'):
            storage_data = tf.identity(input_tensor).numpy()
    else:
        storage_data = tf.identity(input_tensor).numpy()

    return {"result": storage_data}

def main():
    A_TOL = 1e-5

    input_data = {
        "input": np.array([1+1j, 2+2j, 3+3j, 4+4j], dtype=np.complex128),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"].flatten(), atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()