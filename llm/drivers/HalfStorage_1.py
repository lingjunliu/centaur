import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    data = torch.tensor(input_dict["data"])

    if not cpu:
        data = data.cuda()

    storage = data.storage()
    result = storage.tolist()

    if not cpu:
        pass

    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    data_np = input_dict["data"]
    
    result = data_np.tolist()

    return {"result": np.array(result)}

def main():
    A_TOL = 0.01

    input_data = {
        "data": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()