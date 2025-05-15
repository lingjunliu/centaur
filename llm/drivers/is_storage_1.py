import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.is_storage(input_tensor.storage())

    if not cpu:
        result = np.array(result)
    else:
        result = np.array(result)

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    result = False
    if isinstance(input_tensor, tf.Tensor):
        result = True
    return {"result": np.array(result)}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()