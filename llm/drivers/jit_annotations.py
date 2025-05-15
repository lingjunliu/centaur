import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    annotation = input_dict.get("annotation", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.jit.annotate(input_tensor, annotation)

    if not cpu and result is not None:
        result = result.cpu()

    if result is not None:
        return {"result": result.numpy()}
    else:
        return {"result": np.zeros_like(input_dict["input"])}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    annotation = input_dict.get("annotation", None)

    result = input_tensor.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.int32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()