import numpy as np

def torch_version(input, cpu=True):
    import torch
    # Input
    x1 = torch.tensor(input["input"])
    dim = input.get("dim", None)
    
    if not cpu:
        x1 = x1.cuda()

    # Output
    if dim is None:
        y = torch.count_nonzero(x1)
    else:
        y = torch.count_nonzero(x1, dim=dim)

    if not cpu:
        y = y.cpu()

    return {"count_nonzero": y.numpy().item()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Input
        x1 = tf.constant(input["input"])
        dim = input.get("dim", None)

        # Output
        if dim is None:
            y = tf.math.count_nonzero(x1)
        else:
            y = tf.math.count_nonzero(x1, axis=dim)

        return {"count_nonzero": y.numpy()}

def main():
    input_data = {"input": np.array([[0, 1, 0], [2, 0, 3]]), "dim": None}

    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)
    
    np.testing.assert_allclose(torch_result["count_nonzero"], tf_result["count_nonzero"], rtol=1e-5, atol=1e-8)
    print("equal")

if __name__ == "__main__":
    main()
