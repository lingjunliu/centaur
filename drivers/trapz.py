import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    y = torch.tensor(input["y"])
    x = torch.tensor(input["x"]) if "x" in input else None
    dx = input["dx"] if "dx" in input else None
    dim = input.get("dim", -1)

    if not cpu:
        y = y.cuda()

    # Apply to torch.trapz
    if x is not None:
        if not cpu:
            x = x.cuda()
        integral = torch.trapz(y, x, dim=dim)
    elif dx is not None:
        integral = torch.trapz(y, dx=dx, dim=dim)
    else:
        integral = torch.trapz(y, dim=dim)

    if not cpu:
        integral = integral.cpu()

    return {"trapz_integral": integral.cpu().numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        y = tf.constant(input["y"])
        x = tf.constant(input["x"]) if "x" in input else None
        dx = input["dx"] if "dx" in input else None
        axis = input.get("dim", -1)

        # Apply to TensorFlow equivalent (np.trapz)
        if x is not None:
            integral = np.trapz(y.numpy(), x.numpy(), axis=axis)
        elif dx is not None:
            integral = np.trapz(y.numpy(), dx=dx, axis=axis)
        else:
            integral = np.trapz(y.numpy(), axis=axis)

        return {"trapz_integral": integral}

def main():
    # Example input for case 1
    input_data_case1 = {
        "y": np.array([1, 2, 3, 4], dtype=np.float32),
        "x": np.array([0, 1, 2, 3], dtype=np.float32),
        "dim": -1
    }

    # Example input for case 2
    input_data_case2 = {
        "y": np.array([1, 2, 3, 4], dtype=np.float32),
        "dx": 1.0,
        "dim": -1
    }

    # Torch example for case 1
    torch_result_case1 = torch_version(input_data_case1)
    print("Torch result case 1:", torch_result_case1)

    # TensorFlow example for case 1
    tf_result_case1 = tensorflow_version(input_data_case1)
    print("TensorFlow result case 1:", tf_result_case1)

    # Assertion for case 1
    assert np.allclose(torch_result_case1["trapz_integral"], tf_result_case1["trapz_integral"]), "Results are not equal for case 1"
    print("case 1 equal")

    # Torch example for case 2
    torch_result_case2 = torch_version(input_data_case2)
    print("Torch result case 2:", torch_result_case2)

    # TensorFlow example for case 2
    tf_result_case2 = tensorflow_version(input_data_case2)
    print("TensorFlow result case 2:", tf_result_case2)

    # Assertion for case 2
    assert np.allclose(torch_result_case2["trapz_integral"], tf_result_case2["trapz_integral"]), "Results are not equal for case 2"
    print("case 2 equal")

if __name__ == "__main__":
    main()