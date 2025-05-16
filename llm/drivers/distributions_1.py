import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    loc = torch.tensor(input_dict["loc"])
    scale = torch.tensor(input_dict["scale"])

    if not cpu:
        loc = loc.cuda()
        scale = scale.cuda()

    dist = torch.distributions.Normal(loc, scale)
    result = dist.sample((5,))

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    loc = tf.constant(input_dict["loc"])
    scale = tf.constant(input_dict["scale"])

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        result = tf.random.normal((5, len(input_dict["loc"])), mean=loc, stddev=scale)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "loc": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()