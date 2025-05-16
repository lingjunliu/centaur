import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    size = input_dict.get("size", None)
    scale_factor = input_dict.get("scale_factor", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = torch.nn.UpsamplingNearest2d(size=size, scale_factor=scale_factor)

    if not cpu:
        m = m.cuda()
        result = m(input_tensor)
        result = result.cpu()
    else:
        result = m(input_tensor)

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    size = input_dict.get("size", None)
    scale_factor = input_dict.get("scale_factor", None)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_shape = input_tensor.shape
        N, C, Hin, Win = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

        if scale_factor is not None:
            if isinstance(scale_factor, float) or isinstance(scale_factor, int):
                Hout = int(Hin * scale_factor)
                Wout = int(Win * scale_factor)
            else:
                Hout = int(Hin * scale_factor[0])
                Wout = int(Win * scale_factor[1])

        elif size is not None:
            if isinstance(size, int):
                Hout = size
                Wout = size
            else:
                Hout = size[0]
                Wout = size[1]

        else:
            raise ValueError("Either size or scale_factor must be specified")

        resized_tensor = tf.image.resize(input_tensor, [Hout, Wout], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        result = resized_tensor.numpy()

        result = result.transpose((0, 1, 2, 3))

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2),
        "scale_factor": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_result_np = torch_result["result"]
    tf_result_np = tf_result["result"]
    tf_result_np = np.transpose(tf_result_np, (0, 2, 3, 1))

    assert np.allclose(torch_result_np, tf_result_np, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()