import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    size = input_dict.get("size", None)
    scale_factor = input_dict.get("scale_factor", None)
    mode = input_dict.get("mode", 'bilinear')
    align_corners = input_dict.get("align_corners", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.interpolate(input_tensor.unsqueeze(0).unsqueeze(0), size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners).squeeze(0).squeeze(0)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        size = input_dict.get("size", None)
        scale_factor = input_dict.get("scale_factor", None)
        mode = input_dict.get("mode", 'bilinear')
        align_corners = input_dict.get("align_corners", False)

        input_tensor = tf.reshape(input_tensor, (1, input_tensor.shape[0], input_tensor.shape[1], 1))

        if size is not None:
            new_height = size[0]
            new_width = size[1]
            resized_tensor = tf.image.resize(input_tensor, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
        elif scale_factor is not None:
            if isinstance(scale_factor, float):
                scale_factor = [scale_factor, scale_factor]
            new_height = int(input_tensor.shape[1] * scale_factor[0])
            new_width = int(input_tensor.shape[2] * scale_factor[1])
            resized_tensor = tf.image.resize(input_tensor, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
        else:
            raise ValueError("Either size or scale_factor must be specified.")

        result = tf.squeeze(resized_tensor, axis=[0, 3])

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "scale_factor": 2.0,
        "mode": 'bilinear',
        "align_corners": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()