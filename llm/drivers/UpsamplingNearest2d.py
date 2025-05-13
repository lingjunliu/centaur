import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    scale_factor = input_dict.get("scale_factor", None)
    size = input_dict.get("size", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.interpolate(input_tensor.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, size=size, mode='nearest').squeeze()

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        scale_factor = input_dict.get("scale_factor", None)
        size = input_dict.get("size", None)

        if scale_factor is not None:
            if isinstance(scale_factor, float):
                new_height = int(input_tensor.shape[0] * scale_factor)
                new_width = int(input_tensor.shape[1] * scale_factor)
                new_size = [new_height, new_width]
            else:
                new_height = int(input_tensor.shape[0] * scale_factor[0])
                new_width = int(input_tensor.shape[1] * scale_factor[1])
                new_size = [new_height, new_width]

        elif size is not None:
            new_size = size
        else:
            raise ValueError("Either scale_factor or size must be specified.")

        result = tf.image.resize(
            tf.expand_dims(tf.expand_dims(input_tensor, axis=0), axis=3),
            size=new_size,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        result = tf.squeeze(result, axis=0)
        result = tf.squeeze(result, axis=2)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "scale_factor": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "size": [4,4]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()