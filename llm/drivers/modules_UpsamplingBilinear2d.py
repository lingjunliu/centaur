import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    size = input_dict.get("size", None)
    scale_factor = input_dict.get("scale_factor", None)
    mode = input_dict.get("mode", 'nearest')
    align_corners = input_dict.get("align_corners", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.interpolate(input_tensor.unsqueeze(0).unsqueeze(0), size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

    if not cpu:
        result = result.cpu()

    return {"result": result.squeeze().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    size = input_dict.get("size", None)
    scale_factor = input_dict.get("scale_factor", None)
    mode = input_dict.get("mode", 'nearest')
    align_corners = input_dict.get("align_corners", None)

    if size is None and scale_factor is None:
        raise ValueError("either size or scale_factor must be defined")

    input_tensor = tf.expand_dims(tf.expand_dims(input_tensor, axis=0), axis=0)

    if scale_factor is not None:
        if isinstance(scale_factor, float) or isinstance(scale_factor, int):
            scale_factor = [scale_factor, scale_factor]
        resized = tf.image.resize(input_tensor, 
                                   tf.cast(tf.shape(input_tensor)[2:4], dtype=tf.float32) * scale_factor,
                                   method=mode.upper(), preserve_aspect_ratio=False, antialias=False)
    else:
        resized = tf.image.resize(input_tensor, tf.constant([size[0], size[1]], dtype=tf.int32), method=mode.upper(), preserve_aspect_ratio=False, antialias=False)

    result = resized.numpy().squeeze()
    return {"result": result}


def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 3).astype(np.float32),
        "scale_factor": 2.0,
        "mode": "bilinear",
        "align_corners": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(2, 3).astype(np.float32),
        "size": (4,6),
        "mode": "bilinear",
        "align_corners": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()