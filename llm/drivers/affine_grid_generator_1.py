import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    size = input_dict["size"]
    scale = input_dict.get("scale", None)

    if scale is not None:
        scale = torch.tensor(scale)
    else:
        batch_size = size[0]
        num_channels = 2 if len(size) == 3 else 2
        scale = torch.eye(num_channels).unsqueeze(0).repeat(batch_size, 1, 1)

    if not cpu:
        if scale is not None:
            scale = scale.cuda()

    if len(size) == 3:
        scale = torch.eye(2).unsqueeze(0).repeat(size[0], 1, 1)
        size = [size[0], 1, size[1], size[2]]

    grid = torch.affine_grid_generator(scale, torch.Size(size), align_corners=False)

    if not cpu:
        grid = grid.cpu()
    
    return {"result": grid.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    size = input_dict["size"]
    scale_np = input_dict.get("scale", None)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        if scale_np is None:
            batch_size = size[0]
            num_channels = 2 if len(size) == 3 else 2
            scale = tf.eye(num_channels, dtype=tf.float32)
            scale = tf.expand_dims(scale, axis=0)
            scale = tf.tile(scale, [batch_size, 1, 1])

        else:
            scale = tf.constant(scale_np, dtype=tf.float32)
            batch_size = size[0]
            scale = tf.expand_dims(scale, axis=0)
            scale = tf.tile(scale, [batch_size, 1, 1])

        N = size[0]
        H = size[1]
        W = size[2]
        num_channels = 2 if len(size) == 3 else size[-1]
        
        x = tf.linspace(-1.0, 1.0, W)
        y = tf.linspace(-1.0, 1.0, H)
        
        X, Y = tf.meshgrid(x, y)
        
        X = tf.reshape(X, (1, H, W, 1))
        Y = tf.reshape(Y, (1, H, W, 1))

        if num_channels == 2:
            grid = tf.concat([X, Y], axis=3)
        elif num_channels == 3:
            z = tf.zeros_like(X)
            grid = tf.concat([X, Y, z], axis=3)
        else:
            raise ValueError("Unexpected num_channels: {}".format(num_channels))
        
        grid = tf.tile(grid, [N, 1, 1, 1])

        if len(size) == 3:
            size = [size[0], 1, size[1], size[2]]

        grid_shape = tf.shape(grid)
        grid_reshaped = tf.reshape(grid, [size[0], size[1] * size[2] * size[3], num_channels])
        scale_transposed = tf.transpose(scale, perm=[0, 2, 1])

        transformed_grid = tf.matmul(grid_reshaped, scale_transposed)
        transformed_grid = tf.reshape(transformed_grid, size[:3] + [num_channels])
        result = transformed_grid.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "size": [2, 3, 4],
        "scale": np.array([[0.5, 0], [0, 0.5]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "size": [2, 3, 4]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "size": [2, 4, 4]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "size": [1, 3, 4, 2]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "size": [1, 3, 4]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()