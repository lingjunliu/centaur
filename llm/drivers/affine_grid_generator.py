import numpy as np
import torch
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    size = input_dict["size"]
    align_corners = input_dict.get("align_corners", False)

    identity = torch.tensor([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0]], dtype=torch.float32)

    theta = identity.unsqueeze(0).repeat(size[0], 1, 1)
    
    grid = torch.affine_grid_generator(theta, torch.Size(size), align_corners=align_corners)
    
    if not cpu:
        grid = grid.cpu()
    
    return {"result": grid.numpy()}

def tensorflow_version(input_dict, cpu=True):
    size = input_dict["size"]
    align_corners = input_dict.get("align_corners", False)
    
    num_batch = size[0]
    height = size[2]
    width = size[3]

    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)

    X, Y = tf.meshgrid(x, y)
    
    X = tf.expand_dims(X, axis=0)
    Y = tf.expand_dims(Y, axis=0)
    
    X = tf.tile(X, [num_batch, 1, 1])
    Y = tf.tile(Y, [num_batch, 1, 1])
    
    stacked = tf.stack([Y, X], axis=-1)

    if align_corners:
        pass
    
    return {"result": stacked.numpy()}

def main():
    A_TOL = 0.01
    
    input_data = {
        "size": [2, 3, 5, 7],
        "align_corners": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_result_reshaped = torch_result["result"].transpose(0, 2, 3, 1)

    assert np.allclose(torch_result_reshaped, tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()