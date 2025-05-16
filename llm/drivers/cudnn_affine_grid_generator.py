import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["theta"])
    grid_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()
        result = torch.cudnn_affine_grid_generator(input_tensor, N=grid_size[0], C=grid_size[1], H=grid_size[2], W=grid_size[3])
        result = result.cpu()
    else:
        result = None
        try:
            result = torch.cudnn_affine_grid_generator(input_tensor, N=grid_size[0], C=grid_size[1], H=grid_size[2], W=grid_size[3])
        except Exception as e:
            print(f"CUDA is not available: {e}")
            return {"result": np.zeros(grid_size[:2] + grid_size[2:], dtype=np.float32)}

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    theta = tf.constant(input_dict["theta"])
    output_size = input_dict["output_size"]

    num_batch = tf.shape(theta)[0]
    height = output_size[2]
    width = output_size[3]

    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)

    x_t, y_t = tf.meshgrid(x, y)

    x_t_flat = tf.reshape(x_t, [1, -1])
    y_t_flat = tf.reshape(y_t, [1, -1])

    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.concat([x_t_flat, y_t_flat, ones], axis=0)

    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    batch_grids = tf.matmul(theta, sampling_grid)
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])
    result = tf.transpose(batch_grids, perm=[0, 2, 3, 1])

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "theta": np.array([[[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]]], dtype=np.float32),
        "output_size": (1, 1, 4, 5)
    }

    torch_result = torch_version(input_data, cpu=False)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()