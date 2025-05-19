import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    indices_tensor = torch.tensor(input["indices"], dtype=torch.long)
    kernel_size = input["kernel_size"]
    stride = input["stride"]
    padding = input["padding"]
    output_size = input.get("output_size", None)

    # Create MaxUnpool2d layer
    unpool = torch.nn.MaxUnpool2d(kernel_size, stride=stride, padding=padding)

    # Apply unpooling
    if output_size is not None:
        output = unpool(input_tensor, indices_tensor, output_size=output_size)
    else:
        output = unpool(input_tensor, indices_tensor)

    if cpu:
        output = output.cpu()

    return {"unpooled": output.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        indices_tensor = tf.constant(input["indices"], dtype=tf.int64)
        kernel_size = input["kernel_size"]
        stride = input["stride"]
        padding = input["padding"]
        output_size = input.get("output_size", None)
        
        def max_unpool_2d(pool, ind, ksize, stride, padding, out_shape=None):
            input_shape = tf.shape(pool, out_type=tf.int64)
            flat_input_size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
            flat_indices = tf.reshape(ind, [flat_input_size])
            if out_shape is None:
                out_h = (input_shape[1] - 1) * stride[0] - 2 * padding + ksize[0]
                out_w = (input_shape[2] - 1) * stride[1] - 2 * padding + ksize[1]
                out_shape = [input_shape[0], out_h, out_w, input_shape[3]]
            output_shape = tf.convert_to_tensor(out_shape, dtype=tf.int64)
            
            flat_output_shape = tf.reduce_prod(output_shape)
            ret = tf.scatter_nd(tf.reshape(flat_indices, [-1, 1]), tf.reshape(pool, [-1]), [flat_output_shape])
            ret = tf.reshape(ret, output_shape)
            return ret

        output = max_unpool_2d(input_tensor, indices_tensor, kernel_size, stride, padding, out_shape=output_size)
        
        return {"unpooled": output.numpy()}

def main():
    input_data = {
        "input": np.array([[[[1., 2.], [3., 4.]]]], dtype=np.float32),
        "indices": np.array([[[[0, 1], [2, 3]]]], dtype=np.int64),
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": 0,
        "output_size": [1, 1, 4, 4]
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert results are the same
    np.testing.assert_allclose(torch_result["unpooled"], tf_result["unpooled"], rtol=1e-3, atol=1e-5)

    if np.allclose(torch_result["unpooled"], tf_result["unpooled"], rtol=1e-3, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()