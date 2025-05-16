import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    output_size = input_dict.get("output_size", None)
    output_ratio = input_dict.get("output_ratio", None)
    return_indices = input_dict.get("return_indices", False)
    random_samples = input_dict.get("random_samples", None)
    if random_samples is not None:
        random_samples = torch.tensor(random_samples)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if random_samples is not None:
            random_samples = random_samples.cuda()

    layer = torch.nn.FractionalMaxPool2d(kernel_size=kernel_size, output_size=output_size,
                                                   output_ratio=output_ratio, return_indices=return_indices,
                                                   )
    if random_samples is not None:
        result, indices = layer(input_tensor, random_samples)
    else:
        result, indices = layer(input_tensor)

    if not cpu:
        result = result.cpu()
        indices = indices.cpu()

    return {"result": result.numpy(), "indices": indices.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        kernel_size = input_dict["kernel_size"]
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        output_size = input_dict.get("output_size", None)
        output_ratio = input_dict.get("output_ratio", None)
        return_indices = input_dict.get("return_indices", False)
        random_samples = input_dict.get("random_samples", None)

        input_shape = input_tensor.shape
        batch_size, channels, height, width = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

        if output_size is None and output_ratio is None:
            raise ValueError("output_size or output_ratio must be specified")

        if output_size is not None and output_ratio is not None:
            raise ValueError("only one of output_size or output_ratio should be specified")

        if output_size is not None:
            output_height, output_width = output_size
        else:
            output_height = int(height * output_ratio[0])
            output_width = int(width * output_ratio[1])
        
        pool_height, pool_width = kernel_size

        if random_samples is None:
            random_samples_height = tf.random.uniform(shape=[batch_size, output_height], minval=0, maxval=1)
            random_samples_width = tf.random.uniform(shape=[batch_size, output_width], minval=0, maxval=1)
        else:
            random_samples_height = tf.constant(random_samples[:, 0:output_height])
            random_samples_width = tf.constant(random_samples[:, output_height:])
            
        row_starts = tf.cast(tf.floor(random_samples_height * (height - pool_height + 1)), dtype=tf.int32)
        col_starts = tf.cast(tf.floor(random_samples_width * (width - pool_width + 1)), dtype=tf.int32)

        patches = tf.image.extract_patches(
            images=input_tensor,
            sizes=[1, pool_height, pool_width, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        patches = tf.reshape(patches, [batch_size, height - pool_height + 1, width - pool_width + 1, channels * pool_height * pool_width])
        
        def fractional_max_pool(input, pool_height, pool_width, row_starts, col_starts, output_height, output_width, channels):
            batch_size = tf.shape(input)[0]
            height = tf.shape(input)[1]
            width = tf.shape(input)[2]

            output = tf.TensorArray(dtype=tf.float32, size=output_height * output_width, dynamic_size=False, clear_after_read=False)
            indices = tf.TensorArray(dtype=tf.int32, size=output_height * output_width, dynamic_size=False, clear_after_read=False)

            def condition(i):
                return i < output_height * output_width

            def body(i, output, indices):
                row = i // output_width
                col = i % output_width

                row_start = row_starts[0, row]
                col_start = col_starts[0, col]
                
                patch = input[0, row_start:row_start + pool_height, col_start:col_start + pool_width, :]
                
                max_val = tf.reduce_max(patch)
                
                output = output.write(i, max_val)
                indices = indices.write(i, row_start * width + col_start)
                
                return i + 1, output, indices

            _, output_final, indices_final = tf.while_loop(condition, body, (0, output, indices))

            output_final = output_final.stack()
            indices_final = indices_final.stack()

            output_final = tf.reshape(output_final, [1, 1, output_height, output_width])
            indices_final = tf.reshape(indices_final, [1, 1, output_height, output_width])
            return output_final, indices_final

        result, indices = fractional_max_pool(input_tensor, pool_height, pool_width, row_starts, col_starts, output_height, output_width, channels)

        return {"result": result.numpy(), "indices": indices.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 1, 20, 20).astype(np.float32),
        "kernel_size": (5, 5),
        "output_size": (10, 10),
        "return_indices": True,
        "random_samples": np.random.rand(1, 20).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()