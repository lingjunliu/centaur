import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    indices_tensor = torch.tensor(input_dict["indices"])
    output_size = input_dict.get("output_size", None)

    if not cpu:
        input_tensor = input_tensor.cuda()
        indices_tensor = indices_tensor.cuda()

    result = torch.nn.MaxUnpool3d(kernel_size=input_dict["kernel_size"], stride=input_dict.get("stride", None), padding=input_dict.get("padding", 0))(input_tensor, indices_tensor, output_size=output_size)

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
        indices_tensor = tf.cast(tf.constant(input_dict["indices"]), dtype=tf.int64)
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", None)
        padding = input_dict.get("padding", 0)
        output_size = input_dict.get("output_size", None)

        input_shape = tf.shape(input_tensor)
        batch_size = input_shape[0]
        channels = input_shape[1]
        input_depth = input_shape[2]
        input_height = input_shape[3]
        input_width = input_shape[4]
        
        indices_shape = tf.shape(indices_tensor)
        output_depth = input_depth * (stride[0] if stride else kernel_size[0]) if output_size is None else output_size[0]
        output_height = input_height * (stride[1] if stride else kernel_size[1]) if output_size is None else output_size[1]
        output_width = input_width * (stride[2] if stride else kernel_size[2]) if output_size is None else output_size[2]

        output_size_tensor = [batch_size, channels, output_depth, output_height, output_width]

        if isinstance(padding, int):
            padding = [padding, padding, padding]

        output_shape = output_size_tensor
        
        output = tf.zeros(output_shape, dtype=input_tensor.dtype)

        def body(b, c, z, y, x, output):
          
            index = indices_tensor[b, c, z, y, x]
            
            z_out = index // (output_height * output_width)
            rem = index % (output_height * output_width)
            y_out = rem // output_width
            x_out = rem % output_width
            
            updates = input_tensor[b, c, z, y, x]
            
            output = tf.tensor_scatter_nd_update(output, [[b, c, z_out, y_out, x_out]], [updates])

            return b, c, z, y, x + 1, output

        def body_y(b, c, z, y, x, output):
            x = tf.constant(0)
            return b, c, z, y + 1, x, tf.while_loop(lambda _b, _c, _z, _y, _x, _: _x < input_width, lambda _b, _c, _z, _y, _x, _output: body(_b, _c, _z, _y, _x, _output), [b, c, z, y, x, output])[5]

        def body_z(b, c, z, y, x, output):
            y = tf.constant(0)
            x = tf.constant(0)
            return b, c, z + 1, tf.while_loop(lambda _b, _c, _z, _y, _x, _: _y < input_height, lambda _b, _c, _z, _y, _x, _output: body_y(_b, _c, _z, _y, _x, _output), [b, c, z, y, x, output])[3], x, output

        def body_c(b, c, z, y, x, output):
            z = tf.constant(0)
            y = tf.constant(0)
            x = tf.constant(0)
            return b, c+1, tf.while_loop(lambda _b, _c, _z, _y, _x, _: _z < input_depth, lambda _b, _c, _z, _y, _x, _output: body_z(_b, _c, _z, _y, _x, _output), [b, c, z, y, x, output])[2], y, x, output

        def body_b(b, c, z, y, x, output):
            c = tf.constant(0)
            z = tf.constant(0)
            y = tf.constant(0)
            x = tf.constant(0)
            return b+1, tf.while_loop(lambda _b, _c, _z, _y, _x, _: _c < channels, lambda _b, _c, _z, _y, _x, _output: body_c(_b, _c, _z, _y, _x, _output), [b, c, z, y, x, output])[1], z, y, x, output

        b = tf.constant(0)
        c = tf.constant(0)
        z = tf.constant(0)
        y = tf.constant(0)
        x = tf.constant(0)

        _,_,_,_,_,result = tf.while_loop(lambda _b, _c, _z, _y, _x, _: _b < batch_size, lambda _b, _c, _z, _y, _x, _output: body_b(_b, _c, _z, _y, _x, _output), [b, c, z, y, x, output])
    
        result = result.numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 2, 2, 2, 2).astype(np.float32),
        "indices": np.random.randint(0, 8, size=(1, 2, 2, 2, 2)).astype(np.int64),
        "kernel_size": (1, 1, 1),
        "stride": (1, 1, 1),
        "padding": 0,
        "output_size": (2, 2, 2),
    }


    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()