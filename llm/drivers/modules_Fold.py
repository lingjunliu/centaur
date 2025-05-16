import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    output_size = input_dict["output_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()

    c = 1
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    oh, ow = (output_size, output_size) if isinstance(output_size, int) else output_size
    
    ih = ((oh + 2 * padding - dilation * (kh - 1) - 1) // stride) + 1
    iw = ((ow + 2 * padding - dilation * (kw - 1) - 1) // stride) + 1
    
    input_tensor = input_tensor.reshape(1, c * ih * iw, kh * kw)
    input_tensor = input_tensor.transpose(1, 2)

    result = torch.nn.functional.fold(input_tensor, (oh, ow), kernel_size, stride=stride, padding=padding, dilation=dilation)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import numpy as np

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    kernel_size = input_dict["kernel_size"]
    output_size = input_dict["output_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)

    input_np = input_dict["input"]
    ks = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    os = output_size if isinstance(output_size, int) else output_size[0]
    s = stride if isinstance(stride, int) else stride[0]
    p = padding if isinstance(padding, int) else padding[0]
    d = dilation if isinstance(dilation, int) else dilation[0]

    input_channels = 1
    oh = ((os + 2 * p - d * (ks - 1) - 1) // s) + 1
    ow = ((os + 2 * p - d * (ks - 1) - 1) // s) + 1
    num_patches = oh * ow
    patches_tf = tf.reshape(input_tensor, [1,num_patches,ks*ks*input_channels])

    output = tf.zeros((os, os, input_channels), dtype=tf.float32)

    patches_tf = tf.reshape(patches_tf, [oh, ow, ks, ks, input_channels])
    for i in range(oh):
        for j in range(ow):
            patch = patches_tf[i, j]
            h_start = i * s
            w_start = j * s

            h_indices = tf.range(h_start, h_start + ks)
            w_indices = tf.range(w_start, w_start + ks)
            hh, ww = tf.meshgrid(h_indices, w_indices)
            indices = tf.stack([tf.reshape(hh, [-1]), tf.reshape(ww, [-1]), tf.zeros(ks*ks, dtype=tf.int32)], axis=1)

            updates = tf.reshape(patch, [-1])
            output = tf.tensor_scatter_nd_add(output, indices, updates)

    result = output.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.], dtype=np.float32),
        "kernel_size": 2,
        "output_size": 4,
        "stride": 1,
        "padding": 0,
        "dilation": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()