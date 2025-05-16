import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", None)
    if stride is not None:
        stride = input_dict["stride"]
    padding = input_dict.get("padding", 0)
    ceil_mode = input_dict.get("ceil_mode", False)
    p = input_dict.get("p", 2)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.lp_pool3d(input_tensor, norm_type=p, kernel_size=kernel_size, stride=stride, ceil_mode=ceil_mode)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", None)
    if stride is None:
      stride = kernel_size
    padding = input_dict.get("padding", 0)
    ceil_mode = input_dict.get("ceil_mode", False)
    p = input_dict.get("p", 2)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)

    input_shape = input_tensor.shape
    
    if len(input_shape) == 4:
      input_tensor = tf.expand_dims(input_tensor, axis=0)

    if len(input_shape) == 5:
      N, C, D, H, W = input_shape
    else:
      N, C, D, H, W = 1, input_shape[0], input_shape[1], input_shape[2], input_shape[3]

    if padding != (0,0,0):
      pad_before = [0, 0, padding[0], padding[1], padding[2]]
      pad_after = [0, 0, padding[0], padding[1], padding[2]]

      paddings = [[pb, pa] for pb, pa in zip(pad_before, pad_after)]
      input_tensor = tf.pad(input_tensor, paddings, "REFLECT")

    input_tensor = tf.pow(tf.abs(input_tensor), p)
    
    result = tf.nn.avg_pool3d(
        input=input_tensor,
        ksize=kernel_size,
        strides=stride,
        padding='VALID',
        data_format='NCDHW'
    )

    result = tf.pow(result, 1.0/p)

    result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(2, 3, 10, 10, 10).astype(np.float32),
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "p": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()