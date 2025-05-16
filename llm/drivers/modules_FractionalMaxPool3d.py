import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    output_size = input_dict["output_size"]
    if "random_samples" in input_dict:
        random_samples = torch.tensor(input_dict["random_samples"])
    else:
        random_samples = torch.rand(input_tensor.size()[-3:])

    if not cpu:
        input_tensor = input_tensor.cuda()
        random_samples = random_samples.cuda()

    m = torch.nn.FractionalMaxPool3d(kernel_size=kernel_size, output_size=output_size, return_indices=False)
    
    with torch.no_grad():
        result = m(input_tensor)
        
    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import numpy as np

    input_tensor = tf.constant(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    output_size = input_dict["output_size"]
    if "random_samples" in input_dict:
        random_samples = tf.constant(input_dict["random_samples"], dtype=tf.float32)
    else:
        random_samples = np.random.rand(*input_tensor.shape[1:4]).astype(np.float32)
        random_samples = tf.constant(random_samples, dtype=tf.float32)

    input_shape = input_tensor.shape
    batch_size = input_shape[0]
    depth, height, width = input_shape[1:4]
    channels = input_shape[4]

    target_depth, target_height, target_width = output_size

    depth_ratio = depth / target_depth
    height_ratio = height / target_height
    width_ratio = width / target_width
    
    depth_padding = int((depth_ratio * (target_depth - 1) + kernel_size[0] - depth) / 2)
    height_padding = int((height_ratio * (target_height - 1) + kernel_size[1] - height) / 2)
    width_padding = int((width_ratio * (target_width - 1) + kernel_size[2] - width) / 2)
    
    padded_input = tf.pad(input_tensor, [[0, 0], [depth_padding, depth_padding], [height_padding, height_padding], [width_padding, width_padding], [0, 0]])
    
    def compute_pooling_region(idx, length, target_length, ratio, padding, kernel_size, random_sample):
        start = int(np.floor(idx * ratio)) + padding
        end = int(np.ceil((idx + 1) * ratio)) + padding
        ksize = min(kernel_size, end - start)
        start = int(np.floor((end - ksize) * random_sample))
        end = start + ksize
        return start, end

    output = np.zeros((batch_size, target_depth, target_height, target_width, channels))
    
    for b in range(batch_size):
        for z in range(target_depth):
            for y in range(target_height):
                for x in range(target_width):
                    depth_start, depth_end = compute_pooling_region(z, depth, target_depth, depth_ratio, depth_padding, kernel_size[0], random_samples[z])
                    height_start, height_end = compute_pooling_region(y, height, target_height, height_ratio, height_padding, kernel_size[1], random_samples[y])
                    width_start, width_end = compute_pooling_region(x, width, target_width, width_ratio, width_padding, kernel_size[2], random_samples[x])
                    
                    region = padded_input[b, depth_start:depth_end, height_start:height_end, width_start:width_end, :]
                    output[b, z, y, x, :] = tf.reduce_max(region, axis=(0, 1, 2)).numpy()

    return {"result": output}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 4, 4, 4, 1).astype(np.float32),
        "kernel_size": (2, 2, 2),
        "output_size": (2, 2, 2),
        "random_samples": np.random.rand(4, 4, 4).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()