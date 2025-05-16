import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    shapes = tuple(torch.Size(shape) for shape in input["shapes"])
    
    # Apply to torch.broadcast_shapes
    broadcasted_shape = torch.broadcast_shapes(*shapes)

    return {"broadcasted_shape": [int(dim) for dim in broadcasted_shape]}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    # Unpack input dictionary
    shapes = input["shapes"]

    # Apply to TensorFlow equivalent
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        shape_list = [tf.TensorShape(shape) for shape in shapes]
        broadcasted_shape = tf.broadcast_static_shape(*shape_list)

        return {"broadcasted_shape": broadcasted_shape.as_list()}

def main():
    # Example input
    input_data = {
        "shapes": [[2, 1, 3], [1, 4, 1]]  # Example shapes to broadcast
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if torch_result["broadcasted_shape"] == tf_result["broadcasted_shape"]:
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()