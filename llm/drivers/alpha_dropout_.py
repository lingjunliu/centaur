import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    training = input_dict.get("training", False)
    inplace = input_dict.get("inplace", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.alpha_dropout(input_tensor, p=p, training=training, inplace=inplace)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        p = input_dict.get("p", 0.5)
        training = input_dict.get("training", False)

        if training:
            alpha = -np.sqrt((1 - p) / (np.var(input_dict["input"]) + p * np.mean(np.square(input_dict["input"]))))
            alpha = tf.cast(alpha, dtype=tf.float32)

            keep_prob = 1 - p
            random_tensor = keep_prob + tf.random.uniform(shape=tf.shape(input_tensor), minval=0, maxval=1)
            binary_mask = tf.floor(random_tensor)
            x = tf.multiply(input_tensor, binary_mask)
            output = alpha * (x - (1 - binary_mask))
            output = tf.clip_by_value(output, clip_value_min=input_tensor.numpy().min(), clip_value_max=input_tensor.numpy().max())
        else:
            output = input_tensor

        output = output.numpy()

    return {"result": output}

def main():
    A_TOL = 0.1

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "p": 0.5,
        "training": True,
        "inplace": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()