import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    training = input_dict.get("training", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.feature_alpha_dropout(input_tensor, p=p, training=training)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        p = input_dict.get("p", 0.5)
        training = input_dict.get("training", False)
        alpha = input_dict.get("alpha", 1.0)
    
        if training:
          random_tensor = tf.random.uniform(tf.shape(input_tensor))
          dropout_mask = tf.cast(random_tensor >= p, dtype=tf.float32)
          output = input_tensor * dropout_mask / (1 - p)
          result = output
        else:
            result = input_tensor

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.0202, 1.0985, 1.3506, -0.6056]], dtype=np.float32),
        "p": 0.5,
        "training": True,
        "alpha": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[0.0202, 1.0985, 1.3506, -0.6056], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        "p": 0.3,
        "training": False,
        "alpha": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()