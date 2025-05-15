import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 0.5)
    training = input_dict.get("training", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.nn.functional.alpha_dropout(input_tensor, p=p, training=training)
    
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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        p = input_dict.get("p", 0.5)
        training = input_dict.get("training", False)
        alpha = input_dict.get("alpha", 1.0)

        if training:
            keep_prob = 1 - p
            random_tensor = tf.random.uniform(shape=tf.shape(input_tensor), dtype=tf.float32)
            keep_mask = random_tensor < keep_prob
            output = tf.where(keep_mask, input_tensor / keep_prob, alpha * tf.ones_like(input_tensor) * (1-keep_prob) / (1-p))
        else:
            output = input_tensor

        result = output.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "p": 0.5,
        "training": True,
        "alpha": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()