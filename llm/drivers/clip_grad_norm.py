import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    parameters = input_dict["parameters"]
    max_norm = input_dict["max_norm"]
    norm_type = input_dict.get("norm_type", 2.0)
    error_if_nonfinite = input_dict.get("error_if_nonfinite", False)

    torch_parameters = [torch.tensor(p, requires_grad=True) for p in parameters]
    for p in torch_parameters:
        p.grad = torch.tensor(np.ones_like(p.detach().numpy()))

    if not cpu:
        torch_parameters = [p.cuda() for p in torch_parameters]

    total_norm = torch.nn.utils.clip_grad_norm_(torch_parameters, max_norm, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite)

    if not cpu:
        total_norm = total_norm.cpu()
    
    return {"total_norm": total_norm.numpy(), "parameters": [p.detach().cpu().numpy() for p in torch_parameters]}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        parameters = input_dict["parameters"]
        max_norm = input_dict["max_norm"]
        norm_type = input_dict.get("norm_type", 2.0)
        error_if_nonfinite = input_dict.get("error_if_nonfinite", False)

        tf_parameters = [tf.Variable(p) for p in parameters]
        gradients = [tf.constant(np.ones_like(p)) for p in parameters]
        
        grad_norm = tf.math.sqrt(tf.add_n([tf.reduce_sum(tf.math.square(grad)) for grad in gradients]))

        scale = max_norm / (grad_norm + 1e-8)
        scale = tf.minimum(scale, 1.0)

        clipped_gradients = [grad * scale for grad in gradients]

        assign_ops = [tf_parameters[i].assign(tf_parameters[i] - clipped_gradients[i]) for i in range(len(tf_parameters))]
        
        with tf.control_dependencies(assign_ops):
            total_norm = tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(grad)) for grad in gradients])).numpy()
        
    return {"total_norm": total_norm, "parameters": [p.numpy() for p in tf_parameters]}


def main():
    A_TOL = 0.01

    input_data = {
        "parameters": [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32)
        ],
        "max_norm": 5.0,
        "norm_type": 2.0,
        "error_if_nonfinite": False
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["total_norm"], tf_result["total_norm"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()