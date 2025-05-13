import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    parameters = input_dict["parameters"]
    max_norm = input_dict["max_norm"]
    norm_type = input_dict.get("norm_type", 2.0)
    error_if_nonfinite = input_dict.get("error_if_nonfinite", False)

    torch_parameters = [torch.tensor(p, requires_grad=True) for p in parameters]

    if not cpu:
        torch_parameters = [p.cuda() for p in torch_parameters]

    total_norm = torch.nn.utils.clip_grad_norm_(torch_parameters, max_norm, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite)
    
    if not cpu:
        total_norm = total_norm.cpu()

    return {"total_norm": total_norm.numpy(), "parameters": [p.detach().cpu().numpy() if not cpu else p.detach().numpy() for p in torch_parameters]}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    parameters = input_dict["parameters"]
    max_norm = input_dict["max_norm"]
    norm_type = input_dict.get("norm_type", 2.0)
    error_if_nonfinite = input_dict.get("error_if_nonfinite", False)

    tf_parameters = [tf.Variable(p) for p in parameters]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        with tf.GradientTape() as tape:
            loss = sum([tf.reduce_sum(p**2) for p in tf_parameters])

        grads = tape.gradient(loss, tf_parameters)
        
        if len(grads) == 0:
            total_norm = tf.constant(0.0, dtype=tf.float32)
        else:
            grads_norm = [tf.norm(g, ord=norm_type) for g in grads]
            if len(grads_norm) == 1:
                total_norm = grads_norm[0]
            else:
                total_norm = tf.norm(tf.stack(grads_norm), ord=norm_type)

        if error_if_nonfinite:
            tf.debugging.check_numerics(total_norm, message="Total norm is non-finite")

        clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_norm)
        
        total_norm = tf.norm(tf.stack([tf.norm(g, ord=norm_type) for g in clipped_grads]), ord=norm_type)

        for i, p in enumerate(tf_parameters):
            p.assign(parameters[i])
            p.assign(p - grads[i] + clipped_grads[i])

        total_norm = total_norm.numpy()
        parameters_np = [p.numpy() for p in tf_parameters]

    return {"total_norm": total_norm, "parameters": parameters_np}

def main():
    A_TOL = 0.01

    input_data = {
        "parameters": [np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32)],
        "max_norm": 5.0,
        "norm_type": 2.0,
        "error_if_nonfinite": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["total_norm"], tf_result["total_norm"], atol=A_TOL), "Total norm results do not match"
    for i in range(len(input_data["parameters"])):
        assert np.allclose(torch_result["parameters"][i], tf_result["parameters"][i], atol=A_TOL), f"Parameter {i} results do not match"

    input_data = {
        "parameters": [np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([0.4, 0.5, 0.6], dtype=np.float32)],
        "max_norm": 0.5,
        "norm_type": 2.0,
        "error_if_nonfinite": False
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["total_norm"], tf_result["total_norm"], atol=A_TOL), "Total norm results do not match"
    for i in range(len(input_data["parameters"])):
        assert np.allclose(torch_result["parameters"][i], tf_result["parameters"][i], atol=A_TOL), f"Parameter {i} results do not match"

    input_data = {
        "parameters": [np.array([1.0, 2.0, 3.0], dtype=np.float32)],
        "max_norm": 100.0,
        "norm_type": 2.0,
        "error_if_nonfinite": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["total_norm"], tf_result["total_norm"], atol=A_TOL), "Total norm results do not match"
    for i in range(len(input_data["parameters"])):
        assert np.allclose(torch_result["parameters"][i], tf_result["parameters"][i], atol=A_TOL), f"Parameter {i} results do not match"

    print("Success")

if __name__ == "__main__":
    main()