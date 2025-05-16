import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    params = input_dict["params"]
    clip_value = input_dict["clip_value"]

    torch_params = [torch.tensor(p, requires_grad=True) for p in params]

    if not cpu:
        torch_params = [p.cuda() for p in torch_params]

    for p in torch_params:
        torch.sum(p).backward(retain_graph=True, create_graph=True)

    torch.nn.utils.clip_grad_value_(torch_params, clip_value)

    if not cpu:
        torch_params = [p.cpu() for p in torch_params]

    return {"result": [p.grad.numpy() if p.grad is not None else None for p in torch_params]}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    params = input_dict["params"]
    clip_value = input_dict["clip_value"]

    tf_params = [tf.Variable(p, dtype=tf.float32) for p in params]

    with tf.GradientTape(persistent=True) as tape:
        loss = sum([tf.reduce_sum(p**2) for p in tf_params])

    grads = tape.gradient(loss, tf_params)

    clipped_grads = []
    for i, grad in enumerate(grads):
        if grad is None:
            clipped_grads.append(None)
        else:
            clipped_grads.append(tf.clip_by_value(grad, -clip_value, clip_value))
    del tape

    return {"result": [g.numpy() if g is not None else None for g in clipped_grads]}

def main():
    A_TOL = 0.01
    input_data = {
        "params": [np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32), np.array([0.5, -0.5], dtype=np.float32)],
        "clip_value": 2.0
    }

    torch_result = torch_version(input_data)
    tensorflow_result = tensorflow_version(input_data)

    for i in range(len(torch_result["result"])):
        if torch_result["result"][i] is not None:
            assert np.allclose(torch_result["result"][i], tensorflow_result["result"][i], atol=A_TOL), "Results do not match"
        else:
            assert tensorflow_result["result"][i] is None, "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()