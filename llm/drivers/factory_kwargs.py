import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    factory_kwargs = input_dict.get("factory_kwargs", {})

    if not cpu:
        for key, value in factory_kwargs.items():
            if isinstance(value, np.ndarray):
                factory_kwargs[key] = torch.tensor(value, device="cuda")
            else:
                factory_kwargs[key] = value
        if 'device' not in factory_kwargs:
            factory_kwargs['device'] = 'cuda'

    linear_layer = torch.nn.Linear(in_features=10, out_features=5)
    if 'weight' in factory_kwargs:
        with torch.no_grad():
            linear_layer.weight = torch.nn.Parameter(torch.tensor(factory_kwargs['weight']))
    if not cpu:
        linear_layer = linear_layer.cuda()

    result = linear_layer

    if not cpu:
        result = result.cpu()

    return {"result": result.weight.detach().numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    factory_kwargs = input_dict.get("factory_kwargs", {})

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        if 'weight' in factory_kwargs:
            weight_np = factory_kwargs['weight']
            def initializer(shape, dtype=None):
                return tf.constant(weight_np.T, dtype=dtype)
        else:
            initializer = tf.keras.initializers.GlorotUniform()


        linear_layer = tf.keras.layers.Dense(units=5, kernel_initializer=initializer, use_bias=False, input_shape=(10,))
        linear_layer(tf.zeros((1, 10)))
        result = linear_layer.kernel.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "factory_kwargs": {
            "weight": np.random.rand(5, 10).astype(np.float32)
        }
    }

    torch_result = torch_version(input_data, cpu=True)
    tf_result = tensorflow_version(input_data, cpu=True)

    assert np.allclose(torch_result["result"], tf_result["result"].T, atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()