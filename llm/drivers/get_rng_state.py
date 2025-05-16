import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    if not cpu:
        torch.cuda.manual_seed(0)
        device = torch.device('cuda')
    else:
        torch.manual_seed(0)
        device = torch.device('cpu')

    result = torch.get_rng_state().to(device)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        tf.random.set_seed(0)
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
                logical_gpus = tf.config.list_logical_devices('GPU')
            except RuntimeError as e:
                print(e)
        tf.random.set_seed(0)

    generator = tf.random.Generator.from_seed(0)
    initial_state = generator.state

    result = initial_state.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {}

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_state = torch_result["result"]
    tf_state = tf_result["result"]

    min_len = min(len(torch_state), len(tf_state))

    assert np.allclose(torch_state[:min_len], tf_state[:min_len], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()