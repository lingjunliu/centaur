import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.backends.cudnn.is_acceptable(input_tensor)
    
    if not cpu:
        result = torch.tensor(result)
        result = result.cpu()

    return {"result": np.array([result]).astype(bool)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    input_tensor = tf.constant(input_dict["input"])

    if not cpu:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) == 0:
            result = False
        else:
            try:
                with tf.device('/GPU:0'):
                    # Attempt to use a batch norm operation which often relies on CuDNN
                    _ = tf.keras.layers.BatchNormalization()(input_tensor)
                    result = True
            except Exception as e:
                result = False

    else:
        result = True

    return {"result": np.array([result]).astype(bool)}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 224, 224, 3).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()