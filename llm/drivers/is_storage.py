import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    try:
        input_data = input_dict["input"]
        if isinstance(input_data, (int, float, list, tuple, np.ndarray)):
            try:
                input_tensor = torch.tensor(input_data)
                if not cpu:
                    input_tensor = input_tensor.cuda()
                is_storage = torch.is_storage(input_tensor)
                if not cpu:
                    is_storage = torch.tensor(is_storage).cpu().item()
                return {"result": np.array(is_storage)}
            except:
                return {"result": np.array(False)}
        else:
            return {"result": np.array(False)}
    except:
        return {"result": np.array(False)}



def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    try:
        input_data = input_dict["input"]
        if isinstance(input_data, (int, float, list, tuple, np.ndarray)):
            try:
                input_tensor = tf.constant(input_data)
                result = False
                try:
                    input_tensor.numpy()
                    result = True
                except:
                    pass
                return {"result": np.array(result)}

            except:
                return {"result": np.array(False)}
        else:
            return {"result": np.array(False)}
    except:
        return {"result": np.array(False)}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": [1,2,3]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": "test"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_res = torch_result["result"]
    tf_res = tf_result["result"]
    assert np.allclose(torch_res, tf_res, atol=A_TOL), f"Results do not match torch:{torch_res}, tf:{tf_res}"
    
    input_data = {
        "input": {}
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": None
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()