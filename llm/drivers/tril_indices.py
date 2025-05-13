import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    n = input_dict["n"]
    row = input_dict["row"]
    col = input_dict["col"]
    offset = input_dict.get("offset", 0)

    if not cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    result = torch.tril_indices(row, col, offset, device=device)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    row = input_dict["row"]
    col = input_dict["col"]
    offset = input_dict.get("offset", 0)

    row_indices = []
    col_indices = []
    
    for i in range(row):
        for j in range(col):
            if i - j <= offset:
                row_indices.append(i)
                col_indices.append(j)
    
    result_indices = np.array([row_indices, col_indices])

    return {"result": result_indices}

def main():
    A_TOL = 0.01

    input_data = {
        "n": 5,
        "row": 4,
        "col": 3,
        "offset": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_result = torch_result["result"]
    tf_result = tf_result["result"]

    max_cols = max(torch_result.shape[1], tf_result.shape[1])
    torch_result_padded = np.pad(torch_result, ((0,0),(0, max_cols - torch_result.shape[1])), 'constant', constant_values=0)
    tf_result_padded = np.pad(tf_result, ((0,0),(0, max_cols - tf_result.shape[1])), 'constant', constant_values=0)

    assert np.allclose(torch_result_padded, tf_result_padded, atol=A_TOL), "Results do not match"

    input_data = {
        "n": 5,
        "row": 5,
        "col": 5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_result = torch_result["result"]
    tf_result = tf_result["result"]

    max_cols = max(torch_result.shape[1], tf_result.shape[1])
    torch_result_padded = np.pad(torch_result, ((0,0),(0, max_cols - torch_result.shape[1])), 'constant', constant_values=0)
    tf_result_padded = np.pad(tf_result, ((0,0),(0, max_cols - tf_result.shape[1])), 'constant', constant_values=0)

    assert np.allclose(torch_result_padded, tf_result_padded, atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()