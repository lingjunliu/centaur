import numpy as np
import logging
import pytest
import torch

from generator.rules import dist_1_rev, dist_7_rev, dist_8_rev, dist_5_rev, dist_11_rev, rule_to_distance

## Abid, please add assertions with corresponding expectations here --Marcelo

@pytest.mark.unit
def test_dist_1():
    ## I don't understand why this is true.
    # res=rule_1({'a':None}, {'b':None})    
    # print(res)

    ## another example
    t1=np.random.rand(3,2,5)
    t2=np.random.rand(4,2)
    # res=rule_1({'a':t1}, {'b':t2})
    # print(res) ## this is 
    assert dist_1_rev({'a':t1}, {'b':t2}) > 0, f"Rule 1 was satisfied for args with shape {t1.shape} and {t2.shape}, but it should not have" # should be higher than 0
    # simulate -1@1 on t1
    t1=np.random.rand(2,2,5) # from 3 to 2
    assert dist_1_rev({'a':t1}, {'b':t2}) > 0, f"Rule 1 was satisfied for args with shape {t1.shape} and {t2.shape}, but it should not have" # should be higher than 0
    ## got worse. let us try the other direction
    t1=np.random.rand(4,2,5) # from 3 to 4
    print(dist_1_rev({'a':t1}, {'b':t2})) ## this is the new best <~~~
    assert dist_1_rev({'a':t1}, {'b':t2}) > 0, f"Rule 1 was satisfied for args with shape {t1.shape} and {t2.shape}, but it should not have" # should be higher than 0
    ## keep going the same direction
    t1=np.random.rand(5,2,5) # from 4 to 5
    print(dist_1_rev({'a':t1}, {'b':t2})) ## this is worse. get back to the previous best
    assert dist_1_rev({'a':t1}, {'b':t2}) > 0, f"Rule 1 was satisfied for args with shape {t1.shape} and {t2.shape}, but it should not have" # should be higher than 0
    t1=np.random.rand(4,2,5) # recovering the best candidate
    ## try mutating the other input. remove one axis
    t2=np.random.rand(4)
    print(dist_1_rev({'a':t1}, {'b':t2})) ## this is much worse (bigger distance). go the other way
    assert dist_1_rev({'a':t1}, {'b':t2}) > 0, f"Rule 1 was satisfied for args with shape {t1.shape} and {t2.shape}, but it should not have" # should be higher than 0
    t2=np.random.rand(4,2,1)
    print(dist_1_rev({'a':t1}, {'b':t2})) ## awesome. this is the new best <~~~
    assert dist_1_rev({'a':t1}, {'b':t2}) > 0, f"Rule 1 was satisfied for args with shape {t1.shape} and {t2.shape}, but it should not have" # should be higher than 0
    t2=np.random.rand(3,2,1)
    print(dist_1_rev({'a':t1}, {'b':t2})) ## worse
    assert dist_1_rev({'a':t1}, {'b':t2}) > 0, f"Rule 1 was satisfied for args with shape {t1.shape} and {t2.shape}, but it should not have" # should be higher than 0
    t2=np.random.rand(5,2,1)
    print(dist_1_rev({'a':t1}, {'b':t2})) ## worse
    assert dist_1_rev({'a':t1}, {'b':t2}) > 0, f"Rule 1 was satisfied for args with shape {t1.shape} and {t2.shape}, but it should not have" # should be higher than 0
    t2=np.random.rand(4,2,1) ## recovering
    t2=np.random.rand(4,1,1)
    print(dist_1_rev({'a':t1}, {'b':t2})) ## worse
    assert dist_1_rev({'a':t1}, {'b':t2}) > 0, f"Rule 1 was satisfied for args with shape {t1.shape} and {t2.shape}, but it should not have" # should be higher than 0
    t2=np.random.rand(4,3,1)
    print(dist_1_rev({'a':t1}, {'b':t2})) ## worse
    assert dist_1_rev({'a':t1}, {'b':t2}) > 0, f"Rule 1 was satisfied for args with shape {t1.shape} and {t2.shape}, but it should not have" # should be higher than 0
    t2=np.random.rand(4,2,1) ## recovering
    t2=np.random.rand(4,2,0) ## worse
    print(dist_1_rev({'a':t1}, {'b':t2})) 
    assert dist_1_rev({'a':t1}, {'b':t2}) > 0, f"Rule 1 was satisfied for args with shape {t1.shape} and {t2.shape}, but it should not have" # should be higher than 0
    t2=np.random.rand(4,2,2) ## better. this is the new best <~~~
    print(dist_1_rev({'a':t1}, {'b':t2}))
    assert dist_1_rev({'a':t1}, {'b':t2}) > 0, f"Rule 1 was satisfied for args with shape {t1.shape} and {t2.shape}, but it should not have" # should be higher than 0
    t2=np.random.rand(4,2,5) ## perfect. 
    print(dist_1_rev({'a':t1}, {'b':t2}))
    assert dist_1_rev({'a':t1}, {'b':t2}) == 0, f"Rule 1 was mot satisfied for args with shape {t1.shape} and {t2.shape}" # should be 0

@pytest.mark.unit
def test_dist_2():
    # Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # Shape: (2, 2)
    arg2 = {"dim": 1}
    assert rule_to_distance[2]['rule_2'](arg1, arg2) == 0, f"Rule 2 was not satisfied for input with shape {arg1['input_tensor'].shape} and dim {arg2['dim']}"

    # Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # Shape: (2, 2)
    arg2 = {"dim": 3}   # Invalid dim
    assert rule_to_distance[2]['rule_2'](arg1, arg2) > 0, f"Rule 2 was satisfied for input with shape {arg1['input_tensor'].shape} and dim {arg2['dim']}, but it should not have been"

@pytest.mark.unit
def test_dist_4():
    # Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.float32)} # float32
    arg2 = {"other_tensor": np.array([[5, 6], [7, 8]], dtype=np.float32)} # float32
    assert rule_to_distance[2]['rule_4'](arg1, arg2) == 0, f"Rule 4 was not satisfied for input with shape {arg1['input_tensor'].shape} and dtype {arg1['input_tensor'].dtype} and other_tensor with shape {arg2['other_tensor'].shape} and dtype {arg2['other_tensor'].dtype}"
    
    # Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.float32)} # float32
    arg2 = {"other_tensor": np.array([[5, 6], [7, 8]], dtype=np.int32)} # int32, not the same as arg1
    assert rule_to_distance[2]['rule_4'](arg1, arg2) > 0, f"Rule 4 was satisfied for input with shape {arg1['input_tensor'].shape} and dtype {arg1['input_tensor'].dtype} and other_tensor with shape {arg2['other_tensor'].shape} and dtype {arg2['other_tensor'].dtype}, but it should not have been"

@pytest.mark.unit
def test_dist_5():
    input = np.random.rand(3, 5)
    index = np.array([[0, 1, 2, 0]])
    assert dist_5_rev({'input': input}, {'index': index}) == 0, f"Rule 5 was not satisfied for input with shape {input.shape} and index with values within [{np.min(index)}, {np.max(index)}]"
    
    input = torch.full((2, 4), 2., dtype=int).numpy()
    index = torch.tensor([[2], [3]]).numpy()
    assert dist_5_rev({'input': input}, {'index': index}) != 0, f"Rule 5 was satisfied for input with shape {input.shape} and index with values within [{np.min(index)}, {np.max(index)}], but it should not have"
    
    # Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # Shape: (2, 2)
    arg2 = {"index": np.array([0, 1])} # Valid index
    assert dist_5_rev(arg1, arg2) == 0, f"Rule 5 was not satisfied for input with shape {arg1['input_tensor'].shape} and index with values within [{np.min(arg2['index'])}, {np.max(arg2['index'])}]"
    
    # Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]])} # Shape: (2, 2)
    arg2 = {"index": np.array([2])} # Invalid index, out of range
    assert dist_5_rev(arg1, arg2) > 0, f"Rule 5 was satisfied for input with shape {arg1['input_tensor'].shape} and index with values within [{np.min(arg2['index'])}, {np.max(arg2['index'])}], but it should not have been"

@pytest.mark.unit
def test_dist_7():
    t1=np.random.rand(2,2)
    t2=np.random.rand(1,2)
    assert dist_7_rev({'a':t1}, {'b':t2}) == 0, f"Rule 7 was not satisfied for args with shape {t1.shape} and {t2.shape}" # should be 0
    t1=np.random.rand(3,2)
    t2=np.random.rand(2,2)
    assert dist_7_rev({'a':t1}, {'b':t2}) > 0, f"Rule 7 was satisfied for args with shape {t1.shape} and {t2.shape}, but it should not have" # should be higher than 0
    t1=np.random.rand(2,100)
    t2=np.random.rand(50,2)
    assert dist_7_rev({'a':t1}, {'b':t2}) == 0, f"Rule 7 was not satisfied for args with shape {t1.shape} and {t2.shape}" # should be 0
    t1=np.random.rand(3,100)
    t2=np.random.rand(100,2)
    assert dist_7_rev({'a':t1}, {'b':t2}) > 0, f"Rule 7 was satisfied for args with shape {t1.shape} and {t2.shape}, but it should not have" # should be higher than 0

@pytest.mark.unit
def test_dist_8():
    # Positive
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.int32)} # int32
    assert dist_8_rev(arg1) == 0, f"Rule 8 was not satisfied for input with shape {arg1['input_tensor'].shape} and dtype {arg1['input_tensor'].dtype}"
    # Positive
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.int64)} # int64
    assert dist_8_rev(arg1) == 0, f"Rule 8 was not satisfied for input with shape {arg1['input_tensor'].shape} and dtype {arg1['input_tensor'].dtype}"
    # Negative
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.complex128)} # float, not int
    assert dist_8_rev(arg1) > 0, f"Rule 8 was satisfied for input with shape {arg1['input_tensor'].shape} and dtype {arg1['input_tensor'].dtype}, but it should not have been"
    arg2 = {"other_tensor": np.array([[5, 6], [7, 8]], dtype=bool)} # int32  
    # Negative
    assert dist_8_rev(arg2) > 0, f"Rule 8 was satisfied for input with shape {arg2['input_tensor'].shape} and dtype {arg2['input_tensor'].dtype}, but it should not have been"
    
    # Negative Example, non-tensor
    arg1 = {"dim": 2} # dim
    assert dist_8_rev(arg1) > 0, f"Rule 8 was satisfied for dim {arg1['dim']}, but it should not have been"
    # Negative Example
    arg1 = {"dim": 3.2}
    assert dist_8_rev(arg1) > 0, f"Rule 8 was satisfied for dim {arg1['dim']}, but it should not have been"

@pytest.mark.unit
def test_dist_11():
    # Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2, 3], [3, 4, 5]])} # Shape: (2, 3)
    arg2 = {"dim": 1}    # dim
    arg3 = {"index": np.array([0, 2])} # Valid index for dim size 3
    assert dist_11_rev(arg1, arg2, arg3) == 0, f"Rule 11 was not satisfied for input with shape {arg1['input_tensor'].shape} and index with values within [{np.min(arg3['index'])}, {np.max(arg3['index'])}] for dim {arg2['dim']}"
    
    # Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2, 3], [3, 4, 5]])} # Shape: (2, 3)
    arg2 = {"dim": 0}     # dim
    arg3 = {"index": np.array([0, 2])} # Invalid index for dim size 2
    assert dist_11_rev(arg1, arg2, arg3) > 0, f"Rule 11 was satisfied for input with shape {arg1['input_tensor'].shape} and index with values within [{np.min(arg3['index'])}, {np.max(arg3['index'])}] for dim {arg2['dim']}, but it should not have been"

@pytest.mark.unit
def test_dist_12():
    # Positive Example
    arg1 = {"low": 2}
    arg2 = {"high": 5} # Valid range
    assert rule_to_distance[2]["rule_12"](arg1, arg2) == 0, f"Rule 12 was not satisfied for low {arg1['low']} and high {arg2['high']}"
    
    # Negative Example:
    arg1 = {"low": 5}
    arg2 = {"high": 2} # Invalid range
    assert rule_to_distance[2]["rule_12"](arg1, arg2) > 0, f"Rule 12 was satisfied for low {arg1['low']} and high {arg2['high']}, but it should not have been"
    
    # Negative Example:
    arg1 = {"low": 2}
    arg2 = {"high": [2, 2]}
    assert rule_to_distance[2]["rule_12"](arg1, arg2) > 0, f"Rule 12 was satisfied for low {arg1['low']} and high {arg2['high']}, but it should not have been"

@pytest.mark.unit
def test_dist_13():
    # Positive Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.float16)} # float16
    assert rule_to_distance[1]["rule_13"](arg1) == 0, f"Rule 13 was not satisfied for input with shape {arg1['input_tensor'].shape} and dtype {arg1['input_tensor'].dtype}"

    # Negative Example:
    arg1 = {"input_tensor": np.array([[1, 2], [3, 4]], dtype=np.complex128)} # comlex128, not float
    assert rule_to_distance[1]["rule_13"](arg1) > 0, f"Rule 13 was satisfied for input with shape {arg1['input_tensor'].shape} and dtype {arg1['input_tensor'].dtype}, but it should not have been"
    
    # Negative Example, non-tensor
    arg1 = {"padding": 2.1}
    assert rule_to_distance[1]["rule_13"](arg1) > 0, f"Rule 13 was satisfied for padding {arg1['padding']}, but it should not have been"
    # Negative Example:
    arg1 = {"padding": 2}
    assert rule_to_distance[1]["rule_13"](arg1) > 0, f"Rule 13 was satisfied for padding {arg1['padding']}, but it should not have been"

if __name__ == "__main__":
    test_dist_8()