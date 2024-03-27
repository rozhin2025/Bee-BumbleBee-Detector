import numpy as np

safe_cast_to_f16 = lambda x: np.round(x, 3).astype(np.float16)


def generate_subsets(set):
    '''
    Generates all subsets of a set
    :param set: list of elements
    :return: list of all subsets of the set
    '''
    if set == []: return [[]]
    subsets = generate_subsets(set[1:])
    return subsets + [[set[0]] + subset for subset in subsets]


if __name__ == "__main__":
    pass
