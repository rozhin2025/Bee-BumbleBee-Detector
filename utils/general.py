import numpy as np
import pandas as pd

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


def convert_classification_report_to_df(report_dict):
    '''
    Converts the classification report dictionary to a pandas DataFrame
    :param report_dict: classification report dictionary
    :return: pandas DataFrame
    '''
    record = {}
    for k, v in report_dict.items():
        if isinstance(v, dict):
            for k_, v_ in v.items():
                record[(k, k_)] = v_
        else:
            record[('', k)] = v
    record = pd.DataFrame.from_records(record, index=[0])
    record.columns = pd.MultiIndex.from_tuples(record.columns)
    return record


if __name__ == "__main__":
    pass
