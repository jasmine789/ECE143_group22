import pandas as pd
import os
import collections


def format_read(*args):
    """
    Function that reads the data and print out the info of the given dataset in a formatted way
    :param args: given dataset path
    :type args: str
    :return: a list of dataset
    """
    assert isinstance(args, tuple) and len(args) > 0

    print_txt = collections.defaultdict(list)
    data = []

    for i, path in enumerate(args):
        assert isinstance(path, str)
        print_txt['Name'].append(path.split('/')[-1])
        print_txt['Size'].append(os.path.getsize(path))
        data.append(pd.read_csv(path))
        print_txt['Num_data'].append(len(data[i]))

    for txt in print_txt:
        print(('{:18}\t'*(len(print_txt[txt])+1)).format(txt, *print_txt[txt], " "))

    return data


if __name__ == '__main__':
    # load csv files
    data_path = ['Killings by State.csv', 'All_killing.csv']
    killing_by_PD, killing_by_state, all_killing = format_read('Killings by PD.csv')