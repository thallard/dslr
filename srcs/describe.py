import pandas as pd
import utils.lib as utils

if __name__ == "__main__":
    df = pd.read_csv('../datasets/dataset_train.csv')

    # Clean dataset
    df = df.drop(['Index', 'Best Hand', 'Birthday', 'First Name', 'Last Name', 'Hogwarts House'], axis=1)
    for data in df:
        df = df[df[data].notna()]

    df.reset_index(inplace=True)

    # Prepare evaluations
    methods = [('Count', len), ('Mean', utils.mymean), ('STD', utils.mystd), ('Min', utils.mymin),
               ('25%', utils.quartile), ('50%', utils.quartile),
               ('75%', utils.quartile), ('Max', utils.mymax), ('Diff', utils.difference)]

    # Print header of columns names
    print(end='      ')
    for data in df:
        if len(data) > 8:
            print(data[:8] + "...", end=str((13 - len(data[:8] + "...")) * " "))
        else:
            print(data, end=str((13 - len(data)) * " "))
    print()

    # Evaluate each column
    for name, method in methods:
        print(name, end=str((6 - len(name)) * " "))
        for data in df:
            # Special case for quartiles
            if method == utils.quartile:
                res = method(df[data], utils.atoi(name) / 100)
            else:
                res = method(df[data])

            # Print with correct padding
            print(res, end=str((13 - len(str(res))) * " "))
        print()
