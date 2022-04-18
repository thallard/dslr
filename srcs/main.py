import time
import numpy as np
import pandas as pd
from plots import histogram, pair_plot, scatter_plot


# Remove empty cells and useless columns from dataset
def clean_dataset(df):
    temp = df

    # Remove rows with empty cell(s)
    temp.replace('', np.NaN, inplace=True)
    temp.dropna(inplace=True)
    temp.reset_index(inplace=True)
    temp.drop(['Index', 'index', 'Birthday', 'Best Hand', 'Last Name', 'First Name'], axis=1, inplace=True)
    return temp


# Main function
def main():
    df = None
    try:
        df = pd.read_csv('../datasets/dataset_train.csv')
    except:
        print("\033[31mImpossible to read data file.\033[0m")
        exit(1)
    finally:
        df = clean_dataset(df)
        histogram.draw_histogram(df, time.time())
        scatter_plot.draw_scatter(df, 'Defense Against the Dark Arts', 'Astronomy')
        pair_plot.draw_pair_plot(df)
    return 0


if __name__ == '__main__':
    main()
