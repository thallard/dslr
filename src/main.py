import numpy as np
import pandas as pd
import histogram
import scatter_plot
from time import time

from src.utils.logger import log

start_time = time()


# Remove empty cells and useless columns from dataset
def clean_dataset(df):
    temp = df

    # Remove rows with empty cell(s)
    temp.replace('', np.NaN, inplace=True)
    temp.dropna(inplace=True)
    temp.reset_index(inplace=True)
    temp.drop(['Index', 'index'], axis=1, inplace=True)
    log("\033[32mClean dataset finished in : " + str(round(time() - start_time, 3)) + "s.\033[0;0m")
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
        histogram.draw_histogram(df, start_time, 'Care of Magical Creatures')
        scatter_plot.draw_scatter(df, start_time, 'Defense Against the Dark Arts', 'Astronomy')


    return 0


if __name__ == '__main__':
    main()
