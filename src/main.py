import numpy as np
import pandas as pd
from histogram import draw_histograms
from src.utils.logger import log
from time import time


start_time = time()


# Remove empty cells and useless columns from dataset
def clean_dataset(df):
    temp = df

    # Remove rows with empty cell(s)
    temp.replace('', np.NaN, inplace=True)
    temp.dropna(inplace=True)
    temp.reset_index(inplace=True)
    temp.drop(['Index', 'index'], axis=1, inplace=True)
    log("\033[32mClean dataset finished in : " + str(round(time() - start_time, 3)) + " ms.\033[0;0m")
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
        # print("\033[35mQuel cours de Poudlard a une répartition des notes homogènes entre les quatres maisons ?\033[0m")
        draw_histograms(df, start_time)

    return 0


if __name__ == '__main__':
    main()
