import time
import matplotlib.pyplot as plt
import src.utils.lib as lib
from src.utils.logger import log


def draw_pair_plot(df, start_time):
    houses, data = lib.create_house_data_dict(df)
    pos = 0
    for x in range(1, len(df.columns) - 1):
        for i in range(1, len(df.columns) - 1):
            pos += 1
            plt.subplot(len(df.columns) - 2, len(df.columns) - 2, pos)
            plt.scatter(data[houses[0]]['Divination'], data[houses[0]]['Divination'])
    plt.show(block=False)
