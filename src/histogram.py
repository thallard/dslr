import time
import src.utils.lib as lib
import matplotlib.pyplot as plt
import numpy as np
from src.utils.logger import log


# Draw histogram with same values between each hogwarts house
def draw_histogram(df, start_time, label):
    print("\033[35mQuel cours de Poudlard a une répartition des notes homogènes entre les quatres maisons?\033[0m")
    houses, houses_data = lib.create_house_data_dict(df)

    # Draw histograms
    for i in range(0, len(houses_data)):
        plt.hist(houses_data[houses[i]]['Care of Magical Creatures'], alpha=0.5, label=houses[i])
    plt.legend()
    plt.xlabel('Marks')
    plt.ylabel('Frequency')
    plt.title(label)
    plt.show()

    log("\033[32mDraw histograms finished in : " + str(round(time.time() - start_time, 3)) + "s.\033[0;0m")

