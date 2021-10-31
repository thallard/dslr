import time
import matplotlib.pyplot as plt
import src.utils.lib as lib
from src.utils.logger import log


# Draw a scatter plot with specified x and y labels
def draw_scatter(df, start_time, xlabel, ylabel):
    print("\033[35mQuelles sont les deux features qui sont semblables ?\033[0m")
    houses, data = lib.create_house_data_dict(df)

    for i in range(0, len(data)):
        plt.scatter(data[houses[i]][xlabel], data[houses[i]][ylabel], label=houses[i])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    log("\033[32mDraw scatter plot finished in : " + str(round(time.time() - start_time, 3)) + "s.\033[0;0m")
