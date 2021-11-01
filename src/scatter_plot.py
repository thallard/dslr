import matplotlib.pyplot as plt
import utils.lib as lib


# Draw a scatter plot with specified x and y labels
def draw_scatter(df, xlabel, ylabel):
    print("\033[35mQuelles sont les deux features qui sont semblables ?\033[0m")
    houses, data = lib.create_house_data_dict(df)

    for i in range(0, len(data)):
        plt.scatter(data[houses[i]][xlabel], data[houses[i]][ylabel], label=houses[i])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
