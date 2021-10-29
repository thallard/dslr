import time
from src.utils.logger import log


def draw_histograms(df, start_time):
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    houses_data = dict()

    # Fill dictionnary with dataset for each house
    for house in houses:
        houses_data[house] = df[df['Hogwarts House'] == house]

    plt.hist()
    log("\033[32mDraw histograms finished in : " + str(round(time.time() - start_time, 3)) + " ms.\033[0;0m")
    return 1

