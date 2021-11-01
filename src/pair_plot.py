import matplotlib.pyplot as plt
import utils.lib as lib


# Draw large amounts of plots using scatter and histogram
def draw_pair_plot(df):
    houses, data = lib.create_house_data_dict(df)
    pos = 0
    histogram = 0

    plt.figure(figsize=(18, 10), dpi=80)
    for x in range(1, len(df.columns) - 1):
        histogram += 1
        for i in range(1, len(df.columns) - 1):
            pos += 1

            # Draw subplot and remove x/y labels
            plt.subplot(len(df.columns) - 2, len(df.columns) - 2, pos)
            plt.xticks([])
            plt.yticks([])

            # Place columns labels on extremity plots
            if i == 1:
                plt.ylabel(df.columns[x], fontsize=5)
            if x == len(df.columns) - 2:
                plt.xlabel(df.columns[i], fontsize=8)

            # Draw histogram or scatter depending on position
            for house in houses:
                if i == histogram:
                    plt.hist(data[house][df.columns[i]], alpha=0.5)
                else:
                    plt.scatter(data[house][df.columns[i]], data[house][df.columns[x]], s=0.1)
    plt.tight_layout()
    plt.show()
