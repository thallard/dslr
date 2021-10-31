import lib
import sys


# Recreate sum function
def mysum(data):
    total = 0
    for i in range(0, len(data)):
        total += data[i]
    return total


# Recreate mean function
def mymean(data):
    return sum(data) / len(data)


# Recreate variance function
def myvariance(data):
    mean = mymean(data)
    deviations = [lib.pow((x - mean), 2) for x in data]
    return mysum(deviations) / len(data)


# Recreate standard deviation function
def mystd(data):
    return lib.sqrt(myvariance(data))


# Recreate min function
def mymin(data):
    res = sys.maxsize
    for i in range(0, len(data)):
        if res > data[i]:
            res = data[i]
    return res


# Recreate max function
def mymax(data):
    res = -sys.maxsize
    for i in range(0, len(data)):
        if res < data[i]:
            res = data[i]
    return res


# Recreate percentile function
def mypercentile(data, coef):
    list_data = data.tolist()
    list_data.sort()

    pos = int(len(list_data) * coef)
    reste = len(list_data) * coef - pos

    if reste > 1 / 2:
        return list_data[int(pos + (1 - reste))]
    return list_data[pos]


# Create a dict and a string array with each name of hogwarts house
def create_house_data_dict(df):
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    houses_data = dict()

    # Fill dictionnary with dataset for each house
    for house in houses:
        houses_data[house] = df[df['Hogwarts House'] == house]
    return houses, houses_data
