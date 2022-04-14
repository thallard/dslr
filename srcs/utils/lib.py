import sys
import math

# Recreate sum function
def mysum(data):
    total = 0
    for i in range(0, len(data)):
        total += data[i]
    return round(float(total), 2)


# Recreate mean function
def mymean(data):
    return round(float(mysum(data) / len(data)), 2)


# Recreate variance function
def myvariance(data):
    mean = mymean(data)
    deviations = [math.pow((x - mean), 2) for x in data]
    return round(float(mysum(deviations) / len(data)), 2)


# Recreate standard deviation function
def mystd(data):
    return round(float(math.sqrt(myvariance(data))), 2)


# Recreate min function
def mymin(data):
    min = sys.maxsize
    for i in range(0, len(data)):
        if min > data[i]:
            min = data[i]
    return round(float(min), 2)


# Recreate max function
def mymax(data):
    max = -sys.maxsize
    for i in range(0, len(data)):
        if max < data[i]:
            max = data[i]
    return round(float(max), 2)


# Recreate first quartile function
def quartile(data, coef):
    list = data.tolist()
    list.sort()

    pos = int(len(list) * coef)
    reste = len(list) * coef - pos

    if reste > 1 / 2:
        return round(list[int(pos + int(reste))], 2)
    return round(float(list[pos]), 2)


# Recreate distance between min and max
def difference(data):
    return round(float(abs(mymin(data) - mymax(data))), 2)


# Recreate ASCII to Integer
def atoi(str):
    resultant = 0
    for i in range(len(str)):
        if not str[i].isdigit():
            return resultant
        resultant = resultant * 10 + (ord(str[i]) - ord('0'))
    return resultant


# Create a dict and a string array with each name of hogwarts house
def create_house_data_dict(df):
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    houses_data = dict()

    # Fill dictionnary with dataset for each house
    for house in houses:
        houses_data[house] = df[df['Hogwarts House'] == house]
    return houses, houses_data
