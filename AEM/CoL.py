import numpy    as np
import pandas   as pd

from City       import City


# Desired CoL at destination: -> Low, Medium, High
# Current CoL
# |
# v
# Low
# Medium
# High

col_matrix = np.array([
    [1,     1.2,    1.4],
    [1.2,   1.4,    1.6],
    [1.4,   1.75,   2]
])
# -> means of the person


# Distance -> <1 week, <2 weeks, <4 weeks, >4 weeks
# Duration:
# |
# v
# <1000
# <6000
# >6000

dist_matrix = np.array([
    [20,    10,     0,      0],
    [50,    30,     10,     5],
    [100,   70,     50,     20]
])
# -> Limit CoL of travel city (any cities over this CoL are too expensive)


df = pd.read_csv("data/cities_final_admin.csv")

cities = []

for idx, row in df.iterrows():
    city = City(row["City"], row["Country"], row["Latitude"], row["Longitude"], row["Admin"])

    cities.append(city)


user = {
    "Income" : None,
    "Want to spend" : None
}



pass