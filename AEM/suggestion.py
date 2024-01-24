import numpy    as np
import pandas   as pd
from enum       import Enum
from geopy      import distance


from .City               import City, CityNotFound
from data.country_data  import country_indexes, country_capital_coords

# Desired CoL at destination: -> Low, Medium, High
# Current CoL
# |
# v
# Low
# Medium
# High

cl_matrix = np.array([
    [1,     1.2,    1.4],
    [1.2,   1.4,    1.6],
    [1.4,   1.75,   2]
])
# -> means of the person


# Duration -> <1 week, <2 weeks, <4 weeks, >4 weeks
# Distance:
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


chosen_cities   = pd.read_csv("data/chosen_cities_final.csv")
all_cities      = pd.read_csv("data/worldcities.csv")
country_CoL     = pd.read_csv("data/countries_CoL.csv")

class Level(Enum):
    Dream   = -1
    Low     = 0
    Medium  = 1
    High    = 2



emotions    = ["sadness", "joy", "love", "anger", "fear", "surprise"]
categories  = ["beach", "mountain", "cultural", "wildlife", "nightlife", "festival"]



def get_available_cities(home_cl_idx: int, home_coords: int, duration: int, income: Level, desire: Level):
    """
    Takes:
    - home_cl_idx:  Index of the Cost of Living in the home town
    - home_coords:  Coordinates of the home town
    - duration:     Planned duration of the travel in weeks
    - income:       Level of income compared to the Cost of Living
    - desire:       Level of desire to spend depending on Income

    Returns a database with all the cities the user can travel to
    """

    df              = chosen_cities

    # index 0 if <1 week, index 1 if <2 weeks, index 2 if <4 weeks, index 3 if >4 weeks
    duration_idx    = (1 < duration <= 2) + (2 < duration <= 4) * 2 + (4 < duration) * 3
    
    df["user_cost"] = 0
    df["distance"]  = 0

    for idx, row in df.iterrows():
        dist = distance.geodesic(home_coords, (row["lat"], row["lng"]))

        dist_idx = (1000 < dist < 6000) + (6000 < dist) * 2

        df.at[idx, "user_cost"] = row["CoL index"] + dist_matrix[dist_idx, duration_idx]
        df.at[idx, "distance"]  = dist


    if income == Level.Dream or desire == Level.Dream:
        return df


    cl_limit        = home_cl_idx * cl_matrix[income.value, desire.value]

    

    return df[df["user_cost"] <= cl_limit]


def get_city_coords(city_name: str, country_name: str):
    try:
        city = City.from_db(all_cities, city_ascii=city_name, country=country_name)
        return city.lat, city.lon
    
    # if city not found, use capital's coords
    except CityNotFound:
        return country_capital_coords[country_name.title()]




def get_home_CoL(city_name: str, country_name: str):
    """
    Gets the Cost of Living in the user's city (or country if we don't have that info)
    """
    try:
        cities = City.get_city_series(chosen_cities, city_ascii=city_name.title(), country=country_name.title())

        if len(cities) == 0:
            raise CityNotFound()
        
        city = cities.iloc[0]
        
        return city["CoL index"]
    
    except CityNotFound:
        # get CoL from Country
        return country_indexes[country_name.title()]


def make_suggestion(user_data):
    emotional_threshold = 0.4
    country_threshold   = 0.4

    trips   = user_data["trips"]
    country = user_data["country"]
    city    = user_data["city"]

    income  = Level.__members__[user_data["standard_of_living"].capitalize()]
    desire  = Level.__members__[user_data["standard_of_holiday"].capitalize()]

    trip_duration   = user_data["duration"]
    home_cl         = get_home_CoL(city, country)
    home_coords     = get_city_coords(city, country)

    cities_in_budget = get_available_cities(home_cl, home_coords, trip_duration, income, desire)


    # determine category to discard due to not being available
    useable_categories = set()
    for category in categories:

        if any(cities_in_budget[category]):
            useable_categories.add(category)


    # emotion 
    category_emotions = {}
    country_emotions =  {}


    # extract category and country emotions
    for trip in trips:
        category            = trip["type"]
        country             = trip["country"]
        positive_emotion    = trip["love"] + trip["joy"] + trip["surprise"]

        if country not in country_emotions:
            country_emotions[country] = []

        country_emotions[country].append(positive_emotion)

        # disregard any categories that are outside the user's budget
        if category in useable_categories:

            if category not in category_emotions:
                category_emotions[category] = []

            category_emotions[category].append(positive_emotion)


    # get average category emotions
    category_emotions_avg = {}    
    for category, emotions in category_emotions.items():
        category_emotions_avg[category] = np.mean(emotions)
    
    highest_emotion         = max(category_emotions_avg.values())

    final_categories        = []
    for category, avg in category_emotions_avg.items():
        if avg + 0.1 >= highest_emotion:
            final_categories.append(category)
    

    cities_in_budget = cities_in_budget[cities_in_budget[final_categories].any(axis=1)]


    # get average country emotions
    country_emotions_avg = {}
    for country, emotions in country_emotions.items():
        country_emotions_avg[country] = np.mean(emotions)


    best_countries = dict(sorted(country_emotions_avg.items(), key=lambda item: item[1], reverse=True))

    # if a city in a visited country (that the user enjoyed) is already there, we suggest that
    for country, avg in best_countries.items():
        if avg < country_threshold:
            break

        if country in cities_in_budget["country"].values:
            cities_in_budget = cities_in_budget[cities_in_budget["country"] == country]
            break

    else:
        if trip_duration <= 1:
            max_dist = 1000
        elif trip_duration <= 2:
            max_dist = 3000
        elif trip_duration <= 4:
            max_dist = 6000
        else:
            max_dist = float("+inf")
        
        cities_in_budget = cities_in_budget[cities_in_budget["distance"] <= max_dist]


    # maximise spending if either income or desire is high
    ascending = not (income in [Level.High, Level.Dream] or desire in [Level.High, Level.Dream])

    return cities_in_budget.sort_values(by="user_cost", ascending=ascending).iloc[0]







# for each trip:
    # record negative and positive emotions
    # sum (sadness, anger, fear) and (joy, love, surprise)


# identify highest category first (only consider categories that fit within budget)

# check whether there is a country they really like

    # if yes
        # suggest city from that country (maybe identify a city they have not yet been to)

    # otherwise
        # filter countries by distance

        # if <1 week: <1000 dist
        # if <2 week: <3000 dist
        # if <4 week: <6000 dist
        # if >4 week: >6000 dist

# if Low/Medium desire to spend
    # minimize CoL by the end

# otherwise (High/Dream)
    # maximize CoL






# get_available_cities(83.1, (51.51, -0.13), 4, Level.Medium, Level.High)




history = {
    'name': 'My name is Pracar.',
    'country': 'Netherlands',
    'city': "I'm stood down.",
    'standard_of_living': 'dream',
    'standard_of_holiday': 'dream',
    'duration': 2.142857142857143,
    'trips': [
        {
            'type': 'beach',
            'country': 'United States',
            'date': ['2015',
            '2015',
            '2015'],
            'sadness': 0.07724712000574385,
            'joy': 0.13651098254748756,
            'love': 0.0015944462269544602,
            'anger': 0.6894029919760569,
            'fear': 0.049038076094218674,
            'surprise': 0.006420567908457349
        },
        {
            'type': 'nightlife',
            'country': 'Croatia',
            'date': ['2020'],
            'sadness': 0.028967636853456498,
            'joy': 0.7850642719268799,
            'love': 0.0019789766520261765,
            'anger': 0.08799616503715516,
            'fear': 0.03722075480222702,
            'surprise': 0.027272100105881693
        },
        {
            'type': 'nightlife',
            'country': 'India',
            'date': ['2016'],
            'sadness': 0.013358872607350351,
            'joy': 0.8816306321280344,
            'love': 0.0012592091225087643,
            'anger': 0.036977014149938314,
            'fear': 0.013566342632685389,
            'surprise': 0.025207898346441132
        }
    ]
}


if __name__ == "__main__":
    suggestion = make_suggestion(history)
    print(suggestion)