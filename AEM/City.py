from __future__ import annotations


# Allows us to import files from parent dir
import sys
from os.path import dirname, abspath

dir             = dirname(abspath(__file__))
parent_folder   = dirname(dir)

sys.path.append(parent_folder)
# https://stackoverflow.com/questions/16780014/import-file-from-parent-directory



import pandas   as pd
from geopy      import distance




class CityNotFound(Exception):
    pass


class MultipleCities(Exception):
    pass


class City:
    name:       str
    country:    str
    admin:      str

    lat:        float
    lon:        float

    def __init__(self, name, country, admin, lat, lon) -> None:
        self.name    = name
        self.country = country
        self.admin   = admin

        self.lat     = lat
        self.lon     = lon

    @staticmethod
    def multiple_occurences(city_name: str, country_name:str=None, state:str=None):
        cities = City.get_city_series(city_name, country_name, state)

        return len(cities) > 1

    @staticmethod
    def series2City(city: pd.Series):
        return City(city["city_ascii"], city["country"], city["admin_name"], city["lat"], city["lng"])
    

    @staticmethod
    def swap_aliases(**kwargs):
        aliases = {
            "city" : "city_ascii",
            "admin" : "admin_name"
        }

        new_kwargs = {}

        for key, value in kwargs.items():
            if key in aliases:
                key = aliases[key]

            new_kwargs[key] = value
        
        return new_kwargs
    

    @staticmethod
    def from_db(df: pd.DataFrame, handle_multiple_results:bool=False, **kwargs):
        
        new_kwargs  = City.swap_aliases(**kwargs)
        cities      = City.get_city_series(df, **new_kwargs)

        if len(cities) == 0:
            raise CityNotFound()



        if len(cities) > 1 and not handle_multiple_results:
            raise MultipleCities()
        
        # either:
        # - 1  city,  handle_multiple_requests unimportant
        # - >1 city & handle_multiple_requests == True

        if "population" in cities.columns:
            city = cities.sort_values(by="population", ascending=False).iloc[0]
        else:
            city = cities.iloc[0]

        return City.series2City(city)


    @staticmethod
    def get_city_series(df: pd.DataFrame, **kwargs):
        for key, value in kwargs.items():
            df = df[df[key] == value.title()]

        return df
    

    def distance(self, city: City) -> distance.Distance:
        return distance.geodesic((self.lat, self.lon), (city.lat, city.lon))

if __name__ == "__main__":
    city_coords = pd.read_csv(f"{parent_folder}/data/chosen_cities_final.csv")

    cities = [
        City.from_db(city_coords, city="Atlanta", handle_multiple_results=True),
        City.from_db(city_coords, city="Perth")
    ]

    print(f"Distance Between {cities[0].name} and {cities[1].name}: {round(cities[0].distance(cities[1]).km, 2)}km")