import json
import yaml

def clear_db(client):
    client.flushdb()

def insert_static_data(client):

    with open('resources/user_data.yaml', "r") as f:
        users = yaml.safe_load(f)

    with open('resources/car_data.yaml', "r") as f:
        cars = yaml.safe_load(f)

    for user in users:
        key = f"user:{user.get('id')}:info"
        car_type = user.get('car_type')
        facilities_preferences = user.get('facilities_preferences')

        client.hset(key, mapping={
            "car_type" : car_type,
            "facilities_preferences": json.dumps(facilities_preferences)
        })

    for car in cars:
        key = f"car:{car.get('id')}:info"
        type_ = car.get('type')
        year = car.get('year')
        capacity = car.get('capacity')
        efficiency = car.get('efficiency')

        client.hset(key, mapping={
            "type" : type_,
            "year" : year,
            "capacity" : capacity,
            "efficiency" : efficiency
        })

    