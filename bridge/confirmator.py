import redis
import paho.mqtt.client as mqtt
import json
import time

# --- Configuration ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

MQTT_BROKER_HOST = 'localhost'
MQTT_BROKER_PORT = 1883
MQTT_TOPIC_REQUEST = 'request/confirmation'
MQTT_TOPIC_RESPONSE = 'response/confirmation'

def get_redis_client():
    """Establishes and returns a Redis client connection."""
    try:
        r = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        r.ping()
        print(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return r
    except redis.exceptions.ConnectionError as e:
        print(f"Error connecting to Redis: {e}")
        return None

# --- Access Check Logic ---
def check_access(request_data, global_schedule):
    car_id = request_data.get('car_id')
    cs_id = request_data.get('cs_id')
    request_time_str = request_data.get('time')

    if not all([car_id, cs_id, request_time_str]):
        return False, "Invalid request data: Missing car_id, cs_id, or request_time."

    try:
        request_time = float(request_time_str)
    except ValueError:
        return False, "Invalid request_time format."

    schedule_for_car = next((entry for entry in global_schedule if entry.get("car_id") == car_id), None)

    if not schedule_for_car:
        return False, f"No schedule found for car_id: {car_id}"

    sessions = schedule_for_car.get("chargingSessions", [])
    for session in sessions:
        if (str(session.get("cs_id")) == str(cs_id) and
            session.get("start_time", 0) <= request_time <= session.get("stop_time", float("inf"))):
            return True, "Access Granted"

    return False, f"Access Denied: No matching session found for cs_id {cs_id} at time {request_time}."

def process_confirmation_requests(redis_client, mqtt_client):
    print("Starting to process confirmation requests...")

    while True:
        try:
            raw_request_data_all = redis_client.lrange('confirmation_request', 0, -1)

            raw_global_schedule_str = redis_client.get('global_schedule')
            if not raw_global_schedule_str:
                print("No global schedule found.")
                time.sleep(5)
                continue

            try:
                raw_global_schedule = json.loads(raw_global_schedule_str)
            except json.JSONDecodeError as e:
                print(f"Error decoding global schedule: {e}")
                time.sleep(5)
                continue

            for raw_request_data in raw_request_data_all:
                if not raw_request_data:
                    continue

                try:
                    request_data = json.loads(raw_request_data)
                except json.JSONDecodeError:
                    print("Invalid JSON in request. Skipping.")
                    continue

                is_accepted, message = check_access(request_data, raw_global_schedule)

                response_payload = {
                    "car_id": request_data.get('car_id'),
                    "cs_id": request_data.get('cs_id'),
                    "port_number": request_data.get("port_number"),
                    "status": "ACCEPTED" if is_accepted else "DENIED",
                    "message": message,
                    "timestamp": time.time()
                }

                mqtt_client.publish(MQTT_TOPIC_RESPONSE, json.dumps(response_payload))
                print(f"Published response to {MQTT_TOPIC_RESPONSE}: {response_payload}")

            if not raw_request_data_all:
                print("No new confirmation requests in Redis.")

            time.sleep(5)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            time.sleep(5)

# --- Main Execution ---
if __name__ == "__main__":
    redis_client = get_redis_client()
    if not redis_client:
        exit("Failed to connect to Redis. Exiting.")

    mqtt_client = mqtt.Client()

    try:
        mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"Error connecting to MQTT Broker: {e}")
        exit("Failed to connect to MQTT. Exiting.")

    print("\n--- Starting main processing loop ---")
    process_confirmation_requests(redis_client, mqtt_client)

    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("Script finished.")
