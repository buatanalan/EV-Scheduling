import json
import redis
import paho.mqtt.client as mqtt
import time
from datetime import datetime
import argparse
from bridge import clear_db

#CONFIG
MQTT_BROKER = "localhost" 
MQTT_PORT = 1883
REDIS_HOST = "localhost"
REDIS_PORT = 6379

#SETUP
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
client = mqtt.Client()

def redis_safe_value(v):
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    elif isinstance(v, bool):
        return str(v).lower()
    elif isinstance(v, (int, float, str)):
        return v
    else:
        return str(v)

#MQTT CALLBACKS
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected with result code {rc}")
    client.subscribe("car/status/#")
    client.subscribe("cs/status/#")
    client.subscribe("car/finish/#")
    client.subscribe("agent/searching")
    client.subscribe("cs/report/#")
    client.subscribe("agent/schedule")
    client.subscribe("car/charging/#")
    client.subscribe("agent/request/charging")
    client.subscribe("agent/request/updates")
    client.subscribe("agent/waiting/charging")
    client.subscribe("schedule/global")
    client.subscribe("request/session/access/#")
    client.subscribe("sim/request")
    client.subscribe("sim/create_station")
    client.subscribe('setup')


def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode()

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        print(f"[WARN] Invalid JSON on topic {topic}")
        return

    timestamp = time.time()
    data["timestamp"] = timestamp

    if topic.startswith("car/status/"):
        car_id = topic.split("/")[-1]
        try:
            r.hset(f"car:status:{car_id}", mapping={
                k: redis_safe_value(v) for k, v in data.items()
            })
            print(f"[REDIS] Updated car:{car_id}")
        except Exception as e:
            print(f"[ERROR] Redis update car:{car_id} failed: {e}")
    elif topic.startswith("car/charging/"):
        car_id = topic.split("/")[-1]
        try:
            r.hset(f"car:status:{car_id}", mapping={
                k: redis_safe_value(v) for k, v in data.items()
            })
            print(f"[REDIS] Updated car:{car_id}")
        except Exception as e:
            print(f"[ERROR] Redis update car:{car_id} failed: {e}")

    elif topic.startswith("cs/status/"):
        cs_parts = topic.split("/")
        cs_id = cs_parts[2]
        port_id = cs_parts[3] if len(cs_parts) > 3 else None
        key = f"cs:{cs_id}" if port_id is None else f"cs:{cs_id}:{port_id}"
        try:
            r.hset(key, mapping={
                k: redis_safe_value(v) for k, v in data.items()
            })
            print(f"[REDIS] Updated {key}")
        except Exception as e:
            print(f"[ERROR] Redis update {key} failed: {e}")

    elif topic == "agent/searching":
        try:
            r.lpush("agent:searching_log", json.dumps(data))
            print("[REDIS] Logged agent searching event")
        except Exception as e:
            print(f"[ERROR] Redis update agent:searching failed: {e}")
    elif topic.startswith("car/finish/"):
        try:
            data = json.loads(msg.payload.decode())
            #Tambahkan timestamp
            data["timestamp"] = datetime.now().isoformat()

            topic_parts = msg.topic.split("/")
            car_id = topic_parts[-1]

            redis_key = f"user:{car_id}:history"
            r.rpush(redis_key, json.dumps(data))

            print(f"Saved data for {car_id} to Redis")
        except Exception as e:
            print(f"Error processing message: {e}")
    elif topic.startswith("cs/report/"):
        try:
            data = json.loads(msg.payload.decode())

            data["timestamp"] = datetime.now().isoformat()

            topic_parts = msg.topic.split("/")
            cs_id = topic_parts[-1]

            redis_key = f"cs:{cs_id}:history"

            r.rpush(redis_key, json.dumps(data))

            print(f" Saved report for CS {cs_id} to Redis (key: {redis_key})")

        except Exception as e:
            print(f" Error processing message: {e}")

    elif topic=="agent/schedule":
        key = "schedule:all"
        if data:
            schedules =  json.loads(msg.payload.decode())
        else:
            schedules = []

        r.set(key, json.dumps(schedules.get("schedule")))

    elif topic=="agent/request/charging":
        try : 
            key = "queue:request"
            if data:
                charging_request = json.loads(msg.payload.decode())
                print(charging_request)

                r.rpush(key, json.dumps(charging_request))
            else:
                print("No charging request data received.")
        except  Exception as e:
            print(f"Error : {e}")

    elif topic == "agent/request/updates":
        try : 
            updates = json.loads(msg.payload.decode())
            updates_list = updates.get("update")
            if not isinstance(updates_list, list):
                print("[WARN] 'update' is missing or not a list")
                return
        
            timestamp = time.time()
            current_raw = r.lrange("queue:request", 0, -1)
            current_list = [json.loads(item) for item in current_raw]

            current_map = {item['car_id']: item for item in current_list}

            for req in updates_list:
                current_map[req['car_id']] = req  

            updated_list = current_map.values()

            r.delete("queue:request")
            r.rpush("queue:request", *(json.dumps(v) for v in updated_list))

            print(f"Stored {len(updates_list)} charging requests in Redis")
        except  Exception as e:
            print(f"Error : {e}")

    elif topic=="agent/waiting/charging":
        try : 
            key_prefix = "queue:waiting"
            if data:
                payload = json.loads(msg.payload.decode())
                sub_key = payload.get("key")
                updates = payload.get("update")

                if sub_key is None or updates is None:
                    print("[WARN] Payload missing 'key' or 'update'")
                else:
                    redis_key = f"{key_prefix}:{sub_key}"
                    r.set(redis_key, json.dumps(updates))
                    print(f"[INFO] Stored waiting charge updates under {redis_key}")
            else:
                print("No waiting charge data received.")
        except  Exception as e:
            print(f"Error : {e}")

    elif topic=="schedule/global":
        try : 
            if data:
                payload = json.loads(msg.payload.decode())
                new_schedule = payload.get("new_schedule")

                r.set('schedule:global', json.dumps(new_schedule))
                print(f"[INFO] Stored waiting charge updates under global_schedule")
            else:
                print("No waiting charge data received.")
        except  Exception as e:
            print(f"Error : {e}")

    elif topic.startswith("request/confirmation/"):
        try : 
            if data:
                car_id = topic.split('/')[-1]
                payload = json.loads(msg.payload.decode())
                payload['car_id'] = car_id
                r.rpush("confirmation_request", json.dumps(payload))
                print(f"[INFO] Stored access request update")
            else:
                print("No waiting charge data received.")
        except  Exception as e:
            print(f"Error : {e}")

    elif topic.startswith('sim/request'):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))

            if payload.get("action") == "get_charging_history":
                request_id = payload.get("id")
                vehicle_id = payload.get("params", {}).get("vehicle_id")

                if not vehicle_id:
                    print(" Missing vehicle_id")
                    return

                redis_key = f"car:history:{vehicle_id}"
                raw_data = r.lrange(redis_key, 0, -1) 

                history = [json.loads(entry) for entry in raw_data]

                response = {
                    "id": request_id,
                    "data": history
                }
                response_topic = f"sim/response/{request_id}"
                client.publish(response_topic, json.dumps(response))
                print(f" Sent response to {response_topic}")

        except Exception as e:
            print(" Error handling message:", e)

    elif topic.startswith('sim/create_station'):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))

            cs_id = payload.get('stationId')

            key = f"cs:{cs_id}:info"
            name = payload.get('name')
            latitude = payload.get('latitude')
            longitude = payload.get('longitude')
            facilities = payload.get('facilities')

            r.hset(key, mapping={
                "name": name,
                "latitude": latitude,
                "longitude": longitude,
                "facilities": redis_safe_value(facilities)
            })
            ports = payload.get('ports')
            for port in ports:
                port_id = port.get('id')
                power = port.get('power')
                price = port.get('price')
                subkey = f"cs:{cs_id}:info:{port_id}"
                r.hset(subkey, mapping={
                "power": power,
                "price": price
            })


        except Exception as e:
            print(" Error handling message:", e)


def main():
    parser = argparse.ArgumentParser(description="Integration script")
    parser.add_argument("clear")
    args = parser.parse_args()

    if args.clear == "clear":
        clear_db.clear_db(r)
        clear_db.insert_static_data(r)

    # --- INIT MQTT ---
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

    print("[BRIDGE] Starting MQTT -> Redis bridge...")
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nMQTT-Redis bridge stopped gracefully by user.")

if __name__ == "__main__":
    main()
