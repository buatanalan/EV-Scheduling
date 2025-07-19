import yaml
import re

# ---- Load YAML ----
file_name = "test_case_5.yaml"
with open(file_name, "r") as f:
    data = yaml.safe_load(f)

# ---- Transform Data ----
for item in data:
    # 1. Change key car_id -> user_id and extract number
    if "car_id" in item:
        car_id = item.pop("car_id")
        user_id = int(re.search(r"\d+", car_id).group())  # extract number
        item["user_id"] = user_id

    # 2. Remove capacity if exists
    item.pop("capacity", None)

    # 3. Remove preferred_facilities if exists
    item.pop("preferred_facilities", None)

    # 4. Add car_type
    item["car_type"] = "bmw i3"

# ---- Save Back to YAML ----
with open(file_name, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)

print("✅ Updated YAML saved to updated_data.yaml")
