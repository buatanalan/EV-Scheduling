import yaml
import pandas as pd
import random

facilities = pd.read_excel('./resources/Sarana.xlsx')

users = [{"id": i, "car_type": "bmw i3", 'facilities_preferences' : random.choices(facilities['Nama Sarana'], k=2)} for i in range(1, 201)]

# Convert to YAML
yaml_output = yaml.dump(users, sort_keys=False)

# Save to file (optional)
with open("resources/user_data.yaml", "w") as f:
    f.write(yaml_output)

print(yaml_output)
