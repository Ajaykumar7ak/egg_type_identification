import os
import json

# Path to your dataset folder
dataset_path = r"C:\dataset"

# Get all subfolders (egg types)
egg_types = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
egg_types.sort()  # optional, to keep consistent order

# Create mapping: egg type -> index
class_indices = {name: idx for idx, name in enumerate(egg_types)}

# Save as JSON
with open(r"C:\veni project\class_labels.json", "w") as f:
    json.dump(class_indices, f, indent=4)

print("✅ class_labels.json created successfully!")
print(class_indices)
