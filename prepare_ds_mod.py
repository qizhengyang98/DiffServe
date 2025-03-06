import gdown
import os
import shutil

url = "https://drive.google.com/drive/folders/1gF1wKHxaA1DAnAkeDBGvuoHPS7aSRaYz?usp=sharing"
gdown.download_folder(url)

source_folder = "DiffServe_AE"
destination_folder = "models"

for file_name in os.listdir(source_folder):
    source_path = os.path.join(source_folder, file_name)
    destination_path = os.path.join(destination_folder, file_name)

    if os.path.isfile(source_path):  # Check if it's a file
        shutil.move(source_path, destination_path)
        print(f"Moved: {source_path} -> {destination_path}")

print("All files moved successfully!")
print(f"Remove {source_folder}...")
os.rmdir(source_folder)