# In line 11 You have to place your json file path of required batch for which you have to download the PDF's
# In line 14 place the path of your folder in which you have to store the downloaded pdf files



import json
import os
import requests

# Replace 'your_file_path.json' with the actual path to your JSON file
file_path = '/mnt/dssk/data_rw/annotated_data/Batch5.json'

# Create a folder for storing PDFs if it doesn't exist
pdf_folder = '/mnt/dssk/data_rw/annotated_data/pdfs'
os.makedirs(pdf_folder, exist_ok=True)

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

for document in data:
    if document["type"] == "Synthetic":
        pdf_url = document["uri"]

        filename = os.path.join(pdf_folder, os.path.basename(pdf_url))

        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(filename, 'wb') as pdf_file:
                pdf_file.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download: {pdf_url}")
