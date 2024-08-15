import requests
import json
import os

# Set the job ID of the parsed PDF
JOB_ID = "YOUR_JOB_ID"

# Set the API endpoint URL
API_URL = f"https://api.cloud.llamaindex.ai/api/parsing/job/{JOB_ID}/result/json"

# Set the headers for the API request
headers = {
    "accept": "application/json",
    "Content-Type": "multipart/form-data",
    "Authorization": f"Bearer {os.getenv("LLAMA_CLOUD_API_KEY")}",
}

# Send the API request to retrieve the parsed metadata
response = requests.get(API_URL, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Load the JSON data from the response
    data = json.loads(response.text)

    # Iterate over the pages in the parsed document
    for page in data["pages"]:
        # Check if the page contains any images
        if "images" in page:
            # Iterate over the images in the page
            for image in page["images"]:
                # Get the image name
                image_name = image["name"]

                # Set the image download URL
                image_url = f"https://api.cloud.llamaindex.ai/api/parsing/job/{JOB_ID}/result/image/{image_name}"

                # Send the API request to download the image
                image_response = requests.get(image_url, headers=headers)

                # Save the image to a file
                with open(image_name, "wb") as file:
                    file.write(image_response.content)

                print(f"Downloaded image: {image_name}")
else:
    print(f"Error: {response.status_code} - {response.text}")
