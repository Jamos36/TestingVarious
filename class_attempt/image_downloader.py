## image downloader

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Function to download an image
def download_image(url, folder):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        filename = os.path.join(folder, url.split("/")[-1])
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {url}")

# Main function to scrape and download images
def download_images_from_webpage(url, output_folder):
    # Make HTTP request
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve the webpage.")
        return

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all image tags
    img_tags = soup.find_all('img')

    # Create folder for downloaded images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Download all images
    for img in img_tags:
        img_url = img.get('src')
        if img_url:
            # Make the URL absolute by joining it with the base URL
            img_url = urljoin(url, img_url)
            download_image(img_url, output_folder)

# Replace with the webpage URL you want to scrape
webpage_url = "https://www.sparkladies.com/article-308.html"
# Replace with your desired output folder
output_folder = r"C:\Users\Jamos\OneDrive\Desktop\Kang_In_Kyung"

download_images_from_webpage(webpage_url, output_folder)
