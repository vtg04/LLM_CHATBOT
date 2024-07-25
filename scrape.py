import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join([p.text for p in soup.find_all('p')])
    return text

urls = ["https://www.apple.com/apple-vision-pro/", "https://support.apple.com/en-in/guide/apple-vision-pro/welcome/visionos","https://support.apple.com/guide/apple-vision-pro/read-pdfs-dev449021435/visionos","https://www.apple.com/apple-vision-pro/specs/","https://support.apple.com/en-in/117810","https://support.apple.com/en-in/guide/apple-vision-pro/tan489cfe222/visionos"]
web_texts = [scrape_website(url) for url in urls]
print(web_texts)
