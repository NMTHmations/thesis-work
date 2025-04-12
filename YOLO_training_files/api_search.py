import requests
import os
import dotenv

secrets = dotenv.dotenv_values()

list_of_search = []

list_of_api_json = [secrets['SECRET_JSONS'].strip('').split('\n')]
print(list_of_api_json)


for x in list_of_api_json:
    response = requests.get(x)
    data = response.json()['images_results']
    list_of_search = list_of_search + data

for item in list_of_search:
    if item['original'] is not None and item['position'] is not None:
        if '.jpg' in item['original'] or 'jpeg' in item['original']:
            os.system(f'curl -o images/{item["position"]}.jpg {item["original"]}')
        if '.png' in item['original']:
            os.system(f'curl -o images/{item["position"]}.png {item["original"]}')
        if '.webp' in item['original']:
            os.system(f'curl -o images/{item["position"]}.webp {item["original"]}')
        if '.gif' in item['original']:
            os.system(f'curl -o images/{item["position"]}.gif {item["original"]}')
        if '.bmp' in item['original']:
            os.system(f'curl -o images/{item["position"]}.bmp {item["original"]}')
        if '.tiff' in item['original']:
            os.system(f'curl -o images/{item["position"]}.tiff {item["original"]}')