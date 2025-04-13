import requests
import os
import sys
import dotenv

secrets = dotenv.dotenv_values()

list_of_search = []

list_of_api_json = secrets['SECRET_JSONS'].split(',\n')

max_num = 0

if os.path.exists('images'):
    files = os.listdir('images')
    for file in files:
        if file != '.' or file != '..':
            try:
                file = file.split('.')
                num = int(file[0])
                if num > max_num:
                    max_num = num
            except:
                continue


for x in list_of_api_json:
    response = requests.get(x)
    data = response.json()['images_results']
    list_of_search = list_of_search + data

for item in list_of_search:
    if item['original'] is not None and item['position'] is not None:
        try:
            if '.jpg' in item['original'] or 'jpeg' in item['original']:
                os.system(f'curl -o images/{max_num + int(item["position"])}.jpg {item["original"]}')
            if '.png' in item['original']:
                os.system(f'curl -o images/{max_num + int(item["position"])}.png {item["original"]}')
            if '.webp' in item['original']:
                os.system(f'curl -o images/{max_num + int(item["position"])}.webp {item["original"]}')
            if '.gif' in item['original']:
                os.system(f'curl -o images/{max_num + int(item["position"])}.gif {item["original"]}')
            if '.bmp' in item['original']:
                os.system(f'curl -o images/{max_num + int(item["position"])}.bmp {item["original"]}')
            if '.tiff' in item['original']:
                os.system(f'curl -o images/{max_num + int(item["position"])}.tiff {item["original"]}')
        except:
            print('Error: Could not download image.')
            continue