import requests 
import time 
import concurrent.futures   # for thread 

img_urls = [
    'https://images.unsplash.com/photo-1516117172878-fd2c41f4a759',
    'https://images.unsplash.com/photo-1524429656589-6633a470097c'
    ]

t1 = time.perf_counter()

# download image one at a time, not using thread 
for img_url in img_urls: 
    img_bytes = requests.get(img_url).content 
    img_name = img_url.split('/')[3]
    img_name = f'{img_name}.jpg'
    with open(img_name, 'wb') as img_file:
        img_file.write(img_bytes)
        print(f'{img_name} was downloaded...')
t2 = time.perf_counter()
print(f'Finished in {t2-t1} seconds')


# Use thread 
# wrap download image in a function
def download_image(img_url):
    img_bytes = requests.get(img_url).content 
    img_name = img_url.split('/')[3]
    img_name = f'{img_name}.jpg'
    with open(img_name, 'wb') as img_file:
        img_file.write(img_bytes)
        print(f'{img_name} was downloaded...')

# create thread pool executor, download asynchronously 
# image download is IO bound work 
t1 = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(download_image, img_urls)   # download using thread 
t2 = time.perf_counter()
print(f'Finished in {t2-t1} seconds')

# for CPU bound work, it is better to uses multi-processing modules 
# threading can slow because it has to manage thread itself 
