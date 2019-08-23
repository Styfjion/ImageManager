import requests
 
url = "http://99.47.152.154:19008/"
 
files = {'file':("test.jpg", open("./test.jpg","rb"), 'image/jpg')}
 
r = requests.post(url,files = files)
result = r.text
print(result)
