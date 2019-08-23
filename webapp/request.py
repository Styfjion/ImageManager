import requests
 
url = "http://99.47.152.154:19007/"
 
files = {'file1':("picture.jpg", open("./picture.jpg","rb"), 'image/jpg'),"file2":("mask.jpg",open("./mask.jpg","rb"),"image/jpg")}
 
r = requests.post(url,files = files)
result = r.text
print(result)
