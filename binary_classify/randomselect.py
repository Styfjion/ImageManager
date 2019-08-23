import os
import shutil
import random
sourcepath = './picture'
targetpath = './negetive'
sourceList = os.listdir(sourcepath)
indexs = random.sample(range(1,len(sourceList)),4000)
image_names = [sourceList[i] for i in indexs]
i = 0
for name in image_names:
    i += 1
    oldpath = sourcepath + '/' + name
    newpath = targetpath + '/' + name
    print(i)
    shutil.copy(oldpath,newpath)
