import PIL.Image as Image
import numpy as np
import os
import random
import cv2

IMAGES_PATH = './picture'
IMAGE_SIZE = 200  # 每张小图片的大小
IMAGE_ROW = 3  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 3  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = './randomresult'  # 图片转换后的地址
if not os.path.isdir(IMAGE_SAVE_PATH):
    os.mkdir(IMAGE_SAVE_PATH)


# 定义图像拼接函数
def image_compose(image_names, number):
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, 
                                 IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + '/' + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize((IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, 
                           (y - 1) * IMAGE_SIZE))
    # to_image.save(IMAGE_SAVE_PATH + '/new-randomfour' + str(number) + '.jpg')
    return to_image

# 在生成的拼图的缝隙划线
def drawline(img,color,thickness):
    startList = []
    endList = []
    row, column, channel = img.shape
    sub_column = column//IMAGE_COLUMN # 200
    sub_row = row//IMAGE_ROW # 400
    
    for i in range(1,IMAGE_ROW):
        startList.append((0, i*sub_row))
        endList.append((column, i*sub_row))
    for j in range(1,IMAGE_COLUMN):
        startList.append((j*sub_column, 0))
        endList.append((j*sub_column, row))
    lineType = 4
    for i in range(len(startList)):
        cv2.line(img, startList[i], endList[i], color, thickness, lineType)
    return img

if __name__ == "__main__":
    imagePathList = os.listdir(IMAGES_PATH)
    length = 10
    qujian = len(imagePathList)//length
    # 随机生成缝隙的宽度
    thickness_list = [6,8,12,14,16,18]
    for i in range(qujian):
        random_idx= random.sample(range(i*length,(i+1)*length),IMAGE_ROW*IMAGE_COLUMN)
        image_names = [imagePathList[i] for i in random_idx]
        pintu = image_compose(image_names,i)
        pintu = cv2.cvtColor(np.asarray(pintu),cv2.COLOR_RGB2BGR)
        # 设置缝隙的颜色，黑色或者白色
        if i <= qujian//3:
            color = (0,0,0)
        else:
            color = (255,255,255)
        thickness = thickness_list[i%6]
        img = drawline(pintu,color,thickness)
        cv2.imwrite(IMAGE_SAVE_PATH + '/' + 'widenine' + str(i) + '.jpg',img)

