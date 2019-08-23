from skimage import io
import os

def corp_margin(img):
	img2=img.sum(axis=2)
	(row,col)=img2.shape
	row_top=0
	raw_down=0
	col_top=0
	col_down=0
	# 750为根据255*3 = 765设置的阈值，可调整
	#确定上边界
	for r in range(0,row):
		if img2.sum(axis=1)[r]<750*col:
				row_top=r
				break
	#确定下边界
	for r in range(row-1,0,-1):
		if img2.sum(axis=1)[r]<750*col:
				raw_down=r
				break

	#确定左边界
	for c in range(0,col):
		if img2.sum(axis=0)[c]<750*row:
				col_top=c
				break
	#确定右边界
	for c in range(col-1,0,-1):
		if img2.sum(axis=0)[c]<750*row:
				col_down=c
				break	
	new_img=img[row_top:raw_down+1,col_top:col_down+1,0:3]
	return new_img, row_top, raw_down, col_top, col_down

if __name__ == '__main__':
	pathList = os.listdir('./picture')
	resultPath = './result'
	for name in pathList:
		imagePath = os.path.join('./picture',name)
		im = io.imread(imagePath)
		img_re = corp_margin(im)
		io.imsave(resultPath+'/'+name,img_re)
		clone_rate = img_re.shape[0]/im.shape[0]
		row_rate = img_re.shape[1]/im.shape[1]
		if clone_rate > 0.9 and row_rate > 0.9:
			print("无白边")
		else:
			print("有白边")