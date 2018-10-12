from PIL import Image,ImageFilter
from multiprocessing import Process
import histogram as htg
import aHash as ah
import pHash as ph
import dHash as dh
import cv2

if __name__ == '__main__':

    # read image files
    img1 = Image.open('images/cow_back3.jpg').resize((224, 224),Image.ANTIALIAS).convert('L')
    img2 = Image.open('images/cow_back4.jpg').resize((224, 224),Image.ANTIALIAS).convert('L')
    # img1.thumbnail((299, 299))
    # img3 = img1.filter(ImageFilter.SHARPEN)
    # img1.show()
    # img2.show()
    # img3.show()

    # Histogram Similarity Calculation
    # regularize the images
    img1_htg = htg.regularizeImage(img1)
    img2_htg = htg.regularizeImage(img2)
    
    hg1 = img1_htg.histogram()
    # print(img1.histogram())
    print('img1 sample points {} '.format(len(hg1)))
    hg2 = img2_htg.histogram()
    # print(img2.histogram())
    print('img2 sample points {} '.format(len(hg2)))

    # draw the histogram in a no-blocking way
    sub_thread = Process(target=htg.drawHistogram, args=(hg1, hg2,))
    sub_thread.start()

    # print the histogram similarity
    print('依据图片直方图距离计算相似度：{}'.format(htg.calMultipleHistogramSimilarity(img1_htg, img2_htg)))

    # aHash Calculation
    print('依据平均哈希算法计算相似度：{}/{}'.format(ah.calaHashSimilarity(img1, img2), 64))

    # pHash Calculation
    print('依据感知哈希算法计算相似度：{}/{}'.format(ph.calpHashSimilarity(img1, img2), 64))

    # dHash Calculation
    print('依据差异哈希算法计算相似度：{}/{}'.format(dh.caldHashSimilarity(img1, img2), 64))
