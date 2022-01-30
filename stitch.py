from re import A
import PIL
import cv2
import numpy as np
import scipy
import tensorflow as tf
import scipy.misc
import scipy.cluster
import binascii

index = 0

def getDom(img):
    img.resize((256, 256))
    ar = np.asarray(img)
    shape = ar.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

    num_colors = 5

    codes, dist = scipy.cluster.vq.kmeans(ar, num_colors)
    vecs, dist = scipy.cluster.vq.vq(ar, codes)
    counts, bins = np.histogram(vecs, len(codes))
    index_max = np.argmax(counts)
    peak = codes[index_max]
    color = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')

    dom = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))

    return dom


def colorDiff(colorA, colorB):
    diff = 0
    diff += abs(colorA[0] - colorB[0])
    diff += abs(colorA[1] - colorB[1])
    diff += abs(colorA[1] - colorB[2])
    return diff


def matchImage(imgList, color, threshold):
    global index
    
    for img in imgList:
        if(colorDiff(getDom(img), color) < threshold):
            print("Found!")
            return img

    try:
        newImg = PIL.Image.open("img/generated_images/" + getPath(index))
        imgList.append(newImg)
        index += 1
    except:
        print("Threshold too strict, loosening restrictions")
        return matchImage(imgList, color, threshold + 30)

    while(colorDiff(getDom(newImg), color) > threshold):
        # print(index)
        try:
            newImg = PIL.Image.open("img/generated_images/" + getPath(index))
            imgList.append(newImg)
            index += 1
        except:
            print("Threshold too strict, loosening restrictions")
            return matchImage(imgList, color, threshold + 30)
    
    return newImg


def getPath(index):
    path = str(index) + ".png"
    return path


path = "img/input/1.png"
img = PIL.Image.open(path)
img_arr = np.array(img.getdata())
wid, ht = img.size
threshold = 30

resize = 256

megaImg = PIL.Image.new('RGB', (wid * 256, ht * 256))
imgList = []

print(img_arr.shape)

currWid, currHt = 0, 0

for i in range(wid):
    for j in range(ht):
        domColor = img.getpixel((i, j))
        matchImg = matchImage(imgList, domColor, threshold)
        matchImg.resize((resize, resize))
        megaImg.paste(matchImg, (currWid, currHt))
        currWid += resize

        path = "img/output/"

        megaImg.save(path + str(i) + "|" + str(j) + ".png")

    currHt += resize

# cv2.imshow('', megaImg)
