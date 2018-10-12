# regularize the image
def regularizeImage(img, size = (8, 8)):
    return img.resize(size).convert('L')

# calculate the hash value
def getHashCode(img, size = (8, 8)):

    pixel = []
    for i in range(size[0]):
        for j in range(size[1]):
            pixel.append(img.getpixel((i, j)))

    mean = sum(pixel) / len(pixel)

    result = []
    for i in pixel:
        if i > mean:
            result.append(1)
        else:
            result.append(0)
    
    return result

# compare the hash value
def compHashCode(hc1, hc2):
    cnt = 0
    for i, j in zip(hc1, hc2):
        if i == j:
            cnt += 1
    return cnt

# average hash algorithm to get the similar value
def calaHashSimilarity(img1, img2):
    img1 = regularizeImage(img1)
    img2 = regularizeImage(img2)
    hc1 = getHashCode(img1)
    hc2 = getHashCode(img2)
    return compHashCode(hc1, hc2)

__all__ = ['calaHashSimilarity']
