# regularize the image
def regularizeImage(img, size = (9, 8)):
    return img.resize(size).convert('L')

# calculate the hash value
def getHashCode(img, size = (9, 8)):

    result = []
    for i in range(size[0] - 1):
        for j in range(size[1]):
            current_val = img.getpixel((i, j))
            next_val = img.getpixel((i + 1, j))
            if current_val > next_val:
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

# different hash algorithm to get the similar value
def caldHashSimilarity(img1, img2):
    img1 = regularizeImage(img1)
    img2 = regularizeImage(img2)
    hc1 = getHashCode(img1)
    hc2 = getHashCode(img2)
    return compHashCode(hc1, hc2)

__all__ = ['caldHashSimilarity']