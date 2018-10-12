import matplotlib.pyplot as plt

# regularize the image
def regularizeImage(img, size = (256, 256)):
    return img.resize(size).convert('RGB')

# draw the histogram to visualize
def drawHistogram(hg1, hg2):
    plt.plot(range(len(hg1)), hg1, color='blue', linewidth=1.5, label='img1')
    plt.plot(range(len(hg2)), hg2, color='red', linewidth=1.5, label='img2')
    plt.legend(loc='upper left')
    plt.title('Histogram Similarity')
    plt.show()

# 4x4
def splitImage(img, part_size = (64, 64)):
    w, h = img.size
    pw, ph = part_size
    data = []
    for i in range(0, w, pw):
        for j in range(0, h, ph):
            data.append(img.crop((i, j, i + pw, j + ph)).copy())
    return data

# according to the splits the dist is calculated
def calSingleHistogramSimilarity(hg1, hg2):
    if len(hg1) != len(hg2):
        raise Exception('not the same sample numbers')
    sum = 0
    for x1, x2 in zip(hg1, hg2):
        if x1 != x2:
            sum += 1 - float(abs(x1 - x2) / max(x1, x2))
        else:
            sum += 1
    return sum / len(hg1)

def calMultipleHistogramSimilarity(img1, img2):
    answer = 0
    for sub_img1, sub_img2 in zip(splitImage(img1), splitImage(img2)):
        answer += calSingleHistogramSimilarity(sub_img1.histogram(), sub_img2.histogram())
    return float(answer / 16.0)

__all__ = ['regularizeImage', 'drawHistogram', 'calMultipleHistogramSimilarity']