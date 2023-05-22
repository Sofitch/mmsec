# MMSec at FAU - Ex3
# Sofia Carvalho

import sys
import glob
import cv2
import numpy as np

from scipy import fftpack

# python .\Ex3.py [BRAND] [DEVICE]
# python .\Ex3.py Canon_Ixus70 0

# ======== Constants ========= #

IMGDIR = "./dresden_db_2_models_6_devices/"
CANON, CASIO = "Canon_Ixus70", "Casio_EX-Z150"
RED, GREEN, BLUE = 0, 1, 2
IMGSIZE = (1024, 1024)
N = 50
MAXRES = 100
MAXLSTSIZE = 10

BRAND = sys.argv[1]
DEVICE = sys.argv[2]

# ============================ #
# ========== Image =========== #
# ============================ #

def cropImage(img, goalSize=IMGSIZE):
    centerV, centerH = img.shape[0] // 2, img.shape[1] // 2
    deltaV, deltaH = goalSize[1] // 2, goalSize[0] //2
    return img[centerV-deltaV:centerV+deltaV,centerH-deltaH:centerH+deltaH]

def correctResidual(r):
    for i, row in enumerate(r):
        for j, pxl in enumerate(row):
            if abs(pxl) > MAXRES:
                r[i, j] = np.sign(pxl) * MAXRES # truncate large residuals
    return r

def getImageNoise(path):
    img = cv2.imread(path)
    crop = cropImage(img) # bound wm area for memory saving
    denoised = cv2.fastNlMeansDenoisingColored(crop, None, 9, 9, 5, 7)
    noise = crop.astype(int) - denoised
    return correctResidual(noise[:,:,BLUE]) # will use monocolored image

def getImageInfo(name):
    imgInfo = name.split("_")
    return (imgInfo[0]+'_'+imgInfo[1], imgInfo[2], imgInfo[3])


# ============================ #
# =========== PRNU =========== #
# ============================ #

def shiftImage(img, i=0, j=0):
    if i == -1:
        img = np.concatenate((img[1:], img[0, None]), axis=0)
    elif i == 1:
        img = np.concatenate((img[-1, None], img[:-1,]), axis=0)
    if j == -1:
        img = np.concatenate((img[:,1:], img[:,0, None]), axis=1)
    elif j == 1:
        img = np.concatenate((img[:,-1, None], img[:,:-1]), axis=1)
    return img

def getCrossCorrelation(a, b):
    aAux, bAux = a - a.mean(), b - b.mean()
    aAuxNorm, bAuxNorm = np.linalg.norm(aAux), np.linalg.norm(bAux)
    return ( np.transpose(aAux).dot(bAux) ) / (aAuxNorm * bAuxNorm)

def getPCE(s, f):
    # get correlations between s and f
    corrMatrix = np.zeros((3, 3))
    for i in range(-1,2):
        for j in range(-1,2):
            s_ij = shiftImage(s, i, j)
            corrMatrix[i+1, j+1] = abs(getCrossCorrelation(s_ij.flatten(), f.flatten())) ** 2
    # get PCE
    corr0 = corrMatrix[1,1]
    corrSumm = corrMatrix.sum() - corr0
    pce = corr0 / (1/8 * corrSumm)
    return pce


# ============================ #
# ========= Feedback ========= #
# ============================ #

def printStatus(i, total, curr):
    print("("+str(i+1)+"/"+str(total)+") "+curr,end="\r")

def full(lst):
    return len(lst) == MAXLSTSIZE

def isUsed(imgInfo, usedInfo):
    return imgInfo[0]==usedInfo[0] and imgInfo[1]==usedInfo[1] and int(imgInfo[2]) <= lastUsed
    
def isSameDevice(imgInfo, usedInfo):
    return imgInfo[0]==usedInfo[0] and imgInfo[1]==usedInfo[1] and int(imgInfo[2]) > lastUsed

def isSameBrand(imgInfo, usedInfo):
    return imgInfo[0]==usedInfo[0] and imgInfo[1]!=usedInfo[1]

def isDifferent(imgInfo, usedInfo):
    return imgInfo[0]!=usedInfo[0]


# ============================ #
# =========== Main =========== #
# ============================ #

# == Fingerprint Extraction == #

# get images of same device
device = BRAND + "_" + DEVICE
imgPaths = glob.glob(IMGDIR + device + "*")

# save noise residuals
noiseResiduals = []
for i in range(N):
    img = imgPaths[i]
    printStatus(i, N, img)
    noiseResiduals.append( getImageNoise(img) )
lastUsed = int(getImageInfo(img[len(IMGDIR):-4])[2])
print("\n")

# calculate fingerprint
f = np.sum(noiseResiduals, axis=0) / N

# == Fingerprint Detection === #

usedInfo = (BRAND, DEVICE, lastUsed)
testImages = [[], [], [], []]
meanPCEs = [0,0,0,0]

# get different types of images
imgPaths = glob.glob(IMGDIR + "*")
for path in imgPaths:
    imgInfo = getImageInfo(path[len(IMGDIR):-4])
    if not full(testImages[0]) and isUsed(imgInfo, usedInfo): testImages[0].append(path)
    elif not full(testImages[1]) and isSameDevice(imgInfo, usedInfo): testImages[1].append(path)
    elif not full(testImages[2]) and isSameBrand(imgInfo, usedInfo): testImages[2].append(path)
    elif not full(testImages[3]) and isDifferent(imgInfo, usedInfo): testImages[3].append(path) 

# calculate their PCEs
for i, imgList in enumerate(testImages):
    summ = 0
    for j, path in enumerate(imgList):
        printStatus(j, MAXLSTSIZE, path)
        summ += getPCE( getImageNoise(path), f)
    meanPCEs[i] = summ / len(imgList)
    print()

# print feedback
print("\nused images: " + str(meanPCEs[0]) +
      "\nsame device: " + str(meanPCEs[1]) +
      "\nsame brand: "  + str(meanPCEs[2]) +
      "\ndifferent: "   + str(meanPCEs[3]) +
      "\n")
