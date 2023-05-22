# MMSec at FAU - Ex1
# Sofia Carvalho

import sys
import random
import numpy as np
import cv2

from os import listdir
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA

# python .\Ex1.py [NUMIMAGES] [ALPHA] [CORRTRESHOLD]
# python .\Ex1.py 50 0.7 70000

# ======== Constants ========= #

IMGDIR, WMDIR, CLEANDIR = "./ucid/", "./wm/", "./clean/"
NUMIMAGES = int(sys.argv[1])
WMAREA = (384, 384)
ALPHA = float(sys.argv[2])
CORRTRESHOLD = int(sys.argv[3])

# ============================ #
# ========= Message ========== #
# ============================ #

def getRandomMessage():
    m = []
    for _ in range(3):
        new = random.choice([0,1])
        m.append(new)
    return m


# ============================ #
# ========= Carriers ========= #
# ============================ #

def newCarrier(length):
    carrier = []
    for _ in range(length):
        new = random.choice([-1,1])
        carrier.append(new)
    return carrier

def newCarriers(amount, length):
    carriers = []
    for _ in range(amount):
        new = newCarrier(length)
        carriers.append(new)
    return carriers

def checkOrthogonal(carriers):
    amount, length = len(carriers), len(carriers[0])
    for i in range(amount):
        for j in range (i+1, amount):
            product = np.dot(carriers[i],carriers[j])
            if product / length > 0.01: return False # orthogonal only if dot is less than 1%
    return True

# Get orthogonal carriers
def makeCarriers(k, N):
    carriers = newCarriers(k, N)
    while checkOrthogonal(carriers) != True:
        carriers = newCarriers(k, N)
    return carriers


# ============================ #
# ========== Image =========== #
# ============================ #

def cropImage(img, goalSize=WMAREA):
    return img[:WMAREA[1],:WMAREA[0]]

def getImageArray(path):
    img = cv2.imread(path).astype(int)
    crop = cropImage(img) # bound wm area for memory saving
    x = crop[:,:,0].flatten() # use monocolored image
    return img, x

def getNoiseResidualArray(path):
    img = cv2.imread(path)
    crop = cropImage(img) # bound wm area for memory saving
    denoised = cv2.fastNlMeansDenoisingColored(crop, None, 3, 3, 7, 11)
    residual = crop[:,:,0].flatten().astype(int) - denoised[:,:,0].flatten()
    return img, residual

def storeAlteredImage(path, imgArray, y):
    shapedY = y.reshape(WMAREA[1], WMAREA[0])
    imgArray[:WMAREA[1],:WMAREA[0],0] = shapedY
    wmImg = cv2.imwrite(path, imgArray.astype('uint8'))

def checkImgForOverflow(y):
    for i in range(len(y)):
        if y[i] > 255: y[i] = 255
        if y[i] < 0: y[i] = 0
    return y


# ============================ #
# ======== WM images ========= #
# ============================ #

# Embed message m in image x
def watermarkMessage(x, m, carriers):
    k = len(m)
    N = len(x)

    # Embedding
    summ = np.zeros(N)
    for i in range(k):
        sign = (1 if m[i] == 1 else -1)
        summ += ALPHA * sign * np.array(carriers[i])
    y = np.round(x + summ)

    # Overflow check
    y = checkImgForOverflow(y)
    
    #print("Done")
    return y


# ============================ #
# ======== Detect WM ========= #
# ============================ #

# Calculate correlation between image and carriers
def lookForEmbeddedMessage(r, carriers):
    correlations = []
    embeddedMessage = []
    
    # Calculate correlation
    for carrier in carriers:
        correlations.append( r.dot(carrier) )
    #print(correlations)

    # Reconstruct embedded message
    for corr in correlations:
        if corr > CORRTRESHOLD: embeddedMessage.append(1)
        elif corr < -CORRTRESHOLD: embeddedMessage.append(0)
    return embeddedMessage

# Conclude about watermark
def checkIfWatermark(m, susM):
    if susM == m: return 1
    elif len(susM) == 3: return 0
    else: return -1

# Detect if image is watermark 
def detectWatermark(susR, carriers, m):
    susM = lookForEmbeddedMessage(susR, carriers)
    return checkIfWatermark(m, susM)

# Print WM detection feedback
def printWMConclusion(conclusion):
    if conclusion == 1: print("Message detected!")
    elif conclusion == 0: print("WM but wrong message")
    else: print("No WM")

def printWMConclusions(conclusions):
    wm, dirty, clean = 0, 0, 0
    for c in conclusions:
        if c == 1: wm += 1
        elif c == 0: dirty += 1
        else: clean += 1
    print("Message detected in "+str(wm)+" images\n"+\
          "Wrong message detected in "+str(dirty)+" images\n"+\
          "No watermark detected in "+str(clean)+" images\n")


# ============================ #
# ======== Remove WM ========= #
# ============================ #

def normalizeVectors(vectors):
    for i, v in enumerate(vectors):
        vectors[i] = v / np.linalg.norm(v)
    return vectors

# Perform PCA to get eigen vectors
def getEigenVectors(pcaMatrix):
    pca = PCA().fit( StandardScaler().fit_transform(pcaMatrix) )
    eV = pca.components_[:3]
    return normalizeVectors(eV)

# Remove WM from image
def cleanWatermark(y, r, eV):
    summ = np.zeros(len(r))
    for v in eV:
        summ += np.rint(r.dot(v) * v)
    return checkImgForOverflow(y - summ)


# ============================ #
# ===== Recover Carriers ===== #
# ============================ #

def getCrossCorrelation(a, b):
    aAux, bAux = a - a.mean(), b - b.mean()
    aAuxNorm, bAuxNorm = np.linalg.norm(aAux), np.linalg.norm(bAux)
    return ( np.transpose(aAux).dot(bAux) ) / (aAuxNorm * bAuxNorm)

def getCarriersCorrelations(c1, c2):
    numCarriers = c1.shape[1]
    corrs = np.zeros((numCarriers, numCarriers))
    for i in range(numCarriers):
        for j in range(numCarriers):
            corrs[i, j] = getCrossCorrelation(c1[:,i], c2[:,j])
    return corrs


# ============================ #
# =========== Main =========== #
# ============================ #

imgNames = list(x[:-4] for x in listdir(IMGDIR))[:NUMIMAGES]
messages = {}
carriers = makeCarriers(3, WMAREA[0]*WMAREA[1])

# ======== WM images ========= #

for name in imgNames: 
    m = getRandomMessage()
    messages[name] = m
    path = IMGDIR+name+".tif"
    imgArray, x = getImageArray(path)

    # Watermark image
    y = watermarkMessage(x, m, carriers)
    path = WMDIR+name+".png"
    storeAlteredImage(path, imgArray, y)

print("Images successfuly watermarked!\n")

# ======== Detect WM ========= #

results = []
for name in imgNames:
    path = WMDIR+name+".png"
    #path = IMGDIR+name+".tif"
    _, susR = getNoiseResidualArray(path)
    result = detectWatermark(susR, carriers, messages[name])
    results.append(result)
printWMConclusions(results)

# ======== Remove WM ========= #

# Get eigen vectors
wmR = []
for name in imgNames:
    path = WMDIR+name+".png"
    _, r = getNoiseResidualArray(path)
    wmR.append(r)
pcaMatrix = np.array(wmR)
eV = getEigenVectors(pcaMatrix)

# Clean WM images
for i, name in enumerate(imgNames):
    path = WMDIR+name+".png"
    imgArray, y = getImageArray(path)
    cleanY = cleanWatermark(y, wmR[i], eV)
    path = CLEANDIR+name+".png"
    storeAlteredImage(path, imgArray, cleanY)

# === Check WM is removed ==== #

results = []
for name in imgNames:
    path = CLEANDIR+name+".png"
    _, cleanR = getNoiseResidualArray(path)  
    result = detectWatermark(cleanR, carriers, messages[name])
    results.append(result)
printWMConclusions(results)

# === Recover WM carriers ==== #

originalC = np.array(carriers).transpose()
subspace = np.array(eV).transpose()

ica = FastICA(n_components=3, random_state=0)
estimatedC = ica.fit_transform(subspace)
correlations = getCarriersCorrelations(originalC, estimatedC)

print("Carriers recovered! Correlations:")
for corr in correlations:
    print(tuple(round(x, 3) for x in corr))
print()
