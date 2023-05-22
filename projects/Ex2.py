# MMSec at FAU - Ex2
# Sofia Carvalho

import base64
import hashlib
import random
import sys
from os import listdir

import numpy as np
import wikipedia
from Crypto import Random
from Crypto.Cipher import AES
from PIL import Image
from scipy.stats import chisquare

# python .\Ex2.py [PAYLOAD] [MODE]
# python .\Ex2.py 1 lsb
# python .\Ex2.py 0.3 ham

# ======== Constants ========= #

COVERDIR, LSBDIR, HAMDIR = "./ucid/", "./stego/lsb/", "./stego/hamming/"
TEXTLIMIT = 500
LSB = "lsb"
HAMMING = "ham"
payload = float(sys.argv[1])
mode = sys.argv[2]

# ============================ #
# ========= Message ========== #
# ============================ #

def getArbitraryMessage(wikiPage):
    text = wikipedia.page(wikipedia.search(wikiPage)[0], auto_suggest=False).content * 1000
    start, end = -1, -1
    while (start == -1 or end == -1):
        r_start = random.randrange(0, len(text) - TEXTLIMIT)
        start = text.find(('.'), r_start) + 2
        r_end = random.randrange(start, len(text))
        end = text.find(('.'), r_end) + 1
        if end >= start + TEXTLIMIT: end = start + TEXTLIMIT
    return text[start:end]


# ============================ #
# ========== Crypto ========== #
# ============================ #

def newKey(keyWord):
    return hashlib.sha256(keyWord.encode()).digest()

def pad(data):
    length = 16 - (len(data) % AES.block_size)
    return data + bytes([length])*length

def unpad(data):
    return data[:-data[-1]]

def encrypt(text, key):
    data = pad(text.encode())
    iv = Random.new().read(AES.block_size)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    # return base64.b64encode(iv + cipher.encrypt(data))
    return iv + cipher.encrypt(data)

def decrypt(encrypted, key):
    # encrypted = base64.b64decode(encrypted)
    iv = encrypted[:AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    data = cipher.decrypt(encrypted[AES.block_size:])
    return unpad(data).decode('utf-8')


# ============================ #
# ========== Image =========== #
# ============================ #

def getImageArray(path):
    img = Image.open(path)
    array = np.array(list(img.getdata()))
    return img.size, array, np.array(array[:,0]) # will use redscale image

def storeStego(path, cArray, cShape, stegoPxls):
    stegoArray = cArray.copy()
    stegoArray[:,0] = stegoPxls
    assert(all(stegoArray[:,0] == stegoPxls))
    stegoArray = stegoArray.reshape(cShape[1], cShape[0], 3)
    stego = Image.fromarray(stegoArray.astype('uint8'))
    stego.save(path, "PNG")
    print("\n> Stego image stored!\n" + message)


# ============================ #
# ==== Bit Aux Functions ===== #
# ============================ #

# convert byteStr to bits
def bytesToBitList(byteStr): 
    bitList = []
    for byte in byteStr:
        bitList += list( map( int, str(bin(byte))[2:].zfill(8) ))
    return bitList

# convert bits to byteStr
def bitListToBytes(bitList):
    bitStr = ''.join(str(bit) for bit in bitList)
    return int(bitStr, 2).to_bytes(_mSize // 8, byteorder='big')

# convert int to bits
def intToBitList(n, dim):
    return list( str(bin(n))[2:].zfill(dim) )

# get least significant bit
def lsb(pxl):
    return pxl & 1

def isPowerOfTwo(n):
    return ((n & (n-1)) == 0) and n != 0


# ============================ #
# ====== LSB Embedding ======= #
# ============================ #

# ======== Embedding ========= #

def lsbEmbedding(mBits, cPxls, payload):
    # size of mBits needs to to match size of cPxls
    size = int(payload * len(cPxls))
    mBits *= size // len(mBits)
    mBits += mBits[:size-len(mBits)]
    # embed message in lsb using payload bpp
    for i in range(size):
        k = int(i / payload) # embed every k pxl
        cPxls[k] = 2 * (cPxls[k] // 2) + mBits[i]
    return cPxls

# ======== Decoding ========= #

def lsbDecoding(imgPxls, payload):
    # get least significant bit of each pixel
    size = int(payload * len(imgPxls))
    mBits = []
    for i in range(size):
        k = int(i / payload) # every k pxl embeds a msg bit
        bit = lsb(imgPxls[k]) # get LSB
        mBits.append(bit)
    print("\n> Message obtained! Payload:", len(mBits) / len(imgPxls))
    return mBits


# ============================ #
# ==== Hamming Embedding ===== #
# ============================ #

def findHammingLength(payload): 
    p = 3
    currentPL = p / (2**p - 1)
    while True: # this loop will likely finish on the first iters
        l = 2**p - 1
        expectedPL = p / l
        if abs(payload-expectedPL) > abs(payload-currentPL): return p-1
        currentPL = expectedPL
        p += 1

# dot product of 2 "all bit" matrices
def bitMatrixDot(a, b):
    a_b = np.dot(a, b)
    return [lsb(el) for el in a_b]

# sum of 2 "all bit" matrices
def bitMatrixAdd(a, b):
    a_b = np.add(a, b)
    return [lsb(el) for el in a_b]

def getHammingMatrix(l, p):
    P = np.ones((p, l-p))
    col = 0
    for i in range(1, l+1):
        if isPowerOfTwo(i): continue
        P[:,col] = intToBitList(i, p)
        col += 1
    return np.concatenate((P, np.identity(p)), axis=1).astype(int)

# ======== Embedding ========= #

def getEmbeddingBit(x, m, H):
    H_x = bitMatrixDot(H, x)
    H_e = bitMatrixAdd(m, H_x)
    if all(el == 0 for el in H_e): return -1 # no need to change
    for col in range(H.shape[1]):
        if all(H[:,col] == H_e): return col # need to change bit #col
    raise Exception("Error in Hamming stego.")

def hammingEmbedding(mBits, cPxls, l, p):
    # size of mBits needs to be multiple of p 
    toFill = p - (_mSize % p)
    mBits = mBits + mBits[:toFill]
    # get hamming matrix
    H = getHammingMatrix(l, p)
    # embed message as hamming
    i, j = 0, 0
    while j < len(mBits):
        x = [lsb(pxl) for pxl in cPxls[i:i+l]] # codeword
        m = mBits[j:j+p] # message to embed
        e = getEmbeddingBit(x, m, H)
        if e >= 0: cPxls[i+e] = cPxls[i+e] ^ 1 # change necessary bit
        i, j = i+l, j+p
    return cPxls

# ======== Decoding ========= #

def hammingDecoding(mSize, imgPxls, l, p):
    # get hamming matrix
    H = getHammingMatrix(l, p)
    # get embedded message in bits
    mBits = []
    i, j = 0, 0
    while j < mSize:
        x = [lsb(pxl) for pxl in imgPxls[i:i+l]] # codeword
        m = bitMatrixDot(H, x) # received message
        mBits += m
        i, j = i+l, j+p
    print("\n> Message obtained! Payload:", p / l)
    return mBits


# ============================ #
# === Detect LSB Embedding === #
# ============================ #

def chiSquareTest(pxls):
    # bins for pixel intensities
    obs, exp = np.zeros(128), np.zeros(128)
    for pxl in pxls:
        exp[pxl // 2] += 0.5
        if pxl % 2 == 0: obs[pxl // 2] += 1
    # normalize data
    exp_ratios = exp / np.sum(exp)
    exp = exp_ratios * obs.sum()
    # perform test
    return chisquare(obs, f_exp=exp)

def testForLSBEmbedding(path):
    _, _, pxls = getImageArray(path)
    _, pValue = chiSquareTest(pxls)
    pValue = round(pValue, 3)
    # print conclusions
    if (pValue > 0.05):
        print("\n> Image \"" + path + "\" is stego (p-value of " + str(pValue) +").")
    else:
        print("\n> Image \"" + path + "\" is not stego (p-value of " + str(pValue) +").")
    

# ============================ #
# =========== Main =========== #
# ============================ #

# =========== Init =========== #

if mode == HAMMING:
    p = findHammingLength(payload) # parity bits
    print("\n> Using (" + str(2**p-1) + "," +  str(2**p-1-p) + \
          ") Hamming code to obtain closest possible payload")

# get encrypted message
message = getArbitraryMessage("Death Note")
_key = newKey("batata")
encryptedMessage = encrypt(message, _key)
mBits = bytesToBitList(encryptedMessage)
_mSize = len(mBits)

# get cover image
covers = list(x[:-4] for x in listdir(COVERDIR))
_name = covers[1] # using first image
path = COVERDIR + _name + ".tif"
cShape, cArray, cPxls = getImageArray(path)

print(_mSize, len(cPxls))

# ======== Embedding ========= #

if mode == LSB:
    lsbPxls = lsbEmbedding(mBits, cPxls, payload)
    path = LSBDIR + _name + ".png"
    storeStego(path, cArray, cShape, lsbPxls)

elif mode == HAMMING:  
    hammingPxls = hammingEmbedding(mBits, cPxls, 2**p-1, p)
    path = HAMDIR + _name + ".png"
    storeStego(path, cArray, cShape, hammingPxls)

else: raise Exception("Unknown mode.")

# ======== Decoding ========= #

if mode == LSB:
    path = LSBDIR + _name + ".png"
    imgShape, imgArray, imgPxls = getImageArray(path)
    mBits = lsbDecoding(imgPxls, payload)[:_mSize]

elif mode == HAMMING:    
    path = HAMDIR + _name + ".png"
    imgShape, imgArray, imgPxls = getImageArray(path)
    mBits = hammingDecoding(_mSize, imgPxls, 2**p-1, p)[:_mSize]

else: raise Exception("Unknown mode.")

# ====== Print Results ======= #

encryptedMessage = bitListToBytes(mBits)
message = decrypt(encryptedMessage, _key)
print(message)

# === Detect LSB Embedding === #

if mode == LSB:
    # test detector with stego image
    path = LSBDIR + _name + ".png"
    testForLSBEmbedding(path)
    # test detector with natural image
    path = COVERDIR + _name + ".tif"
    testForLSBEmbedding(path)

# Detects LSB embedding until payload of ~0.3 bpp

print()
