# MmSec projects (2022)

Three projects - respectively on watermarking, steganography, and fingerprints - developed for the Multimedia Security Course at FAU.

## Exercise 1

The first project focuses on understanding the vulnerabilities of additive spread spectrum watermarks by conducting a concrete attack. We embed a watermark on an image using Code Division Multiple Access (CDMA), remove the watermark from the image using Principal Components Analysis (PCA), and then use Independent Component Analysis (ICA) to estimate the original watermark carriers. A more detailed description of the project can be found on [e1_watermarking.pdf](https://github.com/Sofitch/mmsec/blob/main/descriptions/e1_watermarking.pdf).

To run:

    python .\Ex1.py [NUMIMAGES] [ALPHA] [CORRTRESHOLD]

> Suggestion: NUMIMAGES, ALPHA, CORRTRESHOLD = 50, 0.7, 70000


## Exercise 2

The second project aims to understand the strengths and limitations of LSB (Least Significant Bit) steganography. We experiment with naive embedding and Hamming-encoded embedding. For both methods, we embed a message in an image, and then decode that message to verify the embedding's effectiveness. We further experiment with using a χ2 (chi-square) test to detect the presence of LSB embedding on an image. A more detailed description of the project can be found on [e2_steganography.pdf](https://github.com/Sofitch/mmsec/blob/main/descriptions/e2_steganography.pdf).

To run:

    python .\Ex2.py [PAYLOAD] [MODE]

> Suggestion: PAYLOAD, MODE = 1, lsb and PAYLOAD, MODE = 0.3, ham


## Exercise 3

This project expermients with PRNU fingerprinting on a subset of the *Dresden Image* database, aiming to evaluate the effectiveness of PRNU fingerprinting in device identification. A more detailed description of the project can be found on [e3_prnu_fingerprint.pdf](https://github.com/Sofitch/mmsec/blob/main/descriptions/e3_prnu_fingerprint.pdf). This projects requires the Dresden Image database to run.

To run:

    python .\Ex3.py [BRAND] [DEVICE]

> Suggestion: BRAND, DEVICE = Canon_Ixus70, 0
