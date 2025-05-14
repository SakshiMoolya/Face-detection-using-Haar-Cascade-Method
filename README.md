# Face-detection-using-Haar-Cascade-Method
This project presents a simple yet effective face detection system using Haar cascade classifiers, implemented in Python via the OpenCV library. The system supports detection of single and multiple human faces in static images by analyzing grayscale representations.

To perform detection, the system uses pre-trained Haar cascade XML files:

haarcascade_frontalface_default.xml for detecting front-facing faces

haarcascade_profileface.xml for detecting side (profile) faces

For multi-face detection, it integrates both frontal and profile classifiers, and additionally processes horizontally flipped images to catch reversed-profile faces (e.g., faces looking to the left instead of right).

An intelligent overlap-checking mechanism is built-in to eliminate duplicate detections of the same face when detected from both frontal and profile cascades.

The user interacts with the program via a command-line interface, selecting the input image and specifying whether the image contains a single person or multiple people. Detected faces are marked using colored bounding boxes and shown in a visual output window using OpenCV’s display tools.

The simplicity of the design, combined with reliable detection in ideal lighting conditions, makes this application an accessible and practical starting point for anyone interested in exploring face detection technologies.
