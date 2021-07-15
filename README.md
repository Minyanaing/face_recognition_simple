# Face Recognition
This tutorial is testing the face recognition. I used [Face recognition](https://pypi.org/project/face-recognition/) library to get measurements of the face and [OpenCV](https://pypi.org/project/opencv-python/) to take the image streams from the webCam.

### Code summary
After detecting the face from the camera, it matched with the face that I already saved in a folder. If the face matches and the name is not recorded in CSV, it put the name and the time in the CSV.

The required packages can be installed:
- pip install opencv-python
- pip install face-recognition
- pip install cmake
- pip install dlib