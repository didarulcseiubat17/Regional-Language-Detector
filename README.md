# Regional-Language-Detector

A region and location detecting application using a tri-combination of Google's TensorFlow Object Detection API,  Convolutional Recurrent Neural Network (CRNN) and a 2-layered Neural Network for language classification.

The Application uses three modules in sequential manner:
1. Google's TensorFlow Object Detection API for Text localisation
2. CRNN for text Recognition
3. 2-layered Neural Network for language classification
*Location is determined by predicted text using Geotext*

### Starting Application
- *Edit the Image path in main.py*
```sh
#Image-Location-Detector/
python3 main.py
```
### Requirements
-  [Python 3.6]
-  [TensorFlow 1.2]
-  [PyTorch]
-  [OpenCv] - *Either build from source or*
```sh
sudo apt-get install python-opencv
```
* [Geotext]
```sh
pip3 install geotext
```
### Downloads
    wget https://www.dropbox.com/s/l0vo83hmvv2aipn/crnn.pth

### Notes
 - Currently `ssd_mobilenet_v1_coco_11_06_2017` model is being used - *boundry box prediction for text detection will be poor*
 - Try to build your own custom dataset using `labelImg`

### Todos 
 - Train custom dataset for Text detection 
 - [Notes on generating custom dataset](http://androidkt.com/train-object-detection/)

### Refrences
 - https://github.com/AKSHAYUBHAT/DeepVideoAnalytics/tree/master/notebooks/OCR
 - https://github.com/datitran/object_detector_app

![alt text](https://raw.githubusercontent.com/manish7294/Image-Location-Detector/master/Screenshot%20from%202017-10-28%2010-37-42.png)
