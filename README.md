# CV-Model-Results-Analysis---1
The main purpose of this code is to assisst in the improvement of your computer vision model. It analyzes your computer vision model and help you to gather new dataset to improve your model.

The code performs following functions:
1) It runs your computer vision model on a yolo dataset
2) Creates result images. It labels whether your result contains True Positive (TP), False Positive (FP), False Negative (FN), True Negative (TN) instances. Examples you can see below, when it detected humans:
![00a3790534a8adc9](https://github.com/SmilerKZ/CV-Model-Results-Analysis---1/assets/35876670/32b3f844-5c50-4e2a-a363-eba30aeb95a2)
![IMG_1930_JPG rf e001fcfa0e16c925f3bc97a5be4184b1](https://github.com/SmilerKZ/CV-Model-Results-Analysis---1/assets/35876670/01c0246b-4ed9-4de4-8c06-119203f823f3)
4) Save the result images in according folders: TP, FP, FN, TN.

Further, you can analyze the result images in FP and FN folders to create the new dataset that will enhance your model

The description of result images:
- The result images contain two images
    - The left image has predicted bounding boxes
    - The right image has bounding boxes from ground truth
- The left image (detected object) can show the following labels:
    - "TP +(dupl.) #:..." means a number of true positive results with possible duplicates (if they exist)
    - "FP #:..." means a number of false positive results
    - "TN" means the result model is labelled as true negative
- The right image (ground truth image) can show the following labels:
    - "TP #" means a number of true positive results
    - "FP #:..." means a number of false positive results
    - "TN" means the result model is labelled as true negative
