### Dehazing_Using_GAN
This repository contains Dehazing vision pipeline for IROS paper "[Vision-Based Autonomous Navigation for Unmanned Surface Vessel in Extreme Marine Conditions]([https://arxiv.org/pdf/2308.04283.pdf](https://ieeexplore.ieee.org/abstract/document/10341867))" 

Below are video results showcasing the effectiveness of our Dehazing vision pipeline. In these examples, the target vessel is not easily detected when using the direct image from the onboard camera. However, after passing the image through our GAN model for dehazing, the visibility is significantly improved. The enhanced images are then processed by the YOLO model, leading to successful detection.

![](https://github.com/AhsanBaidar/Dehazing_Using_GAN/blob/master/output.gif)

### Before you start.

Clone Respository:
```
git clone https://github.com/AhsanBaidar/Dehazing_Using_GAN
```
Navigate to folder Dehazing_Using_GAN:
```
cd Dehazing_Using_GAN
```

### Training a Model
The training dataset for both scenarios are provided in the folder named "USV_Dataset/(Sand_Storm/Fog)". [Download from here](https://drive.google.com/file/d/1eSbEGbbhgpUkWH5dhGoz5n_bIC4a59eE/view?usp=sharing). Start training by running the script train.py:

### Inference mode
Pre-trained model weights for both scenarios are provided in [weights](https://drive.google.com/file/d/1AlQRfiPewFXTpkbNggKIdE7WKM8yHZcN/view?usp=sharing) Save the folder and save in the current directory.
To process image input, run the following Image_Output.py:
To process video input, run the following Video_Output.py:


  

