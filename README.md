# vietnamese-handwriting-recognition-ocr

![intro](https://i.imgur.com/6MDY4Lm.jpg)

Handwriting OCR for Vietnamese Address using state-of-the-art CRNN model implemented with Tensorflow. This was a challenge proposed by the Cinnamon AI Marathon.

## Challenge Description

![challenge](https://i.imgur.com/vwlsx52.png)

Given an image of a Vietnamese handwritten line, we need to use an OCR model to transcribe the image into text like above.

## Requirements
* tensorflow 2.0+
* scikit-learn
* opencv-python
* editdistance

## Dataset 

The dataset, which have 1838 images and its labels in json file, is provided by Cinnamon AI.

Here are 10 samples of the dataset:
![dataset](https://i.imgur.com/fTXetk0.jpg)

Here is the structure of the json file containing the labels:
![label](https://i.imgur.com/UKzANSI.png)

Due to the large size of the dataset (>350 MB), the zip file can be downloaded at the google drive link: https://drive.google.com/file/d/1-hAGX91o45NA4nv1XUYw5pMw4jMmhsh5/view?usp=sharing

## Architecture

Ideally, we want to detect text from a text image:

![architecture](https://i.imgur.com/dH3mK1H.png)

However, character segmentation is not practical because:
![architecture](https://i.imgur.com/AyUqcPp.png)
* Too time comsuming
* Too expensive 
* Impossible in most cases

For example, the above character segmentation is fine but the below one is challenging. In fact, the traditional method will face a problem where two or more characters are too close to each other like this:
![architecture](https://i.imgur.com/jBbSJ19.png)

This project will use state of the art CRNN model which is a combination of CNN, RNN and CTC loss for image-based sequence recognition tasks, specially OCR (Optical Character Recognition) task which is perfect for CAPTCHAs.

![architecture](https://i.imgur.com/npfKiCa.jpg)

This model is much more superior than traditional way which does not involve any bounding box detection for each character (character segmentation). 

In this model, the image will be dissected by a fixed number of timesteps in the RNN layers so as long as each character is seperated by two or three parts to be processed and decoded later then the spacing between each character is irrelevant like so:

![architecture](https://i.imgur.com/TOpXFan.png)

Here is more details of my CRNN architecture:


```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 118, 2167, 1 0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 118, 2167, 64 640         input_1[0][0]                    
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 39, 722, 64)  0           conv2d[0][0]                     
__________________________________________________________________________________________________
activation (Activation)         (None, 39, 722, 64)  0           max_pooling2d[0][0]              
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 39, 722, 128) 73856       activation[0][0]                 
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 13, 240, 128) 0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 13, 240, 128) 0           max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 13, 240, 256) 295168      activation_1[0][0]               
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 13, 240, 256) 1024        conv2d_2[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 13, 240, 256) 0           batch_normalization[0][0]        
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 13, 240, 256) 590080      activation_2[0][0]               
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 13, 240, 256) 1024        conv2d_3[0][0]                   
__________________________________________________________________________________________________
add (Add)                       (None, 13, 240, 256) 0           batch_normalization_1[0][0]      
                                                                 activation_2[0][0]               
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 13, 240, 256) 0           add[0][0]                        
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 13, 240, 512) 1180160     activation_3[0][0]               
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 13, 240, 512) 2048        conv2d_4[0][0]                   
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 13, 240, 512) 0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 13, 240, 512) 2359808     activation_4[0][0]               
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 13, 240, 512) 2048        conv2d_5[0][0]                   
__________________________________________________________________________________________________
add_1 (Add)                     (None, 13, 240, 512) 0           batch_normalization_3[0][0]      
                                                                 activation_4[0][0]               
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 13, 240, 512) 0           add_1[0][0]                      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 13, 240, 1024 4719616     activation_5[0][0]               
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 13, 240, 1024 4096        conv2d_6[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 4, 240, 1024) 0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 4, 240, 1024) 0           max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 1, 240, 1024) 0           activation_6[0][0]               
__________________________________________________________________________________________________
lambda (Lambda)                 (None, 240, 1024)    0           max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
bidirectional (Bidirectional)   (None, 240, 1024)    6295552     lambda[0][0]                     
__________________________________________________________________________________________________
bidirectional_1 (Bidirectional) (None, 240, 1024)    6295552     bidirectional[0][0]              
__________________________________________________________________________________________________
dense (Dense)                   (None, 240, 141)     144525      bidirectional_1[0][0]            
__________________________________________________________________________________________________
the_labels (InputLayer)         [(None, 240)]        0                                            
__________________________________________________________________________________________________
input_length (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
label_length (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
ctc (Lambda)                    (None, 1)            0           dense[0][0]                      
                                                                 the_labels[0][0]                 
                                                                 input_length[0][0]               
                                                                 label_length[0][0]               
==================================================================================================
Total params: 21,965,197
Trainable params: 21,960,077
Non-trainable params: 5,120
__________________________________________________________________________________________________
```

If this CRNN model is confusing to understand for you, then you should check out my other CAPTCHA solver project on my github which has a simplier CRNN model to understand.

More information regarding the implementation can be found in the jupyter notebook in the github.

The number of callbacks I used are very helpful which are ModelCheckpoint, EarlyStopping and ReduceLROnPlateau which allows my model to keep on improving after 2 hours of training. 

## Result

It took around 2 hours to train my model up to epoch 80 before early stopping callback is triggered with the lowest loss is 16.53810.

![graphloss](https://i.imgur.com/ynyvnF4.png)

As we can see, the loss for validation continue to increase for the first 6 epoches and sharply drop onwards and mostly stable all the way till epoch 80.


We need to have the right evaluation/metrics for OCR task with edit distance library.

This is inspired from https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/data/evaluation.py

This only helps to calculate three evaluation metris for any OCR task:
- CER (Character Error Rate)
- WER (Word Error Rate)
- SER (Sequence Error Rate)

Here is my metrics for a test set:

Character Error Rate: 0.04761427177354741
Word Error Rate:      0.15659406463634423
Sequence Error Rate:  0.8097826086956522

We got a pretty good results with CER at 4% and WER at 15%! 

I'm certain if I continue to apply for other techniques, this will help to reduce these numbers down. For example, I can try add to an attention layer between my CNN and RNN layers.

Here are the prediction for each image in my test set:

![result](https://i.imgur.com/bCaNyl5.png)


## License

This project is licensed under the MIT License - see the LICENSE.md file for details
