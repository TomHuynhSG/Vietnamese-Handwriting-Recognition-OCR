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

Here is more details of CRNN architecture:

![architecture](https://i.imgur.com/7f1IU0Q.png)

As you can see in this diagram, the last layer of CNN produces a feature vector of the shape 4\*8\*4 then we flatten the first and third dimension to be 16 and keep the second dimension to be the same to produce 16\*8. It's effective to cut the original image to be 8 vertical parts (red lines) and each parts contains 16 feature numbers. Since we have 8 parts to be processed as the output of CNN then we also choose 8 for our time step in the LSTM layer. After stacked LSTM layers with softmax (SM) activation function, we have CTC loss to optimize our probability table.

More information regarding the implementation can be found in the jupyter notebook in the github.

## Result

We need to have the right evaluation/metrics for OCR task with edit distance library.

This is inspired from https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/data/evaluation.py

This only helps to calculate three evaluation metris for any OCR task:
- CER (Character Error Rate)
- WER (Word Error Rate)
- SER (Sequence Error Rate)

Here is my result for a test set:

[To be continue]


## License

This project is licensed under the MIT License - see the LICENSE.md file for details
