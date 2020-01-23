# Sequential Optial Character Recognition

A runnable version of the code is available at [this Google Colab notebook](https://colab.research.google.com/drive/13TGQ6AV5kd8f0ForZYHP0-TEggXlo0o4)

## Problem Definition
The problem is inspired by [How to train a Keras model to recognize text with variable length](https://www.dlology.com/blog/how-to-train-a-keras-model-to-recognize-variable-length-text/) and redefined as:

* Given a video footage showing an image from left to right, recover the text sentence printed on the image. To simplify the problem, we assume the sentence is randomly drawn from the [GRID corpus](http://staffwww.dcs.shef.ac.uk/people/J.Barker/assets/cooke-2006-jasa-ecbf8f7ef7cb429e9621317bfc64a67002a4c465be3c1a3f6144eeed058ee634.pdf).
* The sentence can appear anywhere on the image and hence any frame in the video.
* The video frame width is smaller than any character width so that no frame captures a full character.
* The video frame height is the same as the image height.
* Sample from the video footage a sequence of image frames as input to the neural net. To make the problem difficult, we use a low sample rate such that no two neighbouring frames in the sampled sequence share common pixels and some pixels from the original image are never captured in the samples. 

![Problem definition by picture](https://raw.githubusercontent.com/liyinnbw/ML/master/SequentialOCR/problem.png)

## Neural Net Architectures
A study on 3 possible neural net structures were carried out, all of them share the same CNN feature-extraction front-end, and the same CTC decoder, but different sequence processing layers. The sequence processing layers studied include MLP (Multi-layer Perceptron), Bi-GRU (Bidirectional GRU), and TCN (Temporal Convolutional Network). 

![3 Neural Net Architectures](https://raw.githubusercontent.com/liyinnbw/ML/master/SequentialOCR/models.png)

![An extended view of the common layers](https://raw.githubusercontent.com/liyinnbw/ML/master/SequentialOCR/neural_net_common.png)

## Results Comparison
### MLP
* Total params: 51,148 (51,028 trainable + 120 non-trainable params)
* Epoch = 150 SER= 0.39176470588235296 WER= 0.06745098039215686 CER= 0.01529411764705878
![MLP result](https://raw.githubusercontent.com/liyinnbw/ML/master/SequentialOCR/result_mlp.png)

### Bi-GRU
* Total params: 78,668 (78,548 trainable + 120 non-trainable params)
* Epoch = 103 SER= 0.12117647058823529 WER= 0.0203921568627451 CER= 0.003495798319327723
![MLP result](https://raw.githubusercontent.com/liyinnbw/ML/master/SequentialOCR/result_bigru.png)

### TCN
* Total params: 78,668 (78,668 trainable + 120 non-trainable params)
* Epoch = 100 SER= 0.10714285714285714 WER= 0.019246031746031747 CER= 0.0033673469387755037
![MLP result](https://raw.githubusercontent.com/liyinnbw/ML/master/SequentialOCR/result_tcn.png)

## Conclusion
* Bi-GRU and TCN produced similar results in terms of sentence/word/character error rates. Both were better than the baseline MLP model which is expected. 
* However, TCN used slightly more parameters to train but took shorter time (wall time) to converge than Bi-GRU. There is an argument in [this paper](https://arxiv.org/pdf/1609.03499.pdf) that this is generally true for TCN as compared to RNN.
