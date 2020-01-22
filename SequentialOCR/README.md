# Sequential Optial Character Recognition

A runnable version of the code is available at [this Google Colab notebook](https://colab.research.google.com/drive/13TGQ6AV5kd8f0ForZYHP0-TEggXlo0o4)

## Problem Definition
The problem is inspired by [How to train a Keras model to recognize text with variable length](https://www.dlology.com/blog/how-to-train-a-keras-model-to-recognize-variable-length-text/) and redefined as:

* Given a video footage showing an image from left to right, recover the text sentence printed on the image. To simplify the problem, we assume the sentence is randomly drawn from the [GRID corpus](http://staffwww.dcs.shef.ac.uk/people/J.Barker/assets/cooke-2006-jasa-ecbf8f7ef7cb429e9621317bfc64a67002a4c465be3c1a3f6144eeed058ee634.pdf).
* The sentence can appear anywhere on the image and hence any frame in the video.
* The video frame width is smaller than any character width so that no frame captures a full character.
* The video frame height is the same as the image height.
* Sample from the video footage a sequence of image frames as input to the neural net. To make the problem difficult, we use a low sample rate such that no two neighbouring frames in the sampled sequence share common pixels and some pixels from the original image are never captured in the samples. 

![Problem definition by picture](https://raw.githubusercontent.com/liyinnbw/ML/master/SequentialOCR/problem_def.png)
