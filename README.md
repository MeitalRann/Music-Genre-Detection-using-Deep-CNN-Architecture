# Music Genre Detection using a Deep CNN Architecture
This toolkit provides the code for training and testing a deep convolutional network for the task of music genre detection.  
Implemented in Tensorflow 2.0 using the Keras API.

## Introduction
This Toolkit was influenced by the feature extraction in [1] and follows the architecture design in [2].  


### Aucostic Feature
The feature that is used in this toolkit is the melspectrogram. the feature is extracted as follows:  
1. Shuffle the input and split into train, validation and test (70%/20%/10%).
2. Compute the melspectrogram from each music recording with window size of 2048samples and step size of 512samples.
3. Split the melspectrograms to ~5sec windows with 50% overlap.


#### Deep CNN Architecture
The CNN was designed to be similar to those used in imageNet challenges.    

| #  | Output size | Layer type | Filter size |
| --- | --- | --- | --- |
| 1 | 1x204x204 | Input | |
| 2 | 32x202x202 | Convolutional | 3x3 |
| 3 | 32x101x101 | Max pooling | 2x2 |
| 4 | 32x99x99 | Convolutional | 3x3 |
| 5 | 32x49x49 | Max pooling | 2x2 |
| 6 | 64x47x47 | Convolutional | 3x3 |
| 7 | 64x23x23 | Max pooling | 2x2 |
| 8 | 64x21x21 | Convolutional | 3x3 |
| 9 | 64x10x10 | Max pooling | 2x2 |
| 10 | 128x8x8 | Convolutional | 3x3 |
| 11 | 128x4x4 | Max pooling | 2x2 |
| 12 | 128x2x2 | Convolutional | 3x3 |
| 13 | 128x1x1 | Max pooling | 2x2 |
| 14 | 512 | Fully connected | |
| 15 | 256 | Fully connected | |
| 16 | 75 | Fully connected | |
| 17 | 75 | Soft max | |
| 18 | 1 | Output | |


## Data:
The dataset that is used in this toolkit is the GTZAN dataset. The dataset contains 10 different music genres. No need to download the dataset, the toolkit downloads it automatically.


## Usage
For training and testing, run the code Main.py in the main toolkit directory. The following are the parameters' options:  

```
# Main.py
# extr 1 : extract the melspectrogram
# extr 0 : don't extract melspectrogram
# mode 1 : train  and itest the CNN model
# mode 0 : use saved model for testing
# prj_dir : path/to/dir

```

## Results
The following is the confusion matrix created after training and testing:
![confusion matrix](/figs/confusion_matrix.png)
Format: ![Alt Text](url)

## License
MIT License

Copyright (c) [2020] [Meital Rannon]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## References
[1] https://github.com/Hguimaraes/gtzan.keras

[2] Sergey Shuvaev, Hamza Giaffar, and Alexei A. Koulakov, “Representations of Sound in Deep Learning of Audio
Features from Music,” Arxiv, Available: https://arxiv.org/abs/1712.02898, 2017.
