# Rank Top Solution for 2018 DAC-SDC

This repo mainly contain the source code of the proposed solution of team ICT-CAS(GPU-platform) for 2018 DAC System Design Contest. The contest information can refer to [2018 DAC-SDC](http://www.cse.cuhk.edu.hk/~byu/2018-DAC-SDC/index.html). We mainly tailored the YOLOv2 to satisfy the contest speed constraint and explore the design space for performance-speed tradeoff. 

The performance of our model is as follow:

| Tested Accuracy(mean IOU) | Speed(FPS)|
|:-----:|:-----:|:-----:|
|0.6975|24.55|


# Prerequisites

- Jetson TX2
- Jetpack3.1

# Code Organzation
- `img` directory include a mini batch of test images for evaluation.

- `src` directory include the source code and model weight.

# Demo script
```
$./compile.sh
$python demo.py
```