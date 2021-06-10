# Repose

Repose is a GAN that aims to produce realistic human poses.\
The goal is that it can be trained by observing the improvisations of a specific human dancer, and produce new poses that their body is capable of.

Pose data matches the [COCO Keypoints](https://cocodataset.org/#keypoints-2020), mapping to [17 points on the body](https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/README.md#keypoint-diagram).

Built using PyTorch and Pipenv.
