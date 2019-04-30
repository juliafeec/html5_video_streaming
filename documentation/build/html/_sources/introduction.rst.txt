Introduction
============

Our product is a multi-camera identification software that is able to detect and identify people’s faces and objects across multiple video streams and return the feed corresponding to the best view of the required target. The use cases of this product include home automation and surveillance. Users could make use of the software, coupled with suitable hardware, to easily keep track of children inside their homes. Commercial clients could use the software when confronted with multiple video sources, to quickly and reliably obtain the best view of a target without manual intervention. This could be applied to the tracking of a target in a busy scene.

The data for this product will be user-provided images that will be used to extract features from pre-trained convolutional neural networks for image recognition, such as FaceNet. During inference time, frames will be captured from the video sources linked to the system in order to identify the most appropriate stream.

After the user provides training data and label, he/she can add the live stream data from the IP camera or surveillance camera as the input for our software, and our software is able to draw bounding box on certain objects, provide the classification of those objects, log the data into a time series .csv etc. file, report certain anomalies and store it on the computer.

