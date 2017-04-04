# NVIDIA DEEP LEARNING WORKSHOP
## Day2 Notes
### Mateusz Grobelny

### Lab: Deep Learning for Image Segmentation
There are a variety of important applications that need to go beyond detecting individual objects within an image, and that will instead segment the image into spatial regions of interest. An example of image segmentation involves medical imagery analysis, where it is often important to separate the pixels corresponding to different types of tissue, blood or abnormal cells, so that you can isolate a particular organ. Another example includes self-driving cars, where it is used to understand road scenes. In this lab, you will learn how to train and evaluate an image segmentation network.

Image segmentation --> classification at the pixel level


### Lab: Neural Network Deployment
Once a deep neural network (DNN) has been trained using GPU acceleration, it needs to be deployed into production. The step after training is called inference, as it uses a trained DNN to make predictions of new data. This lab will show three approaches for deployment. The first approach is to directly use inference functionality within a deep learning framework, in this case DIGITS and Caffe. The second approach is to integrate inference within a custom application by using a deep learning framework API, again using Caffe, but this time through its Python API. The final approach is to use the NVIDIA TensorRT™, which will automatically create an optimized inference run-time from a trained Caffe model and network description file. You will learn about the role of batch size in inference performance, as well as various optimizations that can be made in the inference process. You will also explore inference for a variety of different DNN architectures trained in other DLI labs.
