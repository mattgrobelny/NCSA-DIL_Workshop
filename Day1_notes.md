# NVIDIA DEEP LEARNING WORKSHOP
## Day1 Notes
### Mateusz Grobelny


## Lecture 1: Deep Learning Demystified
A high level, general introduction to deep learning that provides a background on the technology. We will provide key definitions used in the industry, talk about how deep learning is effective across many domains, why it works as a technology and how NVIDIA services and software and hardware assist organizations to quickly take advantage of deep learning attributes.

*Hyper parameters?*

*How to quantify if network is accurate enough?*

## Lecture 2: Best Practices for Applying Deep Learning
Learn the characteristics of problems that benefit from deep learning and the dataset/project attributes that make deep learning successful. Specific life science/bioinformatics use cases will be used.

### Best Practices

Data prep
- Large data set --> larger data more likely to generalize
- large --> millions
- Reuse --> models should be re-useable
- Need or payoff ! ROI
- Fault tolerance
- Overfitting --> loss of generalization ( Works well on train, but inaccurate on test)
Network is too big or data set is too small

### Training, Validation, Test set (Rule of thumb %)

Training:
- Data set used to train network (60%)

Validation:
- periodically during training check (20%) --> Check for Overfitting

Test set:
- only use to evaluate progress (20%)  

*Normalization of dataset?*

*Precision vs Recall?*

Precision
- proportion of retrieved set that are relevant
- intuition: How much junk did we give to the user?
- P = Pr(relevant|retrieved) = TP/(TP + FP)

Recall:
- fraction of all relevant documents that were found
- intuition: how much of the good stuff did we miss?
- R = Pr(retrieved| relevant )

## Lab 1: Getting Started with Deep Learning
Learn how to leverage deep neural networks (DNN) within the deep learning workflow to solve a real-world image classification problem using NVIDIA DIGITS. You will walk through the process of data preparation, model definition, model training and troubleshooting. You will use validation data to test and try different strategies for improving model performance using GPUs. On completion of this lab, you will be able to use NVIDIA DIGITS to train a DNN on your own image classification application.

Intro to:
- Deep learning
- Workflow of training

### Convolutional Neural network
#### Convolution Kernel - (Filter)
- Do not init to 0
- Size? (3X3 vs 5x5 ... ) odd number
- Size of filter allows for fine res capture of details

### Training
- Forward pass --> leading to a loss function which determines the difference between predicted label and actual
- Backward propagation --> adjustment of weights based on loss function
- Batch size (Power of two --> as long as it fits into memory)

### Batch
- some subset of images used to train network after which the weights are adjusted with one Backward pass

### Epoch
- One run of training on the training dataset
- After each epoch, the model is validated with the validation dataset

### Hyper-parameters
- Values set prior to training
- Effects:
  - Speed
  - accuracy
  - Learning rate, decay rate, batch size?
- Epoch - number of complete passes
- Activation functions - sigmoid, Tanh, ReLU
- Pooling - Down-sampling techniques (no weights)
  - Stride value (step length between down-sampling projections or Convolutions)
  - Example: Max pooling (projects most important features?)


- **Select Dataset** - Choose one of your databases to use for training.
- **Training Epochs** - Select the number of epochs for training.  An epoch is one iteration through the training data.  The number of epochs to use depends on the data and the model, but can be as few as 5 or over 100.
- **Snapshot Interval** - The frequency, in epochs, that the model state is saved to a file.
- **Validation Interval** - The frequency, in epochs, that the accuracy of the model is computed against the validation data.  This is important so you can monitor the progress of your training sessions.
- **Random Seed** - Specifies the value of seed should be used by the random number generator.  By setting this value, then the initial model will be randomize to the same state for different training sessions.
- **Batch Size** - The batch size is the number of images to use at one time.  The larger the batch size, the more parallelism that can be achieved and the faster the training will progress.  The batch size will be constrained by the size of available memory in your GPU.  You typically want to use the largest value possible.
- **Base Learning Rate** - This value specifies at what rate the network will learn. The weights of the model are found using some gradient descent method.  The value describes the size of the step to be taken for each iteration.  Too large of a value and the weights will change too quickly and the model may not converge.  Too small of a value and the solution will take longer to converge.

## Lunch Break

Improving MNIST training:
- Inverse black and white color

# Object Detection
- R-CNN = Region CNN

Working with varying size images: Fully- Convolutional Network 
