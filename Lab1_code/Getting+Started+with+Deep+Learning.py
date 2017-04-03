
# coding: utf-8

# ## Getting Started With Deep Learning

# By Craig Tierney

# ### Overview

# Deep learning allows machines to achieve near-human levels of visual recognition, disrupting many applications by replacing hand-coded software with predictive models learned directly from data.  This lab introduces the machine learning workflow and provides hands-on experience with using deep neural networks (DNN) to solve a challenging real-world image classification problem.  We will work through all phases of the deep learning workflow, including data preprocessing, training, evaluation, and methods to improve training accuracy (including data augmentation and network optimization). You will also see the benefits of GPU acceleration in the model training process.  On completion of this lab, you will have the knowledge to train a DNN on your own image classification dataset.
# 
# 

# ## Verify your Run Environment
# 
# Before we begin, let's verify [WebSockets](http://en.wikipedia.org/wiki/WebSocket) are working on your system.  To do this, execute the cell block below by giving it focus (clicking on it with your mouse), and hitting Ctrl-Enter, or pressing the play button in the toolbar above.  If all goes well, you should see some output returned below the grey cell.  
# 
# You will know the lab is processing when you see a solid circle in the top-right of the window that looks like this: ![](images/jupyter_executing.png)
# Otherwise, when it is idle, you will see the following: ![](images/jupyter_idle.png)
# For troubleshooting, please see [Self-paced Lab Troubleshooting FAQ](https://developer.nvidia.com/self-paced-labs-faq#Troubleshooting) to debug the issue.

# In[ ]:

print "The answer should be three: " + str(1+2)


# Let's execute the cell below to display information about the GPUs running on the server.

# In[ ]:

get_ipython().system(u'nvidia-smi')


# ## Introduction to Deep Learning
# 
# Deep learning is the use of many hidden layers in an artificial neural network to train a model to learn and understand the data without an inherent understanding of the data.  Deep learning is used by many different disciplines to allow computers to learn from vast amounts of data.  Recent advances in computer vision, object detection and natural language processing can be attributed to the adoption of Deep Learning techniques.
# 
# One key to the success of deep learning is the convolutional neural network (CNN).  In a traditional neural network, artificial neurons in a layer are fully connected to the previous and next layer.  In a [convolutional neural network](http://en.wikipedia.org/wiki/Convolutional_neural_network), overlapping regions of the network are associated instead of the entire layer.  Biologically inspired, a convolutional network acts as filters to process pieces of data.  The result is that these networks can act as feature extractors at both coarse and fine scales, eliminating the need to design custom feature extractors previously required in traditional computer vision.  CNNs can be trained to recognize structure in large sets of data, including images, voice, and text.
# 
# The workflow of deep learning is a multi-phase, iterative process.  First, data must be gathered and pre-processed.  While the Internet and big data have provided access to massive quantities of data, the data often need to be verified, labeled, and pre-processed for consistency.  Second, the data are used to train a network.  Pre-existing networks can be used or new ones can be developed.  Both the network and the training process have many variables that can be modified and tuned, which affect the training rate and accuracy.  Third, models need to be tested to verify they are working as expected.  Often at this point, one iterates over the steps to improve the results.  This includes reprocessing data, modifying networks, modifying parameters of the networks or solvers, and retesting until the desired results are obtained.
# 
# In this tutorial today, you will work through all phases of the deep learning workflow to solve a classic machine learning problem: text recognition.  We will be using a multi-layer convolutional neural network to train our model to recognize handwritten digits from the [MNIST] (http://yann.lecun.com/exdb/mnist/) database. The MNIST database contains thousands of images of handwritten numbers from 0 to 9.  This dataset has been used by both those new and experienced to machine learning as standard for image recognition. 
# 
# To simplify the process of the deep learning workflow, we will be using NVIDIA's [DIGITS](https://developer.nvidia.com/digits).  DIGITS is a Deep Learning GPU Training System that helps users develop and test convolutional neural networks. DIGITS supports multiple frameworks, data formats, and training goals including image classification and object detection.
# 
# In addition, DIGITS also includes a workload manager.  The workload manager allows the user to launch multiple jobs of different types, and it coordinates access to the local resources.  If multiple GPUs are present, jobs can run simultaneously.  If more jobs are created than the resources available, the jobs will be queued until resources become free.  The DIGITS dashboard allows the user to monitor all of the active jobs and their current state.
# 
# We will be using the [Caffe](http://caffe.berkelyvision.org) framework.  Caffe was created by the Berkeley Vision and Learning Center (BVLC).  Caffe is a flexible and extensible framework that provides researchers the means to train networks without writing all of the code necessary to do so.  Model training can be parallelized across multiple GPUs to accelerate learning.
# 
# This tutorial will cover the typical tasks of the deep learning workflow.  First, we will create a database from the MNIST data.  Second, we will train a model to classify the MNIST images.  Third, we will test the trained model against other test data and analyze the results.  After this, we will augment the data and modify the standard network to try and improve our image classification accuracy.

# ## Starting DIGITS
# 
# When you start DIGITS, you will be taken to the home screen.  
# 
# ![](images/digits_home.png)
# 
# You can create new datasets or new models.  This home page will show all of your currently processing and completed models and datasets.  The default window pane shows the models.  If you wish to look at the datasets, select the **Datasets** tab on the left.  On the right there are two tabs: **New Datasets** and **New Models**.  Select the tab you wish to create a new dataset or net model.  When you do, you will be presented a menu of choices.
# 
# ![](images/digits_zoom_new_dataset.png)
# 
# You can choose either **Classification**, **Object Detection** or **Other**.  In this tutorial, we will be using the **Classification** option.  
# 
# When creating a new model, you will get a similar drop down menu.
# 
# ![](images/digits_zoom_new_model.png)
# 
# To start DIGITS, <a href="/digits/" target="_blank">click here</a>.

# ## Task - Create a Database
# First, we want to create a database from the MNIST data. To create a database, select **Classification** from the **New Dataset** menu.    At this point you may need to enter a username.  If requested, just enter any name in lower-case.
# 
# In the **New Dataset** window, you want to set the following fields to the values specified:
# 
# - Image Type : Grayscale
# - Image Size : 28 x 28
# - Training Images: /home/ubuntu/data/train_small
# - Select **Separate test images folder** checkbox
# - Test Images : /home/ubuntu/data/test_small
# - Dataset Name : MNIST Small
# 
# Your screen should look like the image below.
# 
# ![](images/digits_new_classification_dataset.png)
# 
# When you have filled in the fields, press the **Create** button.
# 
# The next window will show you the progress of the job and the estimated time to completion.  This shouldn't take more than a minute.
# 
# When it is done, you can explore the database.  Find the **Explore** Button at the bottom of the **Create DB (train)** panel.  
# 
# ![](images/digits_explore_db.png)
# 
# Here you can scan through all of the images in the database. When the database is created, the image order is randomized. Models will train faster and more accurately when they do not process the images in order (process all of the images of zeros, then all of the images of ones, etc.). When you explore your database, your database will be in a different order than the one shown here. 
# 
# On this page you can see several examples of the handwritten digits.  Some are neat, some are sloppy, but all are different.  We want our system to be able to properly classify each variant.

# ## Task 2 - Create the Model
# Now that we have a database of images, lets train our network.  At the top right of every page, the name of our training system, **DIGITS**, is visible.  If you click the name it will take you back to your home page.  From here, we can select **Classification** from the **New Model** menu.  
# 
# In the **New Image Classification Model** page, there are many options available to configure and tune your training session.  Some of the more typically used ones are:  
# 
# - **Select Dataset** - Choose one of your databases to use for training.
# - **Training Epochs** - Select the number of epochs for training.  An epoch is one iteration through the training data.  The number of epochs to use depends on the data and the model, but can be as few as 5 or over 100.
# - **Snapshot Interval** - The frequency, in epochs, that the model state is saved to a file.
# - **Validation Interval** - The frequency, in epochs, that the accuracy of the model is computed against the validation data.  This is important so you can monitor the progress of your training sessions.
# - **Random Seed** - Specifies the value of seed should be used by the random number generator.  By setting this value, then the initial model will be randomize to the same state for different training sessions.
# - **Batch Size** - The batch size is the number of images to use at one time.  The larger the batch size, the more parallelism that can be achieved and the faster the training will progress.  The batch size will be constrained by the size of available memory in your GPU.  You typically want to use the largest value possible.
# - **Base Learning Rate** - This value specifies at what rate the network will learn. The weights of the model are found using some gradient descent method.  The value describes the size of the step to be taken for each iteration.  Too large of a value and the weights will change too quickly and the model may not converge.  Too small of a value and the solution will take longer to converge. 
# 
# DIGITS currently has built-in support for three networks.  [LeNet](http://yann.lecun.com/exdb/lenet/) is a convolutional network originally developed to recognize hand written digits. In 2012 [AlexNet](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) won the [ImageNet](http://image-net.org/) competition using a Deep Neural network instead of traditional computer vision techniques. This revolutionized the field of computer vision, and within a couple of years all of the top entries in the ImageNet competition were based on Deep Neural Networks.  [GoogleNet](https://arxiv.org/abs/1409.4842) in 2014 set a new standard of image classification in the ImageNet competition.
# 
# DIGITS also supports two frameworks.  Caffe is the one with which we will be working today.  Torch (http://torch.ch/) is another framework that is good at image classification, as well as speech recognition and language processing.
# 
# To train our model, we want to set the following options:
# 
# - Select the **MNIST small** dataset
# - Set the number of **Training Epochs** to 10
# - Set the framework to **Caffe**
# - Set the model to **LeNet**
# - Set the name of the model to **MNIST small**
# 
# ![](images/digits_create_new_model.png)
# 
# When you have set all of these options, press the Create button.  You are now training your model!  For this configuration, the model training should complete in less than a minute.  While the training progresses, the statistics of the model are updated in the window.  The chart in the middle provides key information as to how well your training is doing.
# 
# ![](images/digits_model_loss_accuracy.png)
# 
# Three quantities are reported: training loss, validation loss, and accuracy.  The values of training and validation loss should decrease from epoch to epoch, although they may jump around some.  The accuracy is the measure of the ability of the model to correctly classify the validation data.  If you hover your mouse over any of the data points, you will see its exact value.  In this case, the accuracy at the last epoch is about 96%.  Your results might be slightly different than what is shown here, since the initial networks are generated randomly.
# 
# An accuracy of 96% sounds pretty good for a model that finished in seconds!  But does it work in practice?

# ### Test The Model
# 
# Lets test the ability of the model to identify other images.  If you go to the bottom of the window, you can test a single image or a list of images.  On the left, type in the path **/home/ubuntu/data/test_small/2/img_4415.png** in the Image Path text box.  Select the **Show visualizations and statistics** checkbox, then select the **Classify One** button.  After a few seconds, a new window is displayed with the image and information about its attempt to classify the image.
# 
# ![](images/digits_classify_one_1.png)
# ![](images/digits_classify_one_2.png)
# 
# The figure above show the top portion of what you will see in your window.  The data in this window provides information about how well the model is doing, but it also provides information regarding what each of the layers is doing. You can see that the model reported that there is a 95.8% chance that your image contained a 2.  It got it right.  
# 
# In the LeNet network, the first few layers scale the data.  The scaled and data layers show how the original image is scaled and what the resulting image looks like.  The conv1 layer used a 5x5 kernel and 20 outputs.  You can see that in the conv1 Weights section there are 20 5x5 images.  Each of these kernels has learned something about the low level features of the images.  Since the kernel is 5x5, as it slides across the 28x28 image, the output from the convolution is 24x24.  This is reported in the conv1 Activiation section where the data output is reported as **[20 24 24]**.  As data are processed through the CNN, the images continue to get smaller which allows the network to detect different features.  
# 
# The pool1 layer has a kernel size of 2, a stride of 2, and is set to MAX.  This means it returns the maximum value within a 2x2 patch, skipping 2 pixels as it moves across the image.  The image size input to this layer was 24x24, and the resulting image size is 12x12.  You can continue to work through all of the layers and see how the data are learned by the output from the Activation sections and how the images are transformed as they progress through the network.
# 
# At the bottom of this page (not shown here), the statistics page reports the **Total Learned Parameters** as 431,080.  This might sound like a large number of weights, but it is quite small compared to other networks.  Networks like AlexNet and GoogleNet have tens of millions or hundreds of millions of weights.  There are even networks that have more than a billion weights.  As our images are only 28x28, we would not have enough information in our data to train networks of those sizes.
# 
# Let's try testing the model with a set of images.  They are shown below.
# 
# <img src="test_images/image-1-1.jpg" width="64px" /> <div style="text-align:center;">image-1-1.jpg</div>
# <img src="test_images/image-2-1.jpg" width="64px" /> <div style="text-align:center;">image-2-1.jpg</div>
# <img src="test_images/image-3-1.jpg" width="64px" /> <div style="text-align:center;">image-3-1.jpg</div>
# <img src="test_images/image-4-1.jpg" width="64px" /> <div style="text-align:center;">image-4-1.jpg</div>
# <img src="test_images/image-7-1.jpg" width="64px" /> <div style="text-align:center;">image-7-1.jpg</div>
# <img src="test_images/image-8-1.jpg" width="64px" /> <div style="text-align:center;">image-8-1.jpg</div>
# <img src="test_images/image-8-2.jpg" width="64px" /> <div style="text-align:center;">image-8-2.jpg</div>
# 
# While these are clearly numbers to us humans, they look different from the images that we inspected earlier.  Some have color, some have backgrounds, and very few look like hand written digits.  How does our model do against these images?
# 
# We can classify multiple files if we put them in the list. In the link below, execute the code block and a link to the file an_image.list will appear.  Right click on an_image.list and save that to a file on your local computer. Remember the directory in which it is saved.

# In[ ]:

from IPython.display import FileLink, FileLinks
FileLinks('test_images_list')


# On the right side of the DIGITS, there is an option to specify an Image List file.  Press the button **Choose File** and select the `an_image.list` file you just downloaded. Then press the **Classify Many** button.  After several seconds, you will see the results from Caffe trying to classify these images with the generated model.  In the image name, the first number is the digit in the image (ex. image-3-1.jpg is a 3). Your results should be similar to this:
# 
# ![](images/classify-many-images-small.png)
# 
# What is shown here is the probability that the model predicts the class of the image.  The results are sorted from highest probability to lowest.  All but the last two of these predictions are incorrect.
# 
# While the accuracy of the model was 96%, it could not correctly classify any of the images that we tested.  What can we do to improve the classification of these images?

# ## Train with more data
# 
# In our first attempt at training, we only used 10% of the full MNIST dataset.  Let's try training with the complete dataset and see how it improves our training.  We can use the clone option in DIGITS to simplify the creation of a new job with similar properties as an older job.  Let's return to the home page by clicking on **DIGITS** in the upper left hand corner.  Then select **Dataset** from the left side of the page to see all of the datasets that you have created.  You will see your **MNIST small** dataset.  When you select that dataset, you will be returned to the results window of that job.  In the right hand corner you will see a button: **Clone Job**. 
# 
# ![](images/digits_clone_dataset.png)
# 
# Press the **Clone Job** button.  
# 
# From here you will see the create dataset template populated with all the options you used when you created the MNIST small dataset.  To create a database with the full MNIST data, change the following settings:
# 
# - Training Images - /home/ubuntu/data/train_full
# - Test Image - /home/ubuntu/data/test_full
# - Dataset Name - MNIST full
# 
# Then press the **Create** button.  This dataset is ten times larger than the other dataset, so it will take a few minutes to process.
# 
# After you have created your new database, follow the same procedure to clone your training model.  In the template, change the following values:
# 
# - Select the MNIST full dataset
# - Change the name to MNIST full
# 
# Then create the model.
# 
# With six times more data, the model will take longer to run. It still should complete in less than a minute. What do you notice that is different about the results? Both the training and validation loss function values are much smaller. In addition, the accuracy of the model is around 99%, possibly greater. That is saying the model is correctly identifying most every image in its validation set. This is a large improvement. However, how well does this new model do on the test images we used previously?
# 
# Using the same procedure from above to classify our set of test images, here are the new results:
# 
# ![](images/digits_classify_many_full.png)
# 
# The model was still only able to classify two of the seven images.  While some of the classifications came in a close second, our model's predictive capabilities were not much greater.  So are we asking too much of this model to try and classify non-handwritten, often colored, digits with our model? 
# 

# ## Improving Model Results - Data Augmentation
# 
# You can see with our seven test images that the backgrounds are not uniform.  In addition, most of the backgrounds are light in color whereas our training data all have black backgrounds.   We saw that increasing the amount of data did help for classifying the handwritten characters, so what if we include more data that tries to address the contrast differences?
# 
# Let's try augmenting our data by inverting the original images.  Let's turn the white pixels to black and vise-versa.  Then we will train our network using the original and inverted images and see if classification is improved.
# 
# To do this, follow the steps above to clone and create a new dataset and model.  The directories for the augmented data are:
# 
# - Training Images - /home/ubuntu/data/train_invert
# - Test Image - /home/ubuntu/data/test_invert
# 
# Remember to change the name of your dataset and model.  When the new dataset is ready, explore the database.  Now you should see images with black backgrounds and white numbers and also white backgrounds and black numbers.
# 
# Now train a new model.  Clone your previous model results, and change the dataset to the one you just created with the inverted images.  Change the name of the model and create a new model.  When the training is complete, the accuracy hasn't really increased over the non-augmented image set.  In fact, the accuracy may have gone down slightly.  We were already at 99% so it is unlikely we were going to improve our accuracy.  Did using an augmented dataset help us to better classify our images?  Here is the result:
# 
# ![](images/digits_classify_many_invert.png)
# 
# By augmenting our dataset with the inverted images, we were able to identify four of the seven images.  While our result is not perfect, our small change to the images to increase our dataset size made a significant difference.

# ## Improving Model Results -- Modify the Network
# 
# Augmenting the dataset improved our results, but we are not identifying all of our test images. Let's try modifying the LeNet network directly. You can create custom networks to modify the existing ones, use different networks from external sources, or create your own. To modify a network, select the Customize link on the right side of the Network dialog box.
# 
# ![](images/network-dialog.png)
# 
# This will open an editor with the LeNet model configuration.  Scroll through the window and look at the code.  The network is defined as a series of layers.  Each layer has a name that is a descriptor of its function.  Each layer has a top and a bottom, or possibly multiples of each, indicating how the layers are connected.  The *type* variable defined what type the layer is.  Possibilities include **Convolution**, **Pool**, and **ReLU**.  All the options available in the Caffe model language are found in the [Caffe Tutorial](http://caffe.berkeleyvision.org/tutorial/).
# 
# At the top of the editor, there is a **Visualize** button.  Pressing this button will visualize all of the layers of the model and how they are connected.  In this window, you can see that the data are initially scaled, there are two sets of Convolution and Pooling layers, and two Inner Products with a Rectilinear Unit (ReLU) connected to the first Inner Product.  At the bottom of the network, there are output functions that return the accuracy and loss computed through the network.
# 
# We are going to make two changes to the network.  First, we are going to connect a ReLU to the first pool.  Second, we are going to change the values of num_output to 75 for the first Convolution (conv1) and 100 for the second Convolution (conv2).  The ReLU layer definition should go below the pool1 definition and look like:
# 
# <code>
# layer {
#   name: "reluP1"
#   type: "ReLU"
#   bottom: "pool1"
#   top: "pool1"
# }
# </code>
# 
# The Convolution layers should be changed to look like:
# 
# <code>
# layer {
#   name: "conv1"
#   type: "Convolution"
#   bottom: "scaled"
#   top: "conv1"
# ...
#   convolution_param {
#     num_output: **75**
# ...
# </code>
# <code>
# layer {
#   name: "conv2"
#   type: "Convolution"
#   bottom: "pool1"
#   top: "conv2"
# ...
#   convolution_param {
#     num_output: **100**
# ...
# </code>
# Note, the ellipis (...) just indicates we removed some of the lines from the layer for brevity.  The only change you have to make is to the value of num_output.
# 
# After making these changes, visualize your new model.  You should see the ReLU unit appear similar to:
# 
# ![](images/add-relu.png)
# 
# Now change the name of the model and press the **Create** button.  When it is complete test the data again.  The results should be similar to:
# 
# ![](images/classify-invert-relu.png)
# 
# Were you able to correctly identify them all? If not, why do you think the results were different?

# ## Next Steps
# 
# In our example here, we were able to identify all of our test images successfully.  However, that is generally not the case.   How would be go about improving our model further?  Typically hyper-parameter searches are done to try different values of model parameters such as learning-rate or different solvers to find settings that improve model accuracy.  We could change the model to add layers or change some of the parameters within the model associated with the performance of the convolution and pooling layers.  In addition, we could try the other models, such as Alexnet.

# ## Summary
# 
# In this tutorial you were provided an introduction to the Deep Learning and all of the steps necessary to classify images including data processing, training, testing, and improving your network through data augmentation and network modifications.  In the training phase, you learned about the parameters that can determine the performance of training a network. By training a subset of the MNIST data as well as the full set, we learned that more data is better.  In testing our model, we found that although the test images were quite different than the training data, we could still correctly classify them.
# 
# Now that you have a basic understanding of Deep Learning and how to train using both Caffe and Digits, what you do next is limited only by your own imagination.  To test what you have learned, there are several good datasets with which to practice.  First, there is the [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.  The CIFAR is a set of 60000  small (32x32) images with numerous classes such as dogs, cats, planes, and trucks.  The CIFAR10 dataset has 10 different classes of data.  The CIFAR100 dataset is an extension with 100 different classes of data.  In addition, the ImageNet database is another dataset with which to test your skills.   
