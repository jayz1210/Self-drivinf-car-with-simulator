# Self-drivinf-car-with-simulator
##INDEX
Acknowledgment
ABSTRACT
1.0  INTRODUCTION
2 LITERATURE REVIEW
2.1  Python
2.2 Tensorflow
2.3 Numpy
2.4  Keras
2.5 Transfer Learning
2.6  Neural Network
2.7  CNN(Convolutional neural network)
2.9 Object Detection
2.10  TensorFlow Object Detection API
3.0 SYSTEM REQUIREMENTS STUDY
3.1 HARDWARE AND SOFTWARE REQUIREMENTS
3.2 CONSTRAINT
4.0 Implementation:
4.1 Vehicle and Person Detection
4.1.1 Data
4.1.2 Vehicle detection
4.1.3Model
4.1.4 Model.summary
4.1.5 Training
3.1.6 Running the Model
4.1.6 Visualizing Intermediate Representations
4.1.6 Output:
4.2 BEHAVIOR CLONING
4.2.1 Approach
4.2.2 Understanding Data
4.2.3 Data preprocessing Before Training :
4.2.3 Network Architecture
4.2.4 Implementation
4.2.5. Results of behaviour Cloning.
4.3 IMPLEMENTATION OF OBJECT DETECTION ON CARLA
4.3.1 Traffic simulation
4.3.2 Install the Model And Label:-
4.3.3  Import the Required Libraries and Model:-
4.3.4  Loading label map:-
4.3.5  Test on Carla Simulator:
6.0 CONCLUSION
7.0 REFRRENCE

Acknowledgment

This project has played a significant role in learning industrial practices and working with the best technologies. However, there have been many who have played direct and indirect roles in this wonderful journey.

First of all, we would like to thank my parents who provided immense love and care in the Covid-19 situation. They took care of our health and made sure that we always gave our best in whatever we do.

Finally, we would like to thank our guide  Prof. Maharishi K. Trivedi for helping in every way possible. Without him, this project wouldn’t have been completed. He provided necessary support whenever required and made sure that progress did not stop. Also, we would like to thank U.V.P.C.E and all other staff members who have helped directly and indirectly in managing all the procedures and making sure that we don’t miss out on any necessary information.
ABSTRACT

In the modern era, vehicles are focused to be automated to give human drivers relaxed driving. In the ﬁeld of automobile various aspects have been considered which makes a vehicle automated. Google, the biggest network has started working on self-driving cars in 2010 and still developing new changes to give a whole new level to automated vehicles

Self Driving Cars are very common these days. With various tech. giants working in this field of research, autonomous mobility i.e. self-driving cars are no longer the talks of fiction. Various driving support systems are now available in the market like lane assist, parking assist, etc. 

The cameras mounted on the front of the car i.e. on the hood of the car, are being used by various self-driving car engineers to implement autonomous driving . Some functions include Road lane recognition, traffic signs detection, object or vehicle detection, and so on. The list is never-ending. In this project, various front-facing cameras (center, left, and right) are used to improve the driving behavior of self-driving cars, and the model is trained in such a way that the car never goes out of the track despite the road curvatures or sharp turns.
1.0  INTRODUCTION

Self-driving cars, also referred to as autonomous cars, are cars that are capable of driving with little to no human input.
Various AI algorithms, mainly Computer Vision, are used to process the information from the sensors. 
For example, we might have one computer vision system that processes the video coming from the cameras around the car, to detect all of the other cars on the road around it. 
All of this information will be used to control the self-driving
Be at the forefront of the autonomous driving industry. With market researchers predicting a $42-billion market and more than 20 million self-driving cars on the road by 2025, the next big job boom is right around the corner.
This project satisfies the requirements of the Vehicle Detection project. The primary objective is to detect other vehicles. and detection in real-time with the help of the Carla simulator 
The 2nd part of this project is behaviour cloning in which we use the UNity Self-driving car simulator and use the concept behaviour cloning in which we were able to drive a car on road autonomously.
We chose to use convolutional neural networks to detect lane lines and cars, rather than the gradient and SVM-based approaches recommended for these projects. we annotated training images with the correct answers by adding extra layers to indicate which parts of the picture were part of lane lines or cars, then trained convolutional neural networks to produce such image masks for other images from the video. The process of curating training data and training convolutional neural networks will be discussed further later in this document.
2 LITERATURE REVIEW
2.1  Python
Python is a popular programming language. It was created by Guido van Rossum, and released in 1991.
It is used for:
●web development (server-side),
●software development,
●mathematics,
●system scripting.
What can Python do?
Python can be used on a server to create web applications.Python can be used alongside software to create workflows.Python can connect to database systems. It can also read and modify files. Python can be used to handle big data and perform complex mathematics.
Python can be used for rapid prototyping or production-ready software development.
Why Python?
Python works on different platforms (Windows, Mac, Linux, Raspberry Pi, etc).
Python has a simple syntax similar to the English language. Python has a syntax that allows developers to write programs with fewer lines than some other programming languages.
Python runs on an interpreter system, meaning that code can be executed as soon as it is written.
2.2 Tensorflow 
TensorFlow was developed by Google and released as open-source in 2015. It grew out of Google’s homegrown machine learning software, which was refactored and optimized for use in production
The name “TensorFlow” describes how you organize and perform operations on data. The basic data structure for both TensorFlow and PyTorch is a tensor. When you use TensorFlow, you perform operations on the data in these tensors by building a stateful dataflow graph, kind of like a flowchart that remembers past events.
The single biggest benefit TensorFlow provides for machine learning development is an abstraction. Instead of dealing with the nitty-gritty details of implementing algorithms, or figuring out proper ways to hitch the output of one function to the input of another, the developer can focus on the overall logic of the application. TensorFlow takes care of the details behind the scenes.

2.3 Numpy
NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.
At the core of the NumPy package, is the ndarray object. This encapsulates n-dimensional arrays of homogeneous data types, with many operations being performed in compiled code for performance.

2.4  Keras
Keras is an API designed for human beings, not machines. Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear & actionable error messages. It also has extensive documentation and developer guides
It wraps the efficient numerical computation libraries Theano and TensorFlow and allows you to define and train neural network models in just a few lines of code.
Benefit :-
●Exascale machine learning.
●Deploy anywhere.
2.5 Transfer Learning 
Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.
It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems and from the huge jumps in skill that they provide on related problems.
In this post, you will discover how you can use transfer learning to speed up training and improve the performance of your deep learning model.
2.6  Neural Network
A neural network is a system that learns how to make predictions by following these steps:
●Taking the input data
●Making a prediction
●Comparing the prediction to the desired output
●Adjusting its internal state to predict correctly the next time
Vectors, layers, and linear regression are some of the building blocks of neural networks. The data is stored as vectors, and with Python you store these vectors in arrays. Each layer transforms the data that comes from the previous layer. You can think of each layer as a feature engineering step, because each layer extracts some representation of the data that came previously.
One cool thing about neural network layers is that the same computations can extract information from any kind of data. This means that it doesn’t matter if you’re using image data or text data. The process to extract meaningful information and train the deep learning model is the same for both scenarios
2.7  CNN(Convolutional neural network)
A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm that can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image, and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.2.8 Fully Connected Layer (FC Layer)
Adding a Fully-Connected layer is a (usually) cheap way of learning non-linear combinations of the high-level features as represented by the output of the convolutional layer. The Fully-Connected layer is learning a possibly non-linear function in that space.
Now that we have converted our input image into a suitable form for our Multi-Level Perceptron, we shall flatten the image into a column vector. The flattened output is fed to a feed-forward neural network and backpropagation applied to every iteration of training. Over a series of epochs, the model is able to distinguish between dominating and certain low-level features in images and classify them using the Softmax Classification technique.
There are various architectures of CNNs available which have been key in building algorithms that power and shall power AI as a whole in the foreseeable future. Some of them have been listed below:
LeNet
AlexNet
VGGNet
GoogLeNet
ResNet
2.9 Object Detection
When humans look at an image, we recognize the object of interest in a matter of seconds. This is not the case with machines. Hence, object detection is a computer vision problem of locating instances of objects in an image.
BasicSteps:-


1) A deep learning model or algorithm is used to generate a large set of bounding boxes spanning the full image (that is, an object localization component)







2.)visual features are extracted for each of the bounding boxes. They are evaluated and it is determined whether and which objects are present in the boxes based on visual features (i.e. an object classification component)






3)In the final post-processing step, overlapping boxes are combined into a single bounding box (that is, non-maximum suppression)
2.10  TensorFlow Object Detection API
The TensorFlow object detection API is the framework for creating a deep learning network that solves object detection problems.
There are already pretrained models in their framework which they refer to as Model Zoo. This includes a collection of pretrained models trained on the COCO dataset, the KITTI dataset, and the Open Images Dataset. These models can be used for inference if we are interested in categories only in this dataset.
MobileNet-SSD
The SSD architecture is a single convolution network that learns to predict bounding box locations and classify these locations in one pass. Hence, SSD can be trained end-to-end. The SSD network consists of base architecture (MobileNet in this case) followed by several convolution layers:


SSD operates on feature maps to detect the location of bounding boxes. Remember – a feature map is of the size Df * Df * M. For each feature map location, k bounding boxes are predicted. Each bounding box carries with it the following information:
●4 corner bounding box offset locations (cx, cy, w, h)
●C class probabilities (c1, c2, …cp)
A standard convolution both filters and combines inputs into a new set of outputs in one step. The depthwise separable convolution splits this into two layers – a separate layer for filtering and a separate layer for combining. This factorization has the effect of drastically reducing computation and model size.


3.0 SYSTEM REQUIREMENTS STUDY
3.1 HARDWARE AND SOFTWARE REQUIREMENTS 
System requirement for the Project:
Common requirements:
●GPU: RTX 2070 or RTX 2080 Ti. GTX 1070, GTX 1080, GTX 1070 Ti, and GTX 1080 Ti
●CPU: 1-2 cores per GPU depending how you preprocess data. > 2GHz; CPU should support the number of GPUs that you want to run. PCIe lanes do not matter.

●RAM:
○– Clock rates do not matter — buy the cheapest RAM.
○– Buy at least as much CPU RAM to match the RAM of your largest GPU.
○– Buy more RAM only when needed.
○– More RAM can be useful if you frequently work with large datasets.

●Hard drive/SSD:
○– Hard drive for data (>= 3TB)
○– Use SSD for comfort and preprocessing small datasets.
●Carla SImulator (use Require High Computing Speed) 
3.2 CONSTRAINT
Deep Learning is very computationally intensive, so you will need a fast CPU with many cores,As abnove specification our PC would not able to train or test training Model.
Because specification of our PC is :
1.I5 10th gen
2.1050ti nvidea graphic card
3.128 SSD  
 We were able to work on project but for training purpose we have to use Google cloud service which is Google Colab , there were many alternatives for Google Colab such as AWS and microsoft asure. Google Colab provide us better specification to train and test Deep learning algorithm. Google Colab specification are:-
CPU	GPU 	TPU
Intel zenon processor with12 gb RAM	Upto Tesla k80 with 12 GB of VRAM	Cloud TPU with 180 tera flop computationm of Xenon


4.0 Implementation: 
 Implementation Environment:-
1.The training of the model and preprocessing has been done on google colab which help to reduce strain on or work by exponentially. They provide us with GPU and TPU for training and testing.
The section 4.1 shows how we initially train the project model on google colab and the basic function behind it.


2.Implementation is done on two simulator:
1.behaviour cloning on Udactiy’s self driving car simulator
2. Objection detection in Carla Simulator 
4.1 Vehicle and Person Detection

In my implementation,weused a Deep Learning approach to image recognition. Specifically, Convolutional Neural Networks (CNNs) to recognize images. However, the task at hand is not just to detect a vehicle’s presence, but rather to point to its location. It turns out CNNs are suitable for these type of problems as well. 

The main idea is that since there is a binary classification problem (vehicle/person), a model could be constructed in such a way that it would have an input size of a small training sample (e.g., 64x64) and a single-feature convolutional layer of 1x1 at the top, which output could be used as a probability value for classification.

Having trained this type of model, the input’s width and height dimensions can be expanded arbitrarily, transforming the output layer’s dimensions from 1x1 to a map with an aspect ratio approximately matching that of the new large input.
4.1.1 Data

Udacity equips students with great resources for training the classifier. Vehicles and non-vehicles and computer-generated HUMAN samples have been used for training.
The total number of vehicle’s images used for training, validation, and testing was about 7500

4.1.2 Vehicle detection
In the first step, the dataset is explored. It is comprised of images taken from the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself. There are two classes, cars and non-cars. The cars have a label of 1.0, whereas the non-cars/human have a label of 0.0

								Samples of data set

There is a total number of 7000samples available, each image is colored and has a resolution of 64x64 pixels. The dataset is split into the training set (7000 samples) and validation set ( 1000+ samples). The distribution shows that the dataset is very balanced, which is important for training the neural network later. Otherwise, it would have a bias towards one of the two classes. 
4.1.3Model

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 64x64 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('CAR') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])
Adding  convolutional layers, and flatten the final result to feed into the densely connected layers.


4.1.4 Model.summary
A neural network is used as a deep-learning approach, to decide which image is a car and which is a HUMAN. The fully-convolutional network looks like this, which is shown by code :
model.summary()

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 62, 62, 16)        448       
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 31, 31, 16)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 29, 29, 32)        4640      
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 12, 12, 64)        18496     
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 6, 6, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2304)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               1180160   
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 513       
=================================================================
Total params: 1,204,257
Trainable params: 1,204,257
Non-trainable params: 0
_________________________________________________________________

4.1.5 Training
Let's train for 15 epochs -- this may take a few minutes to run.
Do note the values per epoch.
The Loss and Accuracy are a great indication of progress of training. It's making a guess as to the classification of the training data, and then measuring it against the known label, calculating the result. Accuracy is the portion of correct guesses.
history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1,validation_data = validation_generator,
      validation_steps=8)

3.1.6 Running the Model
Let's now take a look at actually running a prediction using the model. This code will allow you to choose 1 or more files from file system, it will then upload them, and run them through the model, giving an indication of whether the object is a car or a human.
import numpy as np
from google.colab import files
from keras.preprocessing import image


uploaded = files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path = '/content/' + fn
  img = image.load_img(path, target_size=(64, 64))
  plt.axis('Off') # Don't show axes (or gridlines)
  plt.subplot(2,3,2)
  plt.imshow(img,cmap='gray')
  plt.show()
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
  if classes[0]>0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a car")
4.1.6 Visualizing Intermediate Representations
As we can see we go from the raw pixels of the images to increasingly abstract and compact representations. The representations downstream start highlighting what the network pays attention to, and they show fewer and fewer features being "activated"; most are set to zero. This is called "sparsity." Representation sparsity is a key feature of deep learning.
To get a feel for what kind of features our convnet has learned, one fun thing to do is to visualize how an input gets transformed as it goes through the convnet.

These representations carry increasingly less information about the original pixels of the image, but increasingly refined information about the class of the image. You can think of a convnet (or a deep network in general) as an information distillation pipeline.
4.1.6 Output:
As we got accuracy of 90% but it's on limited data if we try on a more complex sample and the model gives the wrong answer.

4.2 BEHAVIOR CLONING

Simulator:- Unity self-driving car simulator

The goals/steps of this project are the following: Use the simulator to collect data of good driving behavior Build, a convolution neural network (using Keras) that predicts steering angles from images Train and validate the model with training and validation set Test that the model successfully drives around the track without leaving the road

4.2.1 Approach

We used a simulator developed under Unity which comes with an inbuilt training mode and autonomous driving mode. This makes it easy to collect data on good driving behavior to ensure the safety of the self-driving car. There are two scenes in this simulator, the hilly region, and the park. then performed the same approach for both of the scenes and developed a CNN which successfully drives the self-driving car under both scenes


Collected data in the training mode of the simulator by taking four rounds of the track. Also, took one round in the opposite direction of the track to increase data augmentation.
Then build a Convolutional Neural Network in Keras that predicts steering angles from images by taking in input various images, steering angles, throttle, brake and speed values.
Training and validating the model with a training and validation set with a relative test size of 0.2using train_test_split () function.
After training the model, the model is tested in the autonomous mode and it is ensured that the self-driving car doesn’t leave the track and overcomes any sharp turns in the track on its own.


This step can be seen in the following figures.

fig:-1:-training 


fig:-2:- Simulation fig


4.2.2 Understanding Data

There are 3 cameras on the car that shows left, center, and right images for each steering angle.
1:- Center Camera 

2:- Left Camera:-

3:- Right Camera 


After recording and save data, the simulator saves all the frame images in the IMG folder and produces a driving_log.csv file which contains all the information needed for data preparation such as the path to the images folder, steering angle at each frame, throttle, brake, and speed values.
there is a driving_log.csv file which stores the path of each of the image and also various values such as steering, throttle, brake and speed [2]. These values are put into our convolutional neural network which predicts the steering angle such that the car remains on track and drives safely.
4.2.3 Data preprocessing Before Training :

performed various techniques like Random Flipping, Random Translation, Random Brightness, and RGB to YUV image conversion to increase data augmentation. Three laps of the safe driving behavior are taken in the forward direction and one in the opposite direction for the generation of data in the training mode of the Unity Simulator

Then, In order to gauge how well the model is working,wesplit my image and steering angle data into a training and validation set with train_test_split() function with a relative test size of 0.2. We tried many of them, andwechoose the following to form the image processing pipeline.
●Random adjust the image brightness/darkness
def img_random_brightness(image):
    brightness = iaa.Multiply((0.2,1.2))
    image = brightness.augment_image(image)
    return image
●Flip the image and steering angle together
def img_flip(image,steering_angle):
    image = cv2.flip(image,1)
    steering_angle = -steering_angle
    return image,steering_angle

●Random zoom
def zoom(image):
    zoom = iaa.Affine(scale=(1,1.3))
    image=zoom.augment_image(image)
    return image
●RGB to YUV color space 

img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
img= img/255.0




●Lookahead, crop the image center portation out
def pan(image):

    pan = iaa.Affine(translate_percent={"x":(-0.1,0.1),"y":(-0.1,0.1)})
    image = pan.augment_image(image)
    return image

●Horizontal crop the image center portation out, with the shifted steering angle
Here are the some screenshots of Data augmented:

fig:-Zoomed image-

Fig:- brightness altered image

Fig :- augmented image.

Fig :- augmented image.


Fig:- original vs preprocessed image

I used a generator so that a part of the training data images is operated upon a given time. In the end, after training the CNN, the model is saved and a plot is depicted to show the training and validation losses. Steering corrections are also introduced along with appropriate camera images i.e.left, right, and center camera images to increase randomness.
4.2.3 Network Architecture
In this research,weimplemented a convolutional neural network model which is implemented with the help of Keras. The model is just like the NVIDIA – Self Driving Car Model and contains five convolutional layers and four dense layers. The model also contains a Dropout layer, A Flatten layer, and one Cropping 2D layer. The data is normalized in the model using a Keras lambda layer. The total number of parameters in the proposed model is 348, 219. 

First of all, a cropping2D layer is defined which takes the input of 320x160 images and crops it in the required state.
The second layer is the Normalization layer using a lambda layer. Further, there are five convolution layers followed by a Dropout and Flatten layer. The dropout layer has a dropout probability of 50% and the last four layers in the model are Dense layers. Finally, the Keras model is compiled with a loss function called mean_squared_error and an Adam Optimizer with a learning rate value of 1.0e-4 ie 0.001. The number of epochs is set to 30 and batch_size is set to 300

The first layer of the network performs image normalization. The normalizer is hard-coded and is not adjusted in the learning process. Performing normalization in the network allows the normalization scheme to be altered with the network architecture, and to be accelerated via GPU processing.
The convolutional layers are designed to perform feature extraction and are chosen empirically through a series of experiments that vary layer configurations. We then use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers.

We follow the five convolutional layers with three fully connected layers, leading to a final output control value which is the inverse-turning radius. The fully connected layers are designed to function as a controller for steering, but we noted that by training the system end-to-end, it is not possible to make a clean break between which parts of the network function primarily as feature extractors, and which serve as controller.

The Summary of layeres are shown in following:-

_Training Samples: 3511
Valid Samples: 878
Model: "sequential_34"________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_153 (Conv2D)          (None, 31, 98, 24)        1824      
_________________________________________________________________
conv2d_154 (Conv2D)          (None, 14, 47, 36)        21636     
_________________________________________________________________
conv2d_155 (Conv2D)          (None, 5, 22, 48)         43248     
_________________________________________________________________
conv2d_156 (Conv2D)          (None, 3, 20, 64)         27712     
_________________________________________________________________
conv2d_157 (Conv2D)          (None, 1, 6, 64)          36928     
_________________________________________________________________
flatten_27 (Flatten)         (None, 384)               0         
_________________________________________________________________
dense_108 (Dense)            (None, 100)               38500     
_________________________________________________________________
dense_109 (Dense)            (None, 50)                5050      
_________________________________________________________________
dense_110 (Dense)            (None, 10)                510       
_________________________________________________________________
dense_111 (Dense)            (None, 1)                 11        
=================================================================
Total params: 175,419
Trainable params: 175,419
Non-trainable params: 0

The accuracy was model is shown by graph :-

4.2.4 Implementation

First of all, the model is trained to generate the model.h5 file which contains the code to train the model with the help of the following command. Using the Unity Simulator, the car can be tested to drive autonomously in the Autonomous Mode of the Simulator by executing the following command

python drive.py

@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)



@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)



4.2.5. Results of behaviour Cloning.
The car is able to steer successfully across the track in both hilly as well as park tracks with a validation loss equal to 0.0326. The pipeline is successful in driving the car around the track and preventing it from driving off the roads. In this paper,wefollowed the approach of recovery driving i.e. driving across one of the lane markings either left or right while collecting data. Due to this, the model is successful in regions where there are no lane markings on one side of the road. Sharp turns are made easy to navigate and hence, safe driving behavior is established by the self-driving car in the simulator.

4.3 IMPLEMENTATION OF OBJECT DETECTION ON CARLA
4.3.1 Traffic simulation
In order for the traffic to feel real in our virtual environment we need to have different variety of vehicle along with it they should be autopilot and should be spawned at different location the map

Code For spawning Traffic in Carla:-
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0) 

    world = client.get_world()

    print("enter total cars:")
    no_of_cars=int(input())
    blueprint_library = world.get_blueprint_library()    
    bp = blueprint_library.filter('model3')[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.
    actor_list.append(vehicle)
    # get the blueprint for the camera sensor 
    blueprint = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('fov', '110')

    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=2.5, z=1.9))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)

    # Add more cars on auto pilot mode

    #transform.location += carla.Location(x=40, y=-3.2)
    for _ in range(0, no_of_cars):
        spawn_point = random.choice(world.get_map().get_spawn_points())
        bp = random.choice(blueprint_library.filter('vehicle'))

  # This time we are using try_spawn_actor. If the spot is already
  # occupied by another object, the function will return None.
        npc = world.try_spawn_actor(bp, spawn_point)
        if npc is not None:
            actor_list.append(npc)
            npc.set_autopilot()
            print('created %s' % npc.type_id) 
   # use sensor to show the ttached acamera view 
    sensor.listen(lambda data: process_img(data))
    time.sleep(50)
4.3.2 Install the Model And Label:-
First we need to download the model which we need to use in order to detect object for this purpose we have used Ssd_mobilenet_v2_320x320_coco17_tpu-8'  .here ssd mobile net is our model and coco will provide our data set for testing purposes .
Code for model and label download:-
#For Model
MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
#for Labels
LABEL_FILENAME = 'mscoco_label_map.pbtxt'
LABELS_DOWNLOAD_BASE = \
    'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
4.3.3  Import the Required Libraries and Model:-
This is done for the installed model to be build so that it could be used for detection for any given image.since it is a pre trained model we do not need any training but we do need proper data set for it to function properly
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])
4.3.4  Loading label map:-
The model will detect the object and give them a particular set of values we need to convert this value back into there given set of name for us to understand.
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                 use_display_name=True)
4.3.5  Test on Carla Simulator:
while True:
    image_np= cv2.imread('person.png')
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.50,
          line_thickness=1,
          agnostic_mode=False)




6.0 CONCLUSION
From this project we were able to create basic models using neural network and deep learning for identifying different objects and differentiating between humans and cars.Then we used pretrained model for object detection using Transfer learning concept and it gives proper result thereafter for steering and throttle control has be done on unity simulator which gives proper result at low speed (ie 15MPH).
7.0 REFRRENCE 
●Deep Learning, 2016.
●Neural Network Methods in Natural Language Processing, 2017.
●Transfer Learning – Machine Learning’s Next Frontier, 2017.
●Transfer Learning, CS231n Convolutional Neural Networks for Visual Recognition
●Deep learning with Python Book by François Chollet
●Carla introductory tutorial sharing_weixin_43450646's blog-CSDN blog
●Yann LeCun’s Deep Learning Course at CDS – NYU Center for Data Science
●GitHub - udacity/self-driving-car-sim at term2_collection
●MIT Deep Learning and Artificial Intelligence Lectures | Lex Fridman
●http://databookuw.com/page/page-9/
●https://jakevdp.github.io/PythonDataScienceHandbook/
●https://www.coursera.org/professional-certificates/tensorflow-in-practice
●End-to-End Deep Learning for Self-Driving Cars | NVIDIA Developer Blog
●https://www.analyticsvidhya.com/blog/2020/03/google-colab-machine-learning-deep-learning/
●https://www.analyticsvidhya.com/blog/2020/04/build-your-own-object-detection-model-using-tensorflow-api/
●GTI Data	
●https://www.udemy.com/course/applied-deep-learningtm-the-complete-self-driving-car-course
