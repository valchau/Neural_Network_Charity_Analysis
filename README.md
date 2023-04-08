# Neural_Network_Charity_Analysis
module 20 
## Analysis Overview
The purpose of this project is to learn about and use deep learning neural networks using TensorFlow in Python to analyze and classify the success of charitable donations. The first two parts of this task used specific steps and defined model. The third part of the task required thought to improve the given neural network model to optimze it. I wasn't able to get 75% accuracy but I did try several attempts to improve the model's accuracy by adding hidden layers, changing their units and activation functions. Some of my attempts actually made the accuracy WORSE so after that, I just did my best to get as close to 75% accuracy as I could in the optimization task.

I learned the background and history of computational neurons as well as current implementations of neural networks as they apply to deep learning. I also learned the  major costs and benefits of different neural networks and learned how to compare these costs to traditional machine learning classification and regression models. Additionally, I learned about implementing neural networks and deep neural networks across a number of different datasets, including image, natural language, and numerical datasets. Finally, I learned how to store and retrieve trained neural network models to optimize the results. This was all new to me.

### Steps
* get colab to find input data file
* preprocessing the data for the neural network model,
* compile, train and evaluate the model,
* optimize the model.
* display results visually and tables
* get colab to write output files to my PC 

### Resources
Data Source: charity_data.csv
Software: Python 3.9, google's colab: https://colab.research.google.com/drive
Websites: https://www.geeksforgeeks.org/ways-to-import-csv-files-in-google-colab/#,  http://neuralnetworksanddeeplearning.com/

## Results
### Data Preprocessing
The columns EIN and NAME are identification information and have been removed from the input data.

The column IS_SUCCESSFUL contains binary data refering to weither or not the charity donation was used effectively. This variable is then considered as the target for our deep learning neural network.

The following columns APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT are the features for our model.

Encoding of the categorical variables, spliting into training and testing datasets and standardization have been applied to the features.

### Compiling, Training, and Evaluating the Model
The first two deep-learning neural network models are comprised of two hidden layers with 80 and 30 neurons respectively.
The input data has 43 features and 25,724 samples.
The output layer is made of a unique neuron as it is a binary classification.
To speed up the training process, we are using the activation function ReLU for the hidden layers. As our output is a binary classification, Sigmoid is used on the output layer.

For the compilation, the optimizer is adam and the loss function is binary_crossentropy.
The model accuracy is under 75%. This is not a satisfying performance to help predict the outcome of the charity donations.

To increase the performance of the model, we applied bucketing to the feature ASK_AMT and organized the different values by intervals.
We increased the number of neurons on one of the hidden layers, then we used a model with three hidden layers.

We also tried a different activation function (tanh) but none of these steps helped improve the model's performance.

## Summary
The deep learning neural network model did not reach the target of 75% accuracy, but I was able to get 72% accuracy. Considering that this target level is pretty average we could say that the model is not the best. Since this task is a binary classification situation, it could use a supervised machine learning model such as the Random Forest Classifier to combine a multitude of decision trees to generate a classified output and evaluate its performance against the deep learning model.
