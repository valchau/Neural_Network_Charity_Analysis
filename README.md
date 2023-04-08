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
Websites: https://www.geeksforgeeks.org/ways-to-import-csv-files-in-google-colab/#,  http://neuralnetworksanddeeplearning.com/, https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6#:~:text=The%20ReLU%20is%20the%20most,neural%20networks%20or%20deep%20learning

## Results
### Data Preprocessing
1. What variables are neither target nor features? 
The columns EIN and NAME are identification information and have been removed from the input data, using this code:
<br>
application_df= application_df.drop(['EIN', 'NAME'],1)
<br>
2. What variables are the features?
The following variables APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT are the features (independent variables) to be used in the model. <hr>
<br>
<img src="https://github.com/valchau/Neural_Network_Charity_Analysis/blob/main/features.PNG" alt="features" >
<br>
3. What variables are the target?
The column IS_SUCCESSFUL contains binary data refering to whether the charity donation was used effectively. This variable is the target (dependent variable) for our deep learning neural network. The following Python code shows how this analysis used this target variable:
<hr>
Split our preprocessed data into our features(X) and target(y) arrays
y = application_df.IS_SUCCESSFUL
X = application_df.drop(columns="IS_SUCCESSFUL")

Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

Now, encoding of the categorical variables, splitting the target variable into training and testing datasets and standardization have been applied. We are ready to create the neural network model and use it.

### Compiling, Training, and Evaluating the Model
The first deep-learning neural network model is comprised of two hidden layers each with its own activation function, with 80 and 20 neurons respectively in the two layers. The output layer is made up of a binary classification so it uses a linear model. To speed up the training process, we are using the activation functions sigmoid and relu for the hidden layers.

<br>
<img src="https://github.com/valchau/Neural_Network_Charity_Analysis/blob/main/firstNN.PNG" alt="first attempt" >
<br>

The compiler used is adam with loss binary_crossentropy
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

<br>
<img src="https://github.com/valchau/Neural_Network_Charity_Analysis/blob/main/firstNNresults.PNG" alt="first attempt" >
<br>

The accuracy is only 47% which is a very low value.


To increase the performance of the model, I applied bucketing to the feature ASK_AMT and organized the different values by intervals.
I also increased the number of neurons on the hidden layers, trying out several models and finally I used a model with four hidden layers, with varying numbers of neurons in each layer and while mostly using the sigmoid activation function, I tried the tanh activation function on the 4th hidden layer. I did not know enough to have a plan - I just tried out several numbers of layers, neurons and mixtures of activation functions before I got close to 75% accuracy. Then I stopped.

<br>
<img src="https://github.com/valchau/Neural_Network_Charity_Analysis/blob/main/finalNN.PNG" alt="final attempt" >
<br>

The results of this model are much better than my first model: 

<br>
<img src="https://github.com/valchau/Neural_Network_Charity_Analysis/blob/main/finallNNresults.PNG" alt="final attempt" >
<br>
 The accuracy is now 72% which is much better than the first attempt. 


## Summary
The deep learning neural network model did not reach the target of 75% accuracy, but I was able to get 72% accuracy. Considering that this target level is pretty average we could say that the model is not the best. Since this task is a binary classification situation, it could use a supervised machine learning model such as the Random Forest Classifier to combine a multitude of decision trees to generate a classified output and evaluate its performance against the deep learning model.
