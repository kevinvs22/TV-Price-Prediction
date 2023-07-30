# TV-Price-Prediction

# Introduction:
TVs are a big part of our daily lives. We use them for a variety of purposes like watching movies, the news, playing video games, listening to music, etc. Their prices vary according to multiple factors. It can be difficult to select the right one for you since there are a lot of options to choose from. 

One of the uses of machine learning is predicting the prices of products, so in this project I implemented the K-Nearest Neighbors (K-NN) algorithm to predict the price of TVs based on some of their features. I chose this algorithm because this technique is a straightforward and efficient machine learning model for classification and regression problems. 

My learning model took TV features as input, and it gave the price class as output. My project is needed because it helps a lot of people that are trying to buy a TV, but they do not know which one to choose. They are able to insert their desired features and the model tells them the price range of that TV.

# Data:
Data collection was performed on Amazon.com. Specifically, I chose 150 TVs randomly in the range of $500 and $1000. The attributes of my data are Display Size (15.6 inches – 75 inches), Refresh Rate (60 hertz or 120 hertz), Resolution (720p – 4K), and Type (LCD, LED, OLED, QLED, NanoCell, ULED, or QNED). I stored the data on a csv file using Excel.

# Implementation:
The libraries I used for my project were pandas and scikit-learn. The input of my system was the TV features and the output of my system was the price class. As part of data preprocessing, I removed the brand name of the TVs, removed the instances with missing values, and I did the Min-Max Scaler. 

A data preparation technique for numerical features is called data scaling. Data scaling is necessary for the success of many machine learning techniques, including the KNN algorithm, linear and logistic regression, and gradient descent approaches. The Min-Max Scaler reduces the data inside the specified range, often between 0 and 1. By scaling features to a predetermined range, it changes data. It scales the values to a particular value range while preserving the original distribution's shape. 

I encoded the Type column since it was the only one that had nonnumeric values. I did not perform any feature extraction. The classifier I used was KNN (K-Nearest Neighbor). For the performance measure of my model, I used multiple evaluation metrics. These were accuracy, precision, recall, F1 score, and k-fold cross validation. I used the time function to determine the time in seconds that my model took to run one iteration. Then I ran it a couple of times and calculated the average time.

# Results:
The precision of my model was 0.56, the recall 0.47, the F1 score 0.51, the cross validation mean score 0.34, the cross validation standard deviation 0.125, and the running time was 0.079 seconds on average.
