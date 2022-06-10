<h1 align="center">
  LSTM | GRU for Regression
</h1>



<p align="center">
  Welcome to the forth programming assignment of the Deep Learning course. 
  <br/><br/>
<!--   <img src="images/sample_image_colorization.jpg"> -->
</p>



The goal is to predict stock price. Prediction stock price is a time-series problem. Time series prediction problems are a difficult type of predictive modeling problem.


Unlike regression predictive modeling, time series also adds the complexity of a sequence dependence among the input variables.


A powerful type of neural network designed to handle sequence dependence is called recurrent neural network. The LSTM and GRU networks are a type of recurrent neural network used in deep learning because they can successfully train very large architectures.



**After this assignment, you will:**

 - Discover how to develop LSTM/GRU networks to address a demonstration time-series prediction problem.
 - Have a comparison between LSTM and GRU.



### Dataset
 
**Google Stock Dataset** This dataset consists of the closing stock price of a share of Google stock during the trading days between December 20, 2013, and December 17, 2021.[link](https://finance.yahoo.com/quote/GOOG/history/?guccounter=1)

> GOOG.close The closing stock price of a share of Google stock.
> 
> GOOG.close is used for this problem.



### **Problem Statement:**
    
    -- Create the Training Data in the Many-To-One relationship for LSTM/GRU (creating inputs and output for model training)
    -- Train different LSTM/GRU models to predict stock price. (predicting `close` feature)
    -- Compare the resluts for LSTM and GRU. (plotting the results in a graph besides comparing the metrics)
    
    > Consider future predictions for the next 30 days. (30 time steps)

