# Report for Exercise 1, Machine Learning Course
I carries out the experiement of two type perceptrons with 10 inputs and 1 output. I code it in Python, with Numpy and Pandas. 

### My findings
I initialized all weights with [1, 1, ...., 1], which equals to [0, 0, ..., 0] as the weight should be randomly given. Details of the codes are enclosed in README.md file.
The fixed perceptron, which I coded with discriminant($\rm \alpha(k)^Ty_j $) \< 10, achieved an error rate of 0.37 after a short iteration of 39.
The variable perceptron, with the same discriminant threshold as the fiexed perceptron, achieved 0.48 in error rate, with the same iteration of 39.
Another variable perceptron is based on iteration time(10000), and after training it derived an error rate of 0.06 and 0.09, which is much better than the other two perceptron. 
I make two comparison from observations. 
1. The fixed variable increment rule may perform better than the variable increment rule when the iteration and training set is rather small. 
2. The variable increment rule based perceptron performs better when the iteration is larger. The error rate on the train set is apparently better than that on the test set. This may suggest some degree of overfitting when perceptroning. Thus, when the samples are not exactly linear separable, avoidance of overfitting is rather important in training.