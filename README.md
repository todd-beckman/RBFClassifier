RBFClassifier by Anthony Beckman
=============

A Classifier and Neural Network for general use implemented in Java.

Main.java demonstrates functionality by classifying which quadrants cartesian coordinates are found.

RBFNetwork.java is a neural network with one input layer, one hidden layer, and one output layer. The
weights from the input layer into the hidden layer are generated as averages among the test data,
which are calculated with the k-means algorithm which is guaranteed to find local optima for the
weights. The neural network learns by updating the weights from the hidden (Gaussian) layer into the
output layer based on the the error. This makes it a supervised learning agent.

RBFClassifier.java is a classifier that manages the RBFNetwork and determines the best-fit class for
given input according to the neural network simply by maximizing the output of the neural network.
