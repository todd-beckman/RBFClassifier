RBFClassifier
=============
By Anthony Beckman

A Classifier and Neural Network for general use implemented in Java.

Main.java demonstrates functionality by classifying which quadrants cartesian coordinates are found.
It also demonstrates a flaw in the accuracy of this neural network: for edge cases (such as, in the
quadrants example, values close to an axis), the accuracy becomes inconsistent between adjacent
clusters.

RBFNetwork.java is a neural network with one input layer, one hidden layer, and one output layer. The
weights from the input layer into the hidden layer are generated externally and given to the network.
The neural network learns by updating the weights from the hidden (Gaussian) layer into the
output layer based on the the error. This makes it a supervised learning agent.

RBFClassifier.java is a classifier that manages the RBFNetwork and determines the best-fit class for
given input according to the neural network simply by maximizing the output of the neural network. It
also generates the hidden layer's weights as averages among the learning data. These averages are
calculated with the k-means algorithm which is guaranteed to find local optima for the weights in a
finite (and typically small) number of steps, unlike many methods of approximation which require
convergence.