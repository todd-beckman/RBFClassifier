RBFClassifier
=============
By Todd Beckman

This project contains source code for a Radial-Basis Function Classifier (RBF Classifier) and Artificial Neural Network (RBF Network) for general use implemented in Java. Additionally, the provided K-Means implementation may be useful independent of RBF.

About the Radial-Basis Function
=============

Radial-Basis Functions are used to approximate functions. They do this by distributing Gaussian curves across the input-space. The closer an input is to a certain Gaussian curve, the higher the resulting output for that Gaussian node. Then, in retrospect, error in the result can be blamed on the Gaussian curves that reacted with this input. The Gaussian curves that cause more error have their weights reduced, while those that tend to lead to closer approximations have their weights increased.

Further reading on the [Radial-Basis Function](http://en.wikipedia.org/wiki/Radial_basis_function).

The Gaussian curve was chosen for this algorithm due to several significant mathematical analyses that can be made.

1. Only directly relevant input affects the network. The curve converges to zero so quickly that fuzziness in learning data can be avoided by properly planning the locations of the curves (see K-Means below).

2. The output of the network is a linear combination of the nodes. That means, unlike many other ANN algorithms, the Network's output can be found simply by adding up the output from all of the Gaussian nodes and classifying the output is a matter of testing the range of the output. This addition is an incredibly fast operation compared to the classification process of competing ANN algorithms.

Further reading on the [Gaussian Function](http://en.wikipedia.org/wiki/Gaussian_function).


About the K-Means Algorithm.
=============

When input space is large or continuous, having enough equally-distributed Gaussian curves to approximate the function properly may not be reasonable or possible. Instead, it is more efficient to place these Gaussian curves near clusters in the input space. That way, the most likely input has more reactivity while outlier data does not throw off the results.

This project uses the K-Means algorithm to solve this problem. This algorithm is guaranteed to not only categorize /n/ observations into /k/ clusters, but to do so in a finite and typically small number of steps, unlike many methods of approximation which require infinite convergence.

Further reading on [K-Means Clustering](http://en.wikipedia.org/wiki/K-means_clustering)

More on the Implementation
============

The RBF Network implemented in this project is an artificial neural network with one input layer, one hidden layer, and one output layer. The number of nodes per layer is independant on the project and is generated at the time that training is to begin.

Training the Network is done either at runtime or by providing a comma-separated value (*.CSV) input file. Then, the learned RBF data can optionally be written to a *.CSV file for later use. This way, retraining will not be necessary and the Network may instead initialize on previously learned RBF data.

A Main class demonstrates functionality by classifying which quadrants cartesian coordinates are found.
