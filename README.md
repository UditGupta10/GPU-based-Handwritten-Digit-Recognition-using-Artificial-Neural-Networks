# GPU-based-Handwritten-Digit-Recognition-using-Artificial-Neural-Networks  
Project done as part of course DS295-Parallel programming  
Achieved speedup compared to prevalent frameworks by performing thread level optimisations using CUDA API (PyCuda).  
The current methods for training a Neural Network(NN) apply the same approach for all layers. However,
this can prove to be suboptimal as we move deeper into the
network as the size of layers, in general, decrease. Through
this project we have shown how we can decrease the training
time of a NN by exploiting the fact that last layers have fewer
parameters and hence the ”optimal” strategy that worked for
initial layers can still be improved. We show the results for
a 3 layer NN(Input, Hidden and output layer) in which our
contribution is the optimization of last 2 layers
