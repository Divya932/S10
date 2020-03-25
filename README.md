# S10
# Advanced Concepts - Training and learning rates
**Learning Rates**
A hyper parameter that determines how fast or how slow or at what speed will our model learn or converge.
It controls how much the weight of our parameters in the network must be adjusted.
A very high learning rate may reduce the loss faster but can cause our model to overshoot and miss the minima
each time. A slow learning rate may take a long time to converge at all.

Each Pixel is related to the final class - where GradCam comes in
Each weight is related to the final class - where training comes in
W = W - alpha(dL/dW)

The loss is related to each weight in the network

Gradient Ascent (dP/dPx) means we're looking for a positive co-relation and gradient descent (dL/dW) means 
we're looking for negative co-relation

If we increase a particular pixel value the activation for that particular channel increases.
For an increase in a weight the loss should decrease(grad descent)(negative corr)
