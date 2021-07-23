# MNIST_Classifier
A MNIST classifier using the Lenet-5 convolutional neural network 

![image](https://user-images.githubusercontent.com/67227358/126797895-930b8e02-b6cb-4794-b84d-4ba62c5e3a62.png)
Though there seems to be a slight gap in the loss convergence between the valid/test set and the actual train set (where the test set converges to a much lower loss value), we can consider the model to be an adequate fit considering the convergent form of the graph. Specifically, we know that the model is not overfitting because the test/valid set loss values does not increase--if not fluctuate--after some number of epochs. While it is still true that the graph depicts some bias in the model (considering the gap between sets), we can consider this to be an ideal result given the limits of the Lenet-5 architecture, since even for a model whose test/valid set loss curve tends upwards after touching train set curve, we would need to trade some degree of bias in order to prevent overfitting of the model.

![image](https://user-images.githubusercontent.com/67227358/126798421-16c47b47-1e14-46a1-ade6-8ac75db63529.png)
Now we will check for which numbers the model failed to predict. This is a list that contains pairs of predicted_labels and true_labels of those the model failed on. It is evident that the model, though it predicts true for more than 99% of the labels, it fails mostly on numbers 8 and 9. We will try to figure out why using the tsne plot below.

![image](https://user-images.githubusercontent.com/67227358/126798000-67e21ba8-c7a0-4b74-b279-bb5d78ae914e.png)
Though this is not an accurate representation of the 84 dimensional linear layer, the tsne plot does communicate some significant insights, especially that of the failed predictions. Using TSNE, we tried to project the high dimensional separation boundaries of  the model onto a 2 dimensional space. If we take a look at the points for the numbers 8 and 9 (which produced most errors), the boundaries seem to be a bit ambiguous  compared to others whose points seem to form a tight cluster of each label. This visualization gives us an intuition of which labels the model is weakest at, and thus a starting point to consider an upgrade to our current model.


![image](https://user-images.githubusercontent.com/67227358/126798155-7ee6a79a-18e4-4098-81ba-1e2e2cd638a2.png)
![image](https://user-images.githubusercontent.com/67227358/126798254-cf4f5dd4-156d-4ce1-be51-9a3a5040d9da.png)
Below is the visualization of the image of 8 and 9 when passed through the first convolutional layer. My intuition was that, the reason why 8 is yielding most errors is because it has the most round edges that are similar to those in other numbers so it might be slightly less distinguishable.  The visual representation of the convolutional layer, however, does not seem to give so much intuition so as to validate this hypothesis.

     
   """ERROR LOG
1. The bias error
   Problem: The printed outputs for a single batch of inputs looked nearly identical in the initial training cycle.
   First, I thought the model is somehow averaging optimal outputs but found out it wasn't as the lower digits for the int-64 dtype
   outputs were slightly different. Also tried toggling with batch numbers, learning rates, number of nuerons, and activation
   functions, none of which made any contribution to solving the problem.
   Solution: The problem was with the absence of bias terms in the fully-connected layers (the linear functions). As it turns out,
   the bias terms in linear functions, when coupled with an activation function (sigmoid in my case), determine at which point the
   neurons fire--that is, the bias point the sigmoid function is centered around. Thus in my case, through several layer of neurons,
   all inputs were made to 'fire' the sigmoid function at a nearly identical point. Should remember to set bias=True.
2. Misunderstanding of the flatten function
   Problem: Though the loss rate significantly reduced after solving the bias error, it was still ocillating between 0.1 and 0.4,
   never quite converging to a lower loss. The problem was with my misunderstanding of the flatten function (+ the architecture of lenet)
   . I first thought the function auto matically flattens all channels into a single dimension and feeds it into the fully-connected layers.
   This led to me creating a 400x120 linear layer to match the number of inputs created by the flatten function, instead of
   using a conv layer that converts 16 to 120 channels. Leaving the thirds conv layer out significantly reduced the number of channels
   going through the fully-connected layers, fatally reducing the complexity of the model.
   What flatten func actually did was simply convert each multi-dimensional channels into a single-dimensional, linear format.
   Solution: Using the Flatten func properly and adding a conv layer that creates 120 channels.
 3. Number of Batches
   I used smaller batches to avoid overloading the GPU, but it seems a smaller batch somehow yields much faster convergence.
   For the same cycle of epoch, the model with a smaller batch yields much better results for some reason.
   UPDATE: I acknowledged that the model seemed to be converging faster because of the smaller validation set, which essentially
   'made the test easier.'
 4. GPU Memory allocation
   Problem: GPU memory keeps crashing.
   Solution:
     -must check if I am actually using the gpu
     -free the gpu cache after every epoch, if the memory runs out in several iterations of epochs
     -if memory runs out in a single epoch, must decrease the size of the batch or comment out some unnnecessary calculations
     in order to reduce memory load.
     -always run the validate set in a batch -- failing to do so explodes the cache
 5. TSNE
   Problem: The scatter plot did not seem to represent the 99% accuracy of the model
   Solution: The loss function that is used for the tsne model is not convex. In other words, there could be multiple local minimums to which the model can converge.
   Taking into account the fact that we are reducing 84dimensions into a two-dimensional space, it seem highly possible that the loss function is stuck in a local minimum.
   Dramatically lowering the learning rate and increasing the number of iterations is the only solution that I found that improves the representation of the model, but the
   issue must be looked into more carefully."""
  

