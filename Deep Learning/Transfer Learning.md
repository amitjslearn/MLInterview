# Transfer Learning
The techinique of using the pre-trained model trained on a data for one task and using it for another task, where the tasks should be similar in terms of data that is used.

Intiution behind this approach is that the pre-trained model has already learnt some patterns (like detecting edges and corners and other stuff) from the previous data that it has been trained on,

Transfer Learning works extremely well on Computer vision tasks and it is also taking up in the NLP.

## Advantages
- Less training is required (which means less time and resources)

## Disadvantages
- can't change the pre-trained model's architecture

## Techniques (how to do transfer learning?)
- generally the last layer is replaced with the new layer with the number of outputs equal to the number of classes in the problem, Then train the network.
- Instead of training all the layers at once, we can freeze all the conv layers and train only the last classification head and see how it does.
  + If the accuracy (or loss) is not good, we can then unfreeze all the layers and train the whole model
- When the model is unfreezed, insted of using a uniform learing rate for all the layers, what we can do is use differnt learning rates for differnet layers, i.e. the initial layers will have lower learning rates and the later layers will have little higher learning rates.