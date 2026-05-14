# GeoGuessr-Computer-Vision

Inspired by the popular game GeoGuessr, this project utilizes 
convolutional neural networks to identify geographical locations based on 
visual cues from street imagery. The model is trained to classify Google
Street View images into one of 6 Southern California cities 
(Los Angeles, San Diego, SLO, Bakersfield, Riverside, or Anaheim).

Southern California poses a challenge for geolocation due to similar environments, 
landscapes, and architecture that make it difficult to distinguish between cities.  
Upon playing the SoCalGuessr Game, I achieved a 14/50 (28%).

The core pipeline is based on the ResNet-18 architecture, consisting of an initial 
convolution layer, 16 hidden convolution layers, and a final fully connected  
classification head. The number of filters scales, starting with 64 and 
increasing to 128, 256, and 512 units. Utilizes 11.7 million parameters. 
The hidden convolutional layers utilize ReLU activations, 
with the head using a Softmax activation. Additionally, the ReLU activations 
are followed by batch normalization. 

The optimizer was an Adam optimizer paired with a CrossEntropyLoss 
Loss Function. 10 epochs took between 2 and 5 minutes. The learning rate was (1 x 10-3), and the batch size was 32. 
Color Jitter and Random Horizontal Flips were implemented to further improve generalization. 
