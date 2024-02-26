# Face-Alter
This library contains an implementation of a Vanilla-VAE trained on the CelebA dataset. The Variational AutoEncoder is used to generate new faces and to alter existing faces. 

You can use the demo and the pre-trained model from the following notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArianAmani/Face-Alter/blob/main/VAE%20Latent%20Exploration.ipynb)

We are using linear interpolation in the latent space to alter the faces. The following gif shows the interpolation between two faces:
![interpolation](gifs/gifs/interpolation.gif 'interpolation')


We can also add or remove an attribute from a face, by finding a direction in the latent space that corresponds to the attribute. To find this direction embedding, we simply subtract the mean of the embeddings of the faces without the attribute from the mean of the embeddings of the faces with the attribute. The following gif shows the addition and removal of the smile attribute from a face (slowly):

![image](gifs/gifs/Smile3_Smiling.gif 'image')

This repository is implemented using Tensorflow.

* The code in this repository was written and implemented as a project for the course `Image Processing and Introduction to Deep Learning` at the Amirkabir University of Technology.
* The code in the `src` directory was written a long time ago and might have some issues, please open an issue if you find any.
