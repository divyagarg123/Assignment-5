# Assignment-5

###  Coding Structure Explained:
1. data_loader.py: files having transforms, augmentations, loading of dataset and creating data loaders. Basically everything u need to do with data before training.
2. utils.py : File having some utilities like check for cuda, drawing graphs for train/test accuracy,loss, drawing misclassified images.
3. train.py: It has the logic for training the model.
4. test.py: logic for testing the model.
5. model.py: File having model structure and forming the layers.
6. main.py/main.ipynb: This file is calling functions from other file to create model, train and test data and draw graphs, plots.

### Normalizations Explained:

**Batch Normalization**: In batch normalization, for a batch of images, respective channels are normalized together. E.g: In below snapshot, batch of 3 images is there.
and 4 channels are there per images. So channel1 gets normalized for 3 images then channel2 and so on.

**Layer Normalization**: In layer normalization, images are separately normalized for all channels. e.g. In below snapshot, image1 is separataly normalized across all channels, same with img2 and img3.

**Group Normalization**:It is like layer normalization but in groups. E.g in below snapshot, img1 has 4 channels divided in 2 groups and separately these are normalized.



![image](https://user-images.githubusercontent.com/109232157/215721343-d4523328-439d-406c-ab3e-28e6e083eb09.png)

In below graphs, accuracy wise all normalizations are performing equally well, in case of loss, Group normalization starting with bit higher loss than others.

![image](https://user-images.githubusercontent.com/109232157/215554694-2ee95c42-11c9-49d1-8b0e-e13e4b8d857d.png)
![image](https://user-images.githubusercontent.com/109232157/215554829-d668800b-8df7-4655-8787-5efabe13fa80.png)




![image](https://user-images.githubusercontent.com/109232157/215554909-45091cb2-6922-4a49-9691-e58062c9da3d.png)
![image](https://user-images.githubusercontent.com/109232157/215554980-54b791e2-16dc-4cf7-b6f8-2d992c23e259.png)
![image](https://user-images.githubusercontent.com/109232157/215555037-f4ce8e7d-3d54-4152-b674-e4b93e31f241.png)
