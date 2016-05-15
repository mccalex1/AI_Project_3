import sys
import os

from sklearn import datasets, svm, metrics, neighbors
from pylab import imread
from PIL import Image
from scipy import misc

import numpy as np   

def learn(ImageSentIn):

    images = datasets.load_files("Training")

    imagesList = []
    targetList = []

    #adds zipped imagename and label to imageList
    for i in range(len(images.filenames)):

        if(images.filenames[i] [-4:] == ".jpg"):
            #http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/cluster/plot_color_quantization.html#example-cluster-plot-color-quantization-py
            currentImage = misc.imread(images.filenames[i])
            currentImage = np.array(currentImage, dtype=np.float64) / 255
            w, h, d = original_shape = tuple(currentImage.shape)
            image_array = np.reshape(currentImage, (w * h, d))
            imagesList.append(image_array)
            targetList.append(images.target[i])


    images_and_labels = list(zip(imagesList, targetList))

    n_samples = len(imagesList)
    data = (np.asarray(imagesList)).reshape((n_samples, -1))

    #change sent in image into numpy array
    ImageToPredict = misc.imread(ImageSentIn)
    ImageToPredict = np.array(ImageToPredict, dtype=np.float64) /255
    w, h, d = original_shape = tuple(ImageToPredict.shape)
    ImageToPredictArray = np.reshape(ImageToPredict, (w*h, d))

    ImageToPredictData = np.asarray([ImageToPredictArray]).reshape(len([ImageToPredictArray]),-1)

    svc = svm.SVC(kernel="linear")
    svc.fit(data, targetList)


#    classifier = svm.SVC(gamma=.001)
#    classifier.fit(data[:n_samples/2], targetList[:int(n_samples/2)])
#    expected = targetList[int(n_samples/2):]
#    predicted = classifier.predict(data[int(n_samples/2):])
    
#    print("Classification report for classifier %s:\n%s\n"
#          % (classifier, metrics.classification_report(expected, predicted)))
#    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    return (svc.predict(ImageToPredictData))

def main():

    argList = sys.argv
    ImageSentIn = argList[1]



    print("Learning....")

    whatAmI = learn(ImageSentIn)


    if whatAmI[0] == 0:
        print("Your image was classified as a dollar sign")

    elif whatAmI[0] == 1:
        print("Your image was classified as a hash")

    elif whatAmI[0] == 2:
        print("Your image was classified as a hat")

    elif whatAmI[0] == 3:
        print("Your image was classified as a heart")

    elif whatAmI[0] == 4:
        print("Your image was classified as a smiley face")

main()
