# ChRIS-COVID-Net
A ChRIS plugin for COVID-Net, a neural network for identifying COVID-19 using chest X-ray images.

COVID-Net's github repo can be found [here](https://github.com/lindawangg/COVID-Net).

## Intro
This repo can be built into a Docker container that includes all required software to run the COVID-Net training process.

The container will include the following libraries (major ones):

* Tensorflow 1.15
* OpenCV 4.2.0.34
* Python 3.6.9
* Numpy
* Scikit-Learn
* Matplotlib
* PyDicom
* Pandas

## Build the container
To build the Docker image, simply run the following:

```
docker image build -t "covid-net" .
```

The container can be ran as a daemon so you can enter the container and start the processes.

```
docker run -dit --name covid-net-test [REPLACE_WITH_IMAGE_ID]
```

## COVIDx Dataset
Note that, once the container is up and running, a few steps are required to obtain the input data: 

A pretrained models need to be downloaded into the 
```
COVID-Net/models
```
directory. Refer [here](https://github.com/lindawangg/COVID-Net/blob/master/docs/models.md) to download the pretrained models.

COVID-Net needs a training dataset that is constructed by the following open source chest radiography datasets:

* https://github.com/ieee8023/covid-chestxray-dataset
* https://github.com/agchung/Figure1-COVID-chestxray-dataset
* https://github.com/agchung/Actualmed-COVID-chestxray-dataset
* https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
* https://www.kaggle.com/c/rsna-pneumonia-detection-challenge (which came from: https://nihcc.app.box.com/v/ChestXray-NIHCC)

More details to obtain the dataset can be found [here](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md).
