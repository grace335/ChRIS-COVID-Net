# ChRIS-COVID-Net
A ChRIS plugin for COVID-Net, a neural network for identifying COVID-19 using chest X-ray images.

For more detailed information about COVID-Net, refer the github repo [here](https://github.com/lindawangg/COVID-Net).

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

The container can run as a daemon so you can enter the container and start the processes.

```
docker run -dit --name covid-net-test [REPLACE_WITH_IMAGE_ID]
```

To attach your terminal to the running container, run:

```
docker attach [REPLACE_WITH_CONTAINER_ID]
```

## COVIDx Dataset
Note that, once the container is up and running, a few steps are required to obtain the input data: 

COVID-Net needs a training dataset that is constructed by the following open source chest radiography datasets:

* https://github.com/ieee8023/covid-chestxray-dataset
* https://github.com/agchung/Figure1-COVID-chestxray-dataset
* https://github.com/agchung/Actualmed-COVID-chestxray-dataset
* https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
* https://www.kaggle.com/c/rsna-pneumonia-detection-challenge (which came from: https://nihcc.app.box.com/v/ChestXray-NIHCC)

The above five datasets need to be present in the
```
COVID-Net/data/
```
directory.

More details to obtain the dataset can be found [here](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md).

Once the datasets and models are ready, we can combine the five datasets into the single COVIDx dataset, by running the create_COVIDx_v3.py script inside the container:

```
python create_COVIDx_v3.py 
```

A sample output can be like this:

```
root@a668c372cb86:/# cd COVID-Net/
root@a668c372cb86:/COVID-Net# python create_COVIDx_v3.py 
Data distribution from covid datasets:
{'normal': 0, 'pneumonia': 33, 'COVID-19': 390}
Key:  pneumonia
Test patients:  ['8', '31']
Key:  COVID-19
Test patients:  ['19', '20', '36', '42', '86', '94', '97', '117', '132', '138', '144', '150', '163', '169', '174', '175', '179', '190', '191COVID-00024', 'COVID-00025', 'COVID-00026', 'COVID-00027', 'COVID-00029', 'COVID-00030', 'COVID-00032', 'COVID-00033', 'COVID-00035', 'COVID-00036', 'COVID-00037', 'COVID-00038', 'ANON24', 'ANON45', 'ANON126', 'ANON106', 'ANON67', 'ANON153', 'ANON135', 'ANON44', 'ANON29', 'ANON201', 'ANON191', 'ANON234', 'ANON110', 'ANON112', 'ANON73', 'ANON220', 'ANON189', 'ANON30', 'ANON53', 'ANON46', 'ANON218', 'ANON240', 'ANON100', 'ANON237', 'ANON158', 'ANON174', 'ANON19', 'ANON195', 'COVID-19(119)', 'COVID-19(87)', 'COVID-19(70)', 'COVID-19(94)', 'COVID-19(215)', 'COVID-19(77)', 'COVID-19(213)', 'COVID-19(81)', 'COVID-19(216)', 'COVID-19(72)', 'COVID-19(106)', 'COVID-19(131)', 'COVID-19(107)', 'COVID-19(116)', 'COVID-19(95)', 'COVID-19(214)', 'COVID-19(129)']
test count:  {'normal': 0, 'pneumonia': 5, 'COVID-19': 100}
train count:  {'normal': 0, 'pneumonia': 28, 'COVID-19': 286}
test count:  {'normal': 885, 'pneumonia': 594, 'COVID-19': 100}
train count:  {'normal': 7966, 'pneumonia': 5451, 'COVID-19': 286}
Final stats
Train count:  {'normal': 7966, 'pneumonia': 5451, 'COVID-19': 286}
Test count:  {'normal': 885, 'pneumonia': 594, 'COVID-19': 100}
Total length of train:  13703
Total length of test:  1579
```

## Run COVID-Net

To run the COVID-Net training process, a model is needed in the 
```
COVID-Net/models
```
directory. Refer [here](https://github.com/lindawangg/COVID-Net/blob/master/docs/models.md) to download the pretrained models.

Once the models are ready, we can start training from a pretrained model by running:

```
python train_tf.py  --weightspath models/COVIDNet-CXR3-B  --metaname model.meta  --ckptname model-1014
```

A sample output can be like this:

```
root@a668c372cb86:/COVID-Net/data# ls
Actualmed-COVID-chestxray-dataset  COVID-19-Radiography-Database  Figure1-COVID-chestxray-dataset  covid-chestxray-dataset  rsna-pneumonia-detection-challenge  test  train
root@a668c372cb86:/COVID-Net# python train_tf.py     --weightspath models/COVIDNet-CXR3-B     --metaname model.meta     --ckptname model-1014
Output: ./output/COVIDNet-lr0.0002
13417 286
WARNING:tensorflow:From train_tf.py:58: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
2020-06-01 16:16:58.372968: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-06-01 16:16:58.387976: E tensorflow/stream_executor/cuda/cuda_driver.cc:318] failed call to cuInit: CUDA_ERROR_UNKNOWN: unknown error
2020-06-01 16:16:58.388120: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (a668c372cb86): /proc/driver/nvidia/version does not exist
2020-06-01 16:16:58.389620: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-01 16:16:58.433609: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2600000000 Hz
2020-06-01 16:16:58.439325: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x57eea00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-01 16:16:58.439384: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From train_tf.py:59: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
WARNING:tensorflow:From train_tf.py:60: The name tf.train.import_meta_graph is deprecated. Please use tf.compat.v1.train.import_meta_graphinstead.
WARNING:tensorflow:From train_tf.py:73: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.
WARNING:tensorflow:From train_tf.py:77: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializerinstead.
Saved baseline checkpoint
Baseline eval:
[[95.  5.  0.]
[ 5. 94.  1.]
[ 5.  4. 91.]]
Sens Normal: 0.950, Pneumonia: 0.940, COVID-19: 0.910
PPV Normal: 0.905, Pneumonia 0.913, COVID-19: 0.989
Training started
  41/1678 [..............................] - ETA: 1:16:07
```
