# Vision_Beyond_Limits_211672


## Problem Statement
To propose and implement a multi-class classification approach to disaster assessment from
the given data set of post-earthquake satellite imagery. We are provided with post earthquake
satellite imagery along with the GeoJSON file containing the extent of damage of each
building. Our task is to take the images, detect and localise the buildings and then classify
them based on the damage inflicted upon them.

### Relevance
We need a satellite image classifier to inform about the disaster in order for the rescue teams
to decide where to head first based on the damage assessed by our model and arrive at the
more damaged localities and save as many lives as possible.

---

### Methodology

#### UNET
* U-net is an encoder-decoder deep learning model which is known to be used in medical
images. It is first used in biomedical image segmentation. U-net contained three main
blocks, down-sampling, up-sampling, and concatenation.
* The important difference between U-net and other segmentation net is that U-net uses
a totally different feature fusion method: concatenation. It concatenates the feature
channel together to get a feature group. It could decrease the loss of features during
convolution layers.
* The U-Net architecture contains two paths: contraction path (also called as the
encoder, The encoder part is used to capture the context in the image using
convolutional layer) and expanding path (also called as the decoder, The decoder part
is used to enable precise localization using transposed convolutions).
* The main idea behind the U-Net is that during the training phase the first half which is
the contracting path is responsible for producing the relevant information by minimising
a cost function related to the operation desired and at the second half which is the
expanding path the network it would be able to construct the output image.

#### RESNET50
* ResNet stands for ‘Residual Network’. ResNet-50 is a convolutional neural network
that is 50 layers deep.
* Deep residual nets make use of residual blocks to improve the accuracy of the models.
The concept of “skip connections,” which lies at the core of the residual blocks, is the
strength of this type of neural network.

---

## Implementation

### Environment Setup

* During development we used Google colab.
* Our minimum Python version is 3.6+, you can get it from here.
* Once in your own virtual environment you can install the packages required to train and run the baseline model.
* Before installing all dependencies run  'pip install numpy tensorflow' for CPU-based machines or  'pip install numpy tensorflow-gpu && conda install cupy' for GPU-based (CUDA) machines, as they are install-time dependencies for some other packages.
* Finally, use the provided requirements.txt file for the remainder of the Python dependencies like so,  'pip install -r requirements.txt' (make sure you are in the same environment as before)


### Localization Training
Below we will walk through the steps we have used for the localization training.
First, we must create masks for the localization, and have the data in specific folders for the model to find and train itself. The steps we have built are described below:

1. Run mask_polygons.py to generate a mask file for the chipped images.
>
> * Sample call: python mask_polygons.py --input /path/to/xBD --single-file --border 2
> * Here border refers to shrinking polygons by X number of pixels. This is to help the model separate buildings when there are a lot of "overlapping" or closely placed polygons.
> * Run python mask_polygons.py --help for the full description of the options.
2. Run data_finalize.sh to setup the image and labels directory hierarchy that the spacenet model expects (it will also run compute_mean.py script to create a mean image that our model uses during training.
> * Sample call: data_finalize.sh -i /path/to/xBD/ -x /path/to/xView2/repo/root/dir/ -s .75
> * -s is a crude train/val split, the decimal you give will be the amount of the total data to assign to training, the rest to validation.
> * You can find this later in /path/to/xBD/spacenet_gt/dataSplit in text files, and easily change them after we have run the script.
> * Run data_finalize.sh for the full description of the options.


3. After these steps have been run you will be ready for the instance segmentation training.

* The original images and labels are preserved in the ./xBD/org/$DISASTER/ directories, and just copies the images to the spacenet_gt directory.
                           
#### The main file is train_model.py and the options are below

A sample call we used is below(You must be in the ./spacenet/src/models/ directory to run the model):
```
$ python train_model.py /path/to/xBD/spacenet_gt/dataSet/ /path/to/xBD/spacenet_gt/images/ /path/to/xBD/spacenet_gt/labels/ -e 100
```

>WARNING: If you have just ran the (or your own) localization model, be sure to clean up any localization specific directories (e.g. ./spacenet) before running the classification pipeline. This will interfere with the damage classification training calls as they only expect the original data to exist in directories separated by disaster name. You can use the split_into_disasters.py program if you have a directory of ./images and ./labels that need to be separated into disasters.

4. You will need to run the process_data.py python script to extract the polygon images used for training, testing, and holdout from the original satellite images and the polygon labels produced by SpaceNet. This will generate a csv file with polygon UUID and damage type as well as extracting the actual polygons from the original satellite images. If the val_split_pct is defined, then you will get two csv files, one for test and one for train.


### Damage Classification Training
```
$ python damage_classification.py --train_data /path/to/XBD/$process_data_output_dir/train --train_csv train.csv --test_data /path/to/XBD/$process_data_output_dir/test --test_csv test.csv --model_out path/to/xBD/baseline_trial --model_in /path/to/saved-model-01.hdf5
```

Sample Call:
```
 ./utils/inference.sh -x /path/to/xView2/ -i /path/to/$DISASTER_$IMAGEID_pre_disaster.png -p /path/to/$DISASTER_$IMAGEID_post_disaster.png -l /path/to/localization_weights.h5 -c /path/to/classification_weights.hdf5 -o /path/to/output/image.png -y
```

---

### File Structure

Source Code
```
 ┣ classification model
 ┃ ┣ damage_classification.py
 ┃ ┣ damage_inference.py
 ┃ ┣ model.py
 ┃ ┣ process_data.py
 ┃ ┗ process_data_inference.py
 ┣ spacenet
 ┃ ┣ inference
 ┃ ┃ ┗ inference.py
 ┃ ┗ src
 ┃ ┃ ┣ features
 ┃ ┃ ┃ ┣ build_labels.py
 ┃ ┃ ┃ ┣ compute_mean.py
 ┃ ┃ ┃ ┗ split_dataset.py
 ┃ ┃ ┗ models
 ┃ ┃ ┃ ┣ dataset.py
 ┃ ┃ ┃ ┣ evaluate_model.py
 ┃ ┃ ┃ ┣ segmentation.py
 ┃ ┃ ┃ ┣ segmentation_cpu.py
 ┃ ┃ ┃ ┣ tboard_logger.py
 ┃ ┃ ┃ ┣ tboard_logger_cpu.py
 ┃ ┃ ┃ ┣ train_model.py
 ┃ ┃ ┃ ┣ transforms.py
 ┃ ┃ ┃ ┗ unet.py
 ┣ utils
 ┃ ┣ combine_jsons.py
 ┃ ┣ data_finalize.sh
 ┃ ┣ inference.sh
 ┃ ┣ inference_image_output.py
 ┃ ┣ mask_polygons.py
 ┃ ┗ png_to_geotiff.py
 ┣ weights
 ┃ ┗ mean.npy
 ┣ Readme.md
 ┗ requirements.txt
```

---

### Results
| Sr. | Metric | Score |
| --- | --- | --- |
| 1. | ACCURACY | 0.81 |
| 1a. | PIXEL ACCURACY | 0.76 |
| 1b. | MEAN CLASS ACCURACY | 0.80 |
| 2. | IOU | 0.71 |
| 2a. | MEAN IOU | 0.56 |
| 3. | PRECISION | 0.51 |
| 4. | RECALL |0.75 |

---

## CONCLUSION
* The above model achieves quite good accuracy in terms of localization of buildings
from satellite imagery as well as classifying the damage suffered post disaster. It is
very efficient in terms of time required to train the model and size of input dataset
provided.
* The optimum loss and best accuracy for localization training was achieved on 30
epochs. The various methods used such as data augmentation and different loss
functions helped us to avoid overfitting the data.
* Hence, this model will help to assess the post disaster damage, using the satellite
imagery.
* This challenge gave us a lot of insight on the satellite image, multi-classification
problem. It made us realise the crucial need to utilise the advantages of deep
learning to solve practical global issues such as post disaster damage assessment
and much more.
