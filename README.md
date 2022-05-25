# Importance of Alternate Signal Representation in Automatic MR Image Diagnosis


## Abstract
Magnetic Resonance (MR) imaging is an indirect process for generating images, in which the scanner first collects
complex frequency space measurements of the electromagnetic activity within a human subject's body. Then radiologist renders the disease diagnosis by interpreting the magnitudes of the complex images that are reconstructed using the inverse Fourier transform of the acquired measurements. Recently, deep learning (DL) models have achieved state-of-the-art performance in diagnosing multiple diseases using these magnitude images as input. We argue that restricting these powerful models to using data that is interpretable by the human eye is sub-optimal. Instead, in this work, we explore the value of alternate signal representations within the MR pipeline to train DL models for automated diagnosis. Through a systematic set of experiments, we show that restricting DL models to the magnitude does not necessarily provide the best possible accuracy. Furthermore, we show that the network trained with alternate signal representations is robust to input noise and provides enhanced benefits when working with low signal-to-noise ratio data. 
Lastly, the qualitative analysis provides insights into why these alternate representations are better than just the magnitude of the image.

![concept.png](https://github.com/anonycodes/MRI/blob/main/images/concept.png)

## Installation and usage
* __Requirements__ Refer to requirements.txt for installing required dependencies.
```
$ pip install -r requirements.txt
```
* __Data Generation__ The fastMRI can be downloaded from [this link](https://fastmri.med.nyu.edu) and the annotations can be found at [this link](https://github.com/microsoft/fastmri-plus/tree/main/Annotations). The original fastMRI data contains volume level slices so generate the Slice level processed data after updating the required paths in the file:
 ```
$ cd data_processing/knee/
$ python knee_singlecoil.py
```
* Generate the train, validation, and test splits
```
$ cd data_processing/knee/
$ python generate_knee_metadata.py
```
* Train the model using the slurm script
```
$ cd src/
$ python rss_classifier.py
```
Update the data_space flag to use the required input for the model.

## Quantitative Results
We do five runs for the following data spaces to obtain the table below:
* ktoi_w_real
* ktoi_w_imag
* ktoi_w_mag
* ktoi_phase
* ktoi_w_realimag
* ktoi_w_magphase

![table.png](https://github.com/anonycodes/MRI/blob/main/images/table_image.png)

* For noise evaluation, the noise amount can be updated in rss_classifier.py

![noise.png](https://github.com/anonycodes/MRI/blob/main/images/noise_figure.png)

