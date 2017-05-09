# Image translation by CycleGAN and pix2pix in Tensorflow

This is my ongoing tensorflow implementation for unpaired image-to-image translation ([Zhu et al., 2017](https://arxiv.org/pdf/1703.10593.pdf)). 

Latest results can be found here, [comparing](docs/run_1.md) paired and unpaired image-to-image translation as well as the showing the transfer of pre-trained networks [transfer of pre-trained networks](docs/run_1.md) between them.

Image-to-image translation learns a mapping from input images to output images, like these examples from the original papers: 

#### CycleGAN: [[Project]](https://junyanz.github.io/CycleGAN/) [[Paper]](https://arxiv.org/pdf/1703.10593.pdf) [[Torch]](https://github.com/junyanz/CycleGAN)
<img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg" width="900"/>

#### Pix2pix:  [[Project]](https://phillipi.github.io/pix2pix/) [[Paper]](https://arxiv.org/pdf/1611.07004v1.pdf) [[Torch]](https://github.com/phillipi/pix2pix)

<img src="https://phillipi.github.io/pix2pix/images/teaser_v3.png" width="900px"/>

## Prerequisites
- Linux or OSX.
- Python 2 or Python 3.
- CPU or NVIDIA GPU + CUDA CuDNN.

## Requirements
Tensorflow 1.0

## Preferred
- Anaconda Python distribution
- PyCharm

## Getting Started
Clone this repository
```sh
git clone https://github.com/tbullmann/imagetranslation-tensorflow.git
cd imagetranslation-tensorflow
```
Install Tensorflow, e.g. [with Anaconda](https://www.tensorflow.org/install/install_mac#installing_with_anaconda)

Create directories or symlink
```sh
mkdir datasets  # or symlink; for datasets
mkdir temp  # or symlink; for checkpoints, test results
```
Download the CMP Facades dataset (generated from http://cmp.felk.cvut.cz/~tylecr1/facade/)
```sh
python tools/download-dataset.py facades datasets
```
Train the model (this may take 1-8 hours depending on GPU, on CPU you will be waiting for a bit)
```sh
python translate.py \
  --model pix2pix \
  --mode train \
  --output_dir temp/facades_train \
  --max_epochs 200 \
  --input_dir datasets/facades/train \
  --which_direction BtoA
```

Test the model
```sh
python translate.py \
  --model pix2pix \
  --mode test \
  --output_dir temp/facades_test \
  --input_dir datasets/facades/val \
  --checkpoint temp/facades_train
```
The test run will output an HTML file at `temp/facades_test/index.html` that shows input/output/target image sets.

For training of the CycleGAN use ```--model CycleGAN``` instead of ```--model pix2pix```. 
Both models use u-net as generator by default but can use faststyle-net when specified by ```--generator faststyle```.

You can look at the loss and computation graph for [pix2pix](docs/run_1_images/Graph_Pix2Pix.png) and [CycleGAN](docs/run_1_images/Graph_CycleGAN.png) using tensorboard:

```sh
tensorboard --logdir=temp/facades_train
```  

If you wish to write in-progress pictures as the network is training, use ```--display_freq 50```. This will update ```temp/facades_train/index.html``` every 50 steps with the current training inputs and outputs.

## TODO

### Finish CycleGAN implementation according to publication Hu et al., 2017
Major issues
- replace the negative log likelihood objective by a least square loss 
- add instance normalization ([Ulyanov D et al., 2016](https://arxiv.org/abs/1607.08022))
- update discriminators using a history of generated images by adding image buffer that stores the 50 previous image
Minor issues
- flexible learning rate for the Adams solver
- unpair images for testing CycleGAN
- add one-direction test mode for CycleGAN
- add identity loss
- add flexibility to padding layers, e.g. reflection as well

### Merge paired and unpaired translastion
- refactor summary to work for both Pix2Pix and CycleGAN
- add dropout layers to faststyle net

### Import and export
- add import of images from different subdirectories, and of image stacks from multi-tiff and hdf5
- images with arbitrary height width and color channels (input/target)
- add mask for unlabeled regions (modify last layer in discriminator (32x32) and L1 loss layer (256x256) before tf.reduce_mean), e.g. here:
```python
  discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
  # ....
  gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
  gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        
```
- add more preprocessing options for augmentation
- move lab_colorization(split input image into brightness and color)

### Testing
- fully test CPU mode and multi-GPU mode
- add different generators 

### Other
- resize-convolution on top of deconvolution for better upsampling ([Odena et al., 2016](http://distill.pub/2016/deconv-checkerboard/))

## Done
- test CycleGAN with u-net generator and log loss and compare with pix2pix: [OK](docs/run_1.md) 
- test CycleGAN with faststyle-net generator and log loss: [OK](docs/run_1.md) 
- proper loss function for generator (maximising discriminator loss) and optional square loss
- test Pix2Pix2 model, transfer checkpoint to CycleGAN: [OK](docs/run_2.md) 

## Acknowledgement

This repository is based on [this](https://github.com/affinelayer/pix2pix-tensorflow) Tensorflow implementation of the paired image-to-image translation ([Isola et al., 2016](https://arxiv.org/pdf/1611.07004v1.pdf)) 

## Citation

If you use this code for your research, please cite the papers this code is based on:

```tex
@article{pix2pix2016,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  journal={arxiv},
  year={2016}
}

@article{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  journal={arXiv preprint arXiv:1703.10593},
  year={2017}
}

@inproceedings{johnson2016perceptual,
  title={Perceptual losses for real-time style transfer and super-resolution},
  author={Johnson, Justin and Alahi, Alexandre and Fei-Fei, Li},
  booktitle={European Conference on Computer Vision},
  pages={694--711},
  year={2016},
  organization={Springer}
}
```