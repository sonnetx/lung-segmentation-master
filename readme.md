# Lung segmentation

## Proposed solution
This code tackles semantic segmentation problems with a UNet - fully convolutional network with an encoder-decoder path. High-resolution features from the contracting path are combined with the upsampled output in order to predict more precise output based on this information, which is the main idea of this architecture.

Softmax function was applied to model output and negative log-likelihood loss was used to train network.
Optimization criterion - Adam with 0.0005 learning rate.

Some kinds of data augmentation were used: horizontal and vertical shift, minor zoom and padding.
All images and masks were resized to 512x512 size before passing the network.
To improve performance was decided to use pretrained on ImageNet encoder from vgg11 network.
This approach slightly improves performance and greatly accelerate network convergence.
Vanilla unet configuration doesn't have batch normalization. Nowadays it is used almost every time, so it was added to improve network convergence too.
Such network configuration outperforms other variations of unet without batch norm and pretrained weights on validation dataset so it was chosen for final evaluation

Networks were trained on a batch of 4 images during more than 50 epochs on average.

After 40 epoch network stops to improve validation score and network began to overfit.

Weights with best validation scores were saved into ```models/``` folder. 
Weights description:

- unet-2v: simple unet + augmentation

- unet-6v: pretrained vgg11 encoder + batch_norm + bilinear upscale + augmentation

Implementation of the described above solution using PyTorch you could find in ``scr/`` folder and `main.ipynb` notebook.


## Evaluation
For evaluation of model output was Jaccard and Dice metrics, well known for such kind of computer vision tasks.
Jaccard also is known as Intersection over Union, while Dice is the same with F1 measure. They are both showing almost the same things - overlap between ground truth and calculated mask. 

## References
- https://arxiv.org/pdf/1505.04597.pdf - U-Net: Convolutional Networks for Biomedical Image Segmentation
- https://arxiv.org/pdf/1811.12638.pdf - Towards Robust Lung Segmentation in Chest Radiographs with Deep Learning
- https://arxiv.org/pdf/1801.05746.pdf - TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation
- https://arxiv.org/pdf/1708.00710.pdf - Accurate Lung Segmentation via Network-WiseTraining of Convolutional Networks
