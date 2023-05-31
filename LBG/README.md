# Learning by Grouping (LBG)

This repository contains the code for the paper:
***Fair and Accurate Decision Making through Group-Aware Learning***

The codebase is built upon implementations of:
- Base models: ResNet, Transformers, BERT, RoBERT
- [DARTS](https://github.com/quark0/darts)

> Note: The sample code defaults to 2 groups. To replicate the results from the paper, follow the experimental settings detailed in the main paper and appendix.

## Requirements

Python 3.8 (or higher)

## Installing dependencies

The dependencies are listed in the `requirements.txt`.
To install them all, do

```
pip install -r requirements.txt
```



## Usage:
**Composing LBG with ResNet:**
You can clone the models that you want to use as your GSCMs. For instance, this repository may be helpful: https://github.com/rwightman/pytorch-image-models.git

Then run lbg_train.py for image classification experiments. Optional enhancements such as cutout and droppath can also be employed. After LBG or DALBG training, the models can be fine-tuned on the entire training set, provided that we used only a portion of the dataset for LBG or DALBG training.

### Example:
For ISIC 2018 Task 3 experiments, go to example/ISIC-18 path and ensure the following setup:

- Place four csv files (metadata.csv, ISIC2018_Task3_Validation_GroundTruth.csv, ISIC2018_Task3_Test_GroundTruth.csv, ISIC2018_Task3_Training_GroundTruth.csv) in the main directory.
- Create three folders for the training, validation, and test images named: ISIC18_test, ISIC18_train, ISIC18_val. The images in ISIC18_test, ISIC18_val, and ISIC18_train should be in .jpg format.

Then, run the following command by replacing the arguments based on your spedicif needs:
``` 
python lbg_train.py --batchsz [batch size] --lr [learning rate] --wd [weight decay] --cat_lr [grouping learning rate] --cat_wd [grouping weight decay] --fair [fair training] --load_balance [using load balance] --report_freq [report after this many iter] --dataset [dataset] --epochs [number of epochs] --image_size [image size] --classes [number of classes] --num_experts [number of experts] --unroll_steps [unrolling steps]
```

Here is an example:

``` 
python lbg_train.py --batchsz 32 --lr 5e-4 --wd 1e-4 --cat_lr 6e-5 --cat_wd 1e-4 --fair --load_balance --report_freq 300 --dataset isic18 --epochs 10 
```

