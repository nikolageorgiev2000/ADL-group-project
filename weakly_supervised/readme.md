# Installation instructions
At the parent folder (not ./weakly_supervised but ./ADL-group-project), run

```git submodule update --init --recursive```

Note this is our own version of sam with dependencies removed so you can't just git clone the original sam repo. This submodule points here [Yeok-c/sam2](https://github.com/Yeok-c/sam2/tree/minimal_dependencies)

1 dependency added to the overall project:
```omegaconf```

# This part of the repo does three things. 

## 1. Using SAM to generate additional segmentation labels for training samples from bounding boxes (using the first 10 images as an example only)
```
# run from weakly_supervised folder
python ./generate_sam_samples_for_training.py --dataset_size 10 
```
This will use the Oxford-IIT data in ```../data```
Then generate the additional samples and output it as ```.pt``` tensors in ```./sam_masks``` and the first 8 for visualization as png

## 2. Using CAM as weakly supervised method
```
python ./cam.py
``` 

## 3. Using CAM as a prompt for SAM 
```
# run from weakly_supervised folder
python ./cam_sam.py --dataset_size 8
```

### Key idea
SAM is very good at segmentation, but is not trained on classification data i.e. if you point to the dog and ask it to segment it, it will likely do well. However if you ask it to segment the class ‘dog’ from the picture it has no ability to do that at all. We want to explore using CAM to prompt a SAM. 

### Background 
SAM is trained with a supervised training method - although it is very good at transferring to domains with no segmentation training data. SAM2 is trained on the SA-1B image segmentation dataset, SA-V video dataset and other video datasets. 
It operates by taking an image and prompts - and it will find semantic masks that segments that item. It may output multiple masks that segment the item. It also has no ability to identify classes (such as what is a cat, what is a dog).

Recent research is interested in how to optimally ‘prompt’ these SAM models with other weakly supervised method. See 
- Automatic Prompt Generation Using Class Activation Maps for Foundational Models: A Polyp Segmentation Case Study
- CS-WSCDNet: Class Activation Mapping and Segment Anything Model-Based Framework for Weakly Supervised Change Detection | IEEE Journals & Magazine | IEEE Xplore, [2305.05803] 
- Segment Anything Model (SAM) Enhanced Pseudo Labels for Weakly Supervised Semantic Segmentation

### Overall pipeline 
Use a trained CAM to output prompts. 	
Use the maximum point of activation map as prompt for SAM
Use a thresholded area of attention map as heuristic to filter out SAM’s segmentation masks (which may sometimes be very small like an eye of the cat)
We also experiment with weighted random sampling the point based on activation of the CAM - but it doesn’t work as well as simply taking the maximum- no results retained here either. 
Feed the output coordinates of the point form CAM into the SAM. SAM will output several masks. Take the highest scored mask.
SAM Model:
Image encoder: MAE pretrained ViT

Total size: 224.4M parameters
Results
```
{
    "classification_accuracy": 1.0,
    "avg_metrics": {
        "iou": 0.7873172445148472,
        "dice": 0.8689747877902603,
        "pixel_precision": 0.8132823664615606,
        "pixel_recall": 0.9537874161059655,
        "pixel_accuracy": 0.8315971172337674,
        "boundary_f1": 0.016792590633678578
    },
    "std_metrics": {
        "iou": 0.1538966013236887,
        "dice": 0.14314345869928824,
        "pixel_precision": 0.1273476663135782,
        "pixel_recall": 0.16356704168606412,
        "pixel_accuracy": 0.10875838949749171,
        "boundary_f1": 0.01970189587070493
    },
    "correct_mask_ratio": 0.8317025440313112
}
```

Full Visualizations
https://drive.google.com/drive/folders/1MkT7KtbhT8KrOmRL2B0WBmmRF_iZv9pN?usp=sharing

# Notes
The ```.py``` files fulfill your assignment requirements but have poor visualization. The ```.pynb``` files have excellent visualization but requires cv2, matplotlib.