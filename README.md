## Airway Segmentation Tutorial --- IMR-Summer-School

### Dataset

#### Dataset Folder Structure

```
BAS
└───train
│   │   case1
│   │   case2
│   └───casex
│       │   input
│       │   mask
│       │   processed
│           │   casex_clean.nii
│           │   casex_label.nii
│           └───casex_box.npy
│   
└───test
    │   case1
    │   case2
```

Use the **casex_clean.nii** for input CT volumes and the corresponding **casex_label.nii** for the ground-truth. For space room saving, the input and mask folder may be discarded.

#### How to visualize the Medical images

- [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php) :  portable.
- [3D-Slicer](https://www.slicer.org/) : various modules and extensions.

### Get Start!

1. First, we provide a **Demo Inference** with a baseline trained weight file that helps you get familiar with the
   airway prediction procedure via the CNN output.

   The entrance is at ***main_code/scripts/Demo_Inference/demo_test.py***. You can try the following command:

    ```
    python demo_test.py --input_path $INPUT_PATH --label_path $LABEL_PATH --output_path $OUTPUT_PATH
    ```

   You could use the ITK-SNAP / 3D-Slicer or any visualization tools to check your results.

2. Second, we provide a **Train Pipeline** that helps you train the airway tree modeling task.

   The entrance is at ***main_code/pipeline/pipeline_train_airway_IMR-Summer-School-2022.py***. You can try the
   following command:

    ```
    python pipeline_train_airway_IMR-Summer-School-2022.py --dataroot $DATASET_DIR --name $EXPERIMENT_NAME --checkpoints_dir $MODEL_LOADDIR \ 
    --model $MODEL --dataset_mode $DATASET_MODE --in_channels $INPUT_CH --out_channels $OUTPUT_CH --gpu_ids $GPU_IDS --suffix $SUFFIX  
    ```

   **TIPS:** the detailed arguments for base configures and training procedures are in ***
   main_code/options/base_options*** and ***main_code/options/train_options*** respectively. Please refer to these two
   files and specify the arguments in the default settings or the command line.

3. Third, we provide some ports for you to extend the project. Specifically,
    + Data Augmentation: In ***main_code/dataloader/airway_dataset.py***, you can conduct extra data augmentation.
    + Loss Function: In ***main_code/util/losses.py***, you can construct other loss functions and call it in the ***
      main_code/models/unet3d_model.py***
    + Model Design: In ***main_code/models/*** you can design your own models inherited from the base_model and use the
      modules in the ***main_code/models/networks.py***


### Environment

Python >= 3.8. The deep learning framework is PyTorch=1.11.0 and Torchvision = 0.12.0

Some python libraries are also necessary, you can use the following command to set up.

```
pip install -r requirements.txt
```

 
