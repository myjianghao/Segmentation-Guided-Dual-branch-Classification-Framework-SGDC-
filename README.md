# Segmentation-Guided-Dual-branch-Classification-Framework-SGDC-
Segmentation Guided Dual-branch Classification Framework(SGDC)

pretrained_model: https://drive.google.com/file/d/1h_ZDd3udIzi4BUWGwbO7Ge5rpdT9VeFp/view?usp=sharing

PSMsFIGC Dataset:https://drive.google.com/file/d/131VxK1jH4pjYEBNuk7WbMkpenltklZFX/view?usp=sharing

trained models in the experiment:

If you want to use it, unzip it in the root directory, Then you can change the property in the config file to reproduce the outcome with the checkpoint of trained models.

The changed PropertyName is as follows:

running_type: 'test'

load_index: 0

version_type: "xxxxxx" (Subfolder name CLS,CLS+LDM,CLS+SEG+CAI,CLS+SEG,CLS+SEG+LDM+CAI)

Note: CLS+SEG should delete the CAI in the U_Res model
