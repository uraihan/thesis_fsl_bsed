# Master's Thesis in Few-Shot Learning for Bioacoustics Sound Event Detection
Umair Raihan
---

This repository is heavily inspired by [DCASE 2023 Task 5 submission by Moummad, et al.,](https://dcase.community/documents/challenge2023/technical_reports/DCASE2023_Moummad_IMT_t5.pdf) titled, "Supervised Contrastive Learning for Pre-Training Bioacoustic Few-Shot Systems". I highly recommended readers to read the original paper.

The approach consists of:
<ul>
<li>Training a feature extractor on the training set</li>
<li>Training a linear classifier on each audio of the validation set (finetuning)</li>
</ul>

# Training Pipeline

1. Create the spectrograms of the training set :\
```create_train.py``` : with argument ```--traindir``` for the folder containing the training datasets and ```--method``` to switch loss functions used when training the encoder between Super Contrastive Loss ("scl"), Angular Margin Loss ("aml"), and Angular Contrastive Loss ("acl").

2. Train feature extractor:\
```train.py```: with arguments ```--traindir``` (the same as above), ```--device``` the device to train on, and others concerning training and data augmentation hyperparameters that can be found in ```args.py``` with its default values.

3. To validate the learned feature extractor using 5-shots :\
```evaluate.py``` : with arguments ```--valdir``` for the folder containing the validation datasets, and others concerning hyperparameters that can also be found in ```args.py```. For example, you can pass :\
```--ft 0 --ftlr 0.01 --ftepochs 20 --method ce --adam```\


To get the scores using DCASE 2024 Task 5 evaluation metrics:\
```evaluation_metrics/evaluation.py``` : with arguments ```-pred_file``` for the predictions csv file created by ```evaluate.py``` (the default location of the file is in : traindir/../../outputs/eval.csv'), ```-ref_files_path``` for the path of validation datasets, and ```-savepath``` for the folder where to save the scores json file

---

### Credits
Huge thank you to Moummad et al., for the DCASE 2023 submission solution that heavily inspires this thesis. Big thank you as well for my supervisors, Dr. Irene Martin Morato and Dr. Annamaria Mesaros, for guiding me in my thesis journey. And last but not least, thank you to Shanshan Wang for the guidance and work on Angular Margin Contrastive Loss that becomes one of the main focus point of this thesis.
