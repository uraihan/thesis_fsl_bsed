# Few-Shot Learning for Bioacoustics Sound Event Detection

This repository is the implementation of my Master of Science (Tech.) thesis project of the said title. It is heavily inspired by [the DCASE 2023 Task 5 submission by Moummad, et al.,](https://dcase.community/documents/challenge2023/technical_reports/DCASE2023_Moummad_IMT_t5.pdf) titled, "Supervised Contrastive Learning for Pre-Training Bioacoustic Few-Shot Systems". I highly recommended readers to read the original paper.

The approach consists of:
<ul>
<li>Training a feature extractor on the training set</li>
<li>Training a linear classifier on each audio of the validation set (finetuning)</li>
</ul>

# Dataset

This project used the DCASE 2024 Task 5 Development Set that can be downloaded from [Zenodo]{https://zenodo.org/records/10829604}

# Training

1. Extract spectrograms of the training set by running the ```create_train.py``` script. This will output a ```.h5``` file that you can use for training.\

Required arguments:\
- ```--traindir```: Specify the directory of the training set.
- ```--features```: Specify the feature to be extracted. Choose between ```melspec``` for mel-spectrogram or ```pcen``` for Per-Channel Energy Normalization (PCEN).
\
Example:\
```python create_train.py --traindir /path/of/your/trainset --features melspec```

2. Train feature extractor by running the ```train.py``` script. This will create the model, run training on the model, and output the trained model.\

Required arguments:\
- ```--traindir```: Specify the directory of the training set.
- ```--device``` Specify the device to train on (e.g. 'cuda', 'cpu', etc.)
- ```--method```: Specify the method of learning (loss function) to train the encoder. Currently supporting Supervised Contrastive Loss ("scl"), Angular Margin Loss ("aml"), Angular Contrastive Loss ("acl"), and Self-Supervised Learning ("ssl").
\
Other optional arguments regarding training and data augmentation hyperparameters can be found in ```args.py``` along its default values.
\
Example:\
```python train.py --traindir /path/of/your/trainset --device cuda:0 --method acl --h5file name-of-your-h5-file.h5```

3. Finetune and validate the learned feature extractor on validation set using N-shots framework by running the ```evaluate.py``` script. This will output a ```.csv``` file that contains predicted annotation that can be directly used to calculate evaluation score using [the DCASE 2024 Task 5 evaluation score repository]{https://github.com/c4dm/dcase-few-shot-bioacoustic/tree/main/evaluation_metrics} provided by DCASE.\

Required arguments:\
- ```--traindir```: Specify the directory of the training set.
- ```--valdir```: Specify the directory of the validation sets
\
Other optional arguments regarding finetuning hyperparameters and their default values can also be found in ```args.py```.
\
Example:\
```python evaluate.py --valdir /path/of/your/valset --ft 0 --ftlr 0.01 --ftepochs 20 --method ce --adam```


# Evaluation

To get the scores, run the ```evaluation_metrics/evaluation.py``` script. This whole folder was taken directly from the DCASE 2024 [Task 5 evaluation metrics repository]{https://github.com/c4dm/dcase-few-shot-bioacoustic/tree/main/evaluation_metrics}

Required arguments:
- ```-pred_file```: Location of the predictions csv file created by ```evaluate.py``` (the default location of the file is in : traindir/../../outputs/eval.csv').
- ```-ref_files_path```: Path of validation dataset.
- ```-savepath```: Location of the directory to save the score in form of ```json``` file

Example:\
```python evaluation_metrics/evaluation.py -pred_file /path/of/output.csv -ref_files_path /path/of/your/valset -savepath /path/of/score/output```

---

### Credits
Huge thank you to Moummad et al., for the DCASE 2023 submission solution that heavily inspires this thesis project. Big thank you as well for my supervisors, Dr. Irene Martin Morato and Dr. Annamaria Mesaros, for guiding me in my thesis journey. Thank you to Shanshan Wang for the guidance and work on Angular Margin Contrastive Loss that became one of the main focus point of this project.
