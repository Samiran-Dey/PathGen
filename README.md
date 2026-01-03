
# PathGen
## Generating crossmodal gene expression from cancer histopathology predicts multimodal decision utility

The repository contains the implementation of the following paper. \
\
Title - **Generating crossmodal gene expression from cancer histopathology predicts multimodal decision utility** \
Authors - Samiran Dey, Christopher R.S. Banerji, Partha Basuchowdhuri, Sanjoy K. Saha, Deepak Parashar, and Tapabrata Chakraborti \
DOI - [https://doi.org/10.48550/arXiv.2502.00568](https://doi.org/10.1038/s41467-025-66961-9)

## Abstract
Emerging research has highlighted that artificial intelligence based multimodal fusion of digital pathology and transcriptomic features can improve cancer diagnosis (grading/subtyping) and prognosis (survival risk) prediction. However, such direct fusion for joint decision is impractical in real clinical settings, where histopathology is still the gold standard for diagnosis and transcriptomic tests are rarely requested, at least in the public healthcare system. With our novel diffusion based crossmodal generative AI model PathoGen, we show that genomic expressions synthesized from digital histopathology jointly predicts cancer grading and patient survival risk with high accuracy (state-of-the-art performance), certainty (through conformal coverage guarantee) and interpretability (through distributed attention maps). 
\
<img src="./images/Overview.png">  </img>


# Data
To download the diagnostic histopathology images, transcriptomic data, clinical data and other metadata, please refer to the following links:
1. [TCGA-GBM](https://portal.gdc.cancer.gov/projects/TCGA-GBM)
2. [TCGA-LGG](https://portal.gdc.cancer.gov/projects/TCGA-LGG)
3. [TCGA-KIRC](https://portal.gdc.cancer.gov/projects/TCGA-KIRC)




# Getting started

## Installation
To install all requirements execute the following line.
```bash
pip install -r requirements.txt 
```
And then clone the repository as follows. 
```bash
git clone https://github.com/Samiran-Dey/PathGen.git
cd PathGen
```

## Data processing
The folder **Process_data** contains the notebooks to preprocess the data. The details of the notebooks are described below.
1. Process WSI data - To download and obtain embeddings of WSI patches from manifests downloaded from the TCGA website.
2. Process transcriptomic - To store the process transcriptomic data divided into subgroups.
3. Process data - To combine the histopathology and transcriptomic data and store them as tensors to provide input to the models.

The folder **Data** contains some further files required to process the data.
1. signatures.csv - The details of the gene groups.
2. org_minmax_scaler - The trained min-max scalers to normalise the transcriptomic data.



## Synthesizing transcriptomic data from histopathology images
The folder **PathGen** contains the code to synthesize transcriptomic features from whole slide images using our novel diffusion-based model PathGen. The checkpoints for the trained models can be downloaded from [here](https://drive.google.com/drive/folders/1vGwPY9WA81F_tDke4mMjKHAjiPtwBdd3?usp=sharing). 

### Training
To train PathGen execute the following.
```bash
python3 PathGen/main.py --data_root_dir PROCESSED_DATA_PATH --results_dir RESULT_DIRECTORY_PATH --max_epochs NUMBER_OF_EPOCHS
```
To resume training from an intermediate epoch execute the following.
```bash
python3 PathGen/main.py --data_root_dir PROCESSED_DATA_PATH --results_dir RESULT_DIRECTORY_PATH --max_epochs NUMBER_OF_EPOCHS --weight_path PATH_OF_WEIGHT_TO_LOAD --start_epoch START_EPOCH_NUMBER
```

### Inference
To run inference execute the following.
```bash
python3 PathGen/main.py --data_root_dir PROCESSED_DATA_PATH --results_dir RESULT_DIRECTORY_PATH --weight_path PATH_OF_WEIGHT_TO_LOAD --op_mode test
```


## Gradation and Survival Risk Estimation
The folder **MCAT_GR** contains the code for gradation and survival risk estimation using synthesised transcriptomic data obtained using PathGen. The checkpoints for the trained models can be downloaded from [here](https://drive.google.com/drive/folders/1EQTALaJmpReP5n_86SSkUnVcuJdTtVQO?usp=sharing).

### Training using real transcriptomic data
To train the model using real transcriptomic data execute the following.
```bash
python3 MCAT_GR/main.py --data_root_dir PROCESSED_DATA_PATH --results_dir RESULT_DIRECTORY_PATH --max_epochs NUMBER_OF_EPOCHS --data_type real --op_mode train —n_timebin NUMBER_OF_SURVIVAL_TIME_BINS --n_grade NUMBER_OF_GRADES
```

To resume training from an intermediate epoch, execute the following.
```bash
python3 MCAT_GR/main.py --data_root_dir PROCESSED_DATA_PATH --results_dir RESULT_DIRECTORY_PATH --max_epochs NUMBER_OF_EPOCHS --data_type real —op_mode train --best_weight_path PATH_OF_WEIGHT_TO_LOAD --start_epoch START_EPOCH_NUMBER —n_timebin NUMBER_OF_SURVIVAL_TIME_BINS --n_grade NUMBER_OF_GRADES
```

### Inference using synthesized transcriptomic data
To perform inference using synthesised transcriptomic data execute the following.
```bash
python3 MCAT_GR/main.py --data_root_dir PROCESSED_DATA_PATH --results_dir RESULT_DIRECTORY_PATH  --data_type syn --op_mode test --best_weight_path PATH_OF_BEST_WEIGHT --test_syn_path PATH_TO_SYNTHESIZED_TRANSCRIPTOMES —n_timebin NUMBER_OF_SURVIVAL_TIME_BINS --n_grade NUMBER_OF_GRADES
```

To perform distributed inference using synthesised transcriptomic data execute the following.
```bash
python3 MCAT_GR/main.py --data_root_dir PROCESSED_DATA_PATH --results_dir RESULT_DIRECTORY_PATH  --data_type syn --op_mode test --best_weight_path PATH_OF_BEST_WEIGHT --test_syn_path PATH_TO_SYNTHESIZED_TRANSCRIPTOMES —n_timebin NUMBER_OF_SURVIVAL_TIME_BINS --n_grade NUMBER_OF_GRADES --test_type distributed
```

### Calibration and uncertainty estimation
Execute the following to perform calibration using synthesised transcriptomic data and uncertainty estimation using synthesised transcriptomic data.
```bash
python3 MCAT_GR/main.py --data_root_dir PROCESSED_DATA_PATH --results_dir RESULT_DIRECTORY_PATH  --data_type syn --op_mode calibrate --best_weight_path PATH_OF_BEST_WEIGHT --test_syn_path PATH_TO_SYNTHESIZED_TRANSCRIPTOMES —n_timebin NUMBER_OF_SURVIVAL_TIME_BINS --n_grade NUMBER_OF_GRADES
```


# Acknowledgement 
1. [UNI](https://github.com/mahmoodlab/UNI?tab=readme-ov-file)
2. [MCAT](https://github.com/mahmoodlab/MCAT/tree/master?tab=readme-ov-file#downloading-tcga-data)


# Citation
```bash
Dey, S., Banerji, C.R.S., Basuchowdhuri, P. et al. Generating crossmodal gene expression from cancer histopathology improves multimodal AI predictions. Nat Commun (2025). https://doi.org/10.1038/s41467-025-66961-9
```

```bash
@article{Dey2025,
  title = {Generating crossmodal gene expression from cancer histopathology improves multimodal AI predictions},
  ISSN = {2041-1723},
  url = {http://dx.doi.org/10.1038/s41467-025-66961-9},
  DOI = {10.1038/s41467-025-66961-9},
  journal = {Nature Communications},
  publisher = {Springer Science and Business Media LLC},
  author = {Dey,  Samiran and Banerji,  Christopher R. S. and Basuchowdhuri,  Partha and Saha,  Sanjoy K. and Parashar,  Deepak and Chakraborti,  Tapabrata},
  year = {2025}
}
```

## Code DOI
[![DOI](https://zenodo.org/badge/902787586.svg)](https://doi.org/10.5281/zenodo.17315072)

## License and Usage
ⓒ Samiran Dey. The models and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution.



