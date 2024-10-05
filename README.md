# PV Power Forecasting

## Key Concepts

:star2: **Large PV data**: We use a large-scale PV dataset. The dataset contains 1-hour resolution data mainly from [DKASC](https://dkasolarcentre.com.au/) (Desert Knowledge Australia Solar Centre)

:star2: **Validated on the Various Sites**: The trained model is validated on 4 different countries (England, Germany, Korea, and USA) and each countries has multiple sites. 

(need to add figures later)

## Results

(add contents after conducting experiments...)

## Getting Started

Our model basically comes from the PatchTST model. We modified the model to fit the PV power forecasting task.
Though you can follow the instructions from the original PatchTST repo, we provide the detailed instructions for the PV power forecasting task.

1. Install pytorch. You can install pytorch from the official website: https://pytorch.org/get-started/locally/  
   (However, we tested on pytorch 2.0.1, CUDA 11.7.)

2. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Create a seperate folder ```./dataset``` and put all the csv files in the directory.

3. Training. All the scripts are in the directory ```./scripts/PatchTST```. The default model is PatchTST/42. For example, if you want to get the multivariate forecasting results for weather dataset, just run the following command, and you can open ```./result.txt``` to see the results once the training is done:
```
sh ./scripts/PatchTST/weather.sh
```

You can adjust the hyperparameters based on your needs (e.g. different patch length, different look-back windows and prediction lengths.). We also provide codes for the baseline models.



## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/yuqinie98/PatchTST

## Contact

If you have any questions or concerns, please contact us: bakseongho@gm.gist.ac.kr or submit an issue

## Citation

If you find this repo useful in your research, please consider citing our paper as follows:

```
@inproceedings{
adding soon
}
```

