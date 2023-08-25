# IOB for CMB spectra
---
`References`
- [camb notebook](https://camb.readthedocs.io/en/latest/CAMBdemo.html)
- [pyoperon](https://github.com/heal-research/pyoperon)
    
`Main contents`
* [Prepare the data](#data)
  - [Generate the TT/TE/EE data (re-generate if need)](#create_cmb_data)
  - [Create noised data](#noised_data)
* [Train the encoder-decoder](#train)
* [Test and variation of the latents](#test)
* [SR with pyoperon](#operon)
* [Inspect the data space](#weighted)

**README:**

**Environment**: Run `Conda env create -f operon.yaml` to replicate the environment I have tested. All dependence including `torch` are in `operon.yaml`;

**Data source**: infinity: /data77/xiaosheng/IOB_cmb_data/data.zip 

 cp and `unzip data.zip` under `IOB_cmb`; The 10000 raw TT data under the `./data/camb_new/` folder, which is about 196 Mb. Data format: data[:6] are the 6 cosmological parameters; data[6:] are the TT power spectrum with l={2,3,...2500}

**With jupyter notebook**

* Run [Generate the TT/TE/EE data (re-generate if need)](#create_cmb_data) (and uncomment the corresponding lines) to generate TE and EE data, can skip if do not need;
* Run [Create noised data](#noised_data) to post-process the TT data (adding cosmic variance and rescaling);
* Can just turn to [Train the encoder-decoder](#train), if the data are already post-processed;
* Can just turn to [Test and variation of the latents](#test), [SR with pyoperon](#operon), and [Inspect the data space (replacement of latents)](#weighted), if you do not want to retrain the model.

**With command lines**:

- **python generate_ps.py** (optionally)
- **python create_noised_data.py --cmb_type=TT**(optionally, need run at the first time)
- **python train.py --model_name=shallow96 --epoch=2048 --cmb_type=TT** (may need to comment `os.environ["CUDA_VISIBLE_DEVICES"]` in the beginning)
- **python test.py --model_name=shallow96 --cmb_type=TT** (may need to comment `os.environ["CUDA_VISIBLE_DEVICES"]` in the beginning)
- **python operon.py --model_name=shallow96 --cmb_type=TT --operon_name=run1**
- **python replace.py --model_name=shallow96 --cmb_type=TT --operon_name=run1 --thin=200** (may need to comment `os.environ["CUDA_VISIBLE_DEVICES"]` in the beginning)

the last command will output two csv files "pareto\_good\_model\_\*" and "individuals\_good_model\_" under the `./data/sr/` folder. They are the "good" expressions of the first latent from operon. When replacing the first latent with the outputs from these expressions, the final weighted mse in the data space is less than 1. The two files correspond to the expressions from the pareto front and all 2000 individual expressions during each operon run.

**Polarized CMB**:

- TE: /data77/xiaosheng/IOB_cmb_data/camb_new_TE.zip;
- EE: /data77/xiaosheng/IOB_cmb_data/camb_new_EE.zip;

can unzip under the `data/camb_new/` folder for use

**File structure** under `IOB_cmb`:
```
IOB_cmb
└───README.md
└───operon.yaml (environment)
└───data
|    └─── camb_new (raw data)
|    └─── camb_new_processed (processed data)
|    └─── sr (outputs of symbolic regression)
└───model (save the best checkpoint during training) 
└───process files
|    └─── generate_ps.py
|    └─── create_noised_data.py
|    └─── train.py
|    └─── test.py
|    └─── operon.py
|    └─── replace.py   
└───functional files
|   └─── operon_sklearn.py
|   └─── pytorchtools.py
|   └─── model_cmb.py
|   └─── load_data_cmb.py
└───example notebook
|   └─── IOB_for_CMB_spectra.ipynb
```