# IOB for CMB spectra
---
**Environment**: Run `Conda env create -f operon.yaml` to replicate the environment I have tested. All dependence including `torch` are in `operon.yaml`; **may need to pip install some package, please check the operon.yaml**

**Data source**: infinity: /data77/xiaosheng/IOB_cmb_data/camb_new_TT.zip 

 cp and `unzip camb_new_TT.zip` under the `./data/camb_new/` folder; There are 10000 raw TT samples that is about 196 Mb. Data format: data[:6] are the 6 cosmological parameters; data[6:] are the TT power spectrum with l={2,3,...2500}
 
Similarly, for **Polarized CMB**:

- TE: /data77/xiaosheng/IOB_cmb_data/camb_new_TE.zip;
- EE: /data77/xiaosheng/IOB_cmb_data/camb_new_EE.zip;

can unzip under the `./data/camb_new/` folder for further use

**With command lines**:

- **python generate_ps.py** (optionally)
- **python create_noised_data.py --cmb_type=TT**(optionally, need run at the first time)
- **python train.py --model_name=shallow96 --epoch=2048 --cmb_type=TT** (may need to comment `os.environ["CUDA_VISIBLE_DEVICES"]` in the beginning)
- **python test.py --model_name=shallow96 --cmb_type=TT** (may need to comment `os.environ["CUDA_VISIBLE_DEVICES"]` in the beginning)
- **python operon.py --model_name=shallow96 --cmb_type=TT --operon_name=run1**
- **python replace.py --model_name=shallow96 --cmb_type=TT --operon_name=run1 --thin=200** . For better visualization, check the `inspect the data space` section in the `IOB_for_CMB_spectra.ipynb`. (may need to comment `os.environ["CUDA_VISIBLE_DEVICES"]` in the beginning)

the last command will output two csv files "pareto\_good\_model\_\*" and "individuals\_good_model\_" under the `./data/sr/` folder. They are the "good" expressions of the first latent from operon. When replacing the first latent with the outputs from these expressions, the final weighted mse in the data space is less than 1. The two files correspond to the expressions from the pareto front and all 2000 individual expressions during each operon run.

**File structure** under `IOB_cmb`: has a trained model in `model`, which can be used for testing directly.
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

`References`
- [camb notebook](https://camb.readthedocs.io/en/latest/CAMBdemo.html)
- [pyoperon](https://github.com/heal-research/pyoperon)
- Refer to [Pytorch IOBs](https://github.com/maho3/pytorch-iobs) and [paper](https://arxiv.org/abs/2305.11213) for more about the IOB methodology and applications.
