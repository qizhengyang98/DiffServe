# DiffServe
Text-to-image generation using diffusion models has gained increasing popularity due to their ability to produce high-quality, realistic images based on text prompts. However, efficiently serving these models is challenging due to their computation-intensive nature and the variation in query demands. In this paper, we aim to address both problems simultaneously through query-aware model scaling. The core idea is to construct model cascades so that easy queries can be processed by more lightweight diffusion models without compromising image generation quality. Based on this concept, we develop an end-to-end text-to-image diffusion model serving system, DIFFSERVE, which automatically constructs model cascades from available diffusion model variants and allocates resources dynamically in response to demand fluctuations. Our empirical evaluations demonstrate that DIFFSERVE achieves up to 24% improvement in response quality while maintaining 19-70% lower latency violation rates compared to state-of-the-art model serving systems.

# Description
This artifact describes the complete workflow to setup the cluster testbed experiments for DiffServe. We describe how to obtain the code, and then describe installing the dependencies. We explain how to run the experiments and plot the results. 

## Structure
The folder `src` contains all the source codes of DiffServe. The folder `traces` contains all workload traces used in our paper and the codes to modify the traces. The folder `logs` contains the logs gathered for each experiment. The folder `plotting` contains the scripts to plot results from the collected logs from the testbed. The folder `models` contains all the discriminators trained for each experiment. The folder `datasets` contains the datasets of images used to compute FID scores.

## Dependencies
For hardware dependencies, DiffServe requires a CPU server with 10 cores, 16G RAM, and a cluster with multiple GPU servers with at least 20G vRAM. 

In our experiments, we used a CPU server for the controller, load\_balancer, sink worker, and client processes, and a cluster with 16 A100-80G GPUs for other workers.

For software dependencies, Linux OS, Python=3.8, and several Python packages listed in the *requirements.txt*.

## Environments and Set-up
Necessary dependencies are listed in the requirements.txt. A Conda environment is recommended for installation.

### Conda environment setup
To set up the environments, firstly clone the repository. Go to the root folder and create a conda environment:
```
conda create -n diffserve python=3.8 ;
conda activate diffserve 
```
Then install all the necessary dependencies:
```
pip install -r requirements.txt ;
```
### Gurobi license
To use DiffServe, we recomment obtaining a Gurobi license. You need to obtain a Gurobi license as following.
- Follow the instructions [here](https://www.gurobi.com/solutions/licensing/) to get a commercial or a free academic license for Gurobi, depending on your use.
- Once you have obtained the license, Gurobi will provide a `gurobi.lic` file.
- Place the license file under the path `gurobi/gurobi.lic`.

## Run Experiments
### Preparation
To run all the benchmarks and reproduce the results in the experiments, you can download the pre-trained discriminators by simply running
```
python prepare_ds_mod.py
```
under the conda environment, which automatically downloads all models from [Google Drive](https://drive.google.com/drive/folders/1gF1wKHxaA1DAnAkeDBGvuoHPS7aSRaYz?usp=sharing) and puts them under the correct paths.

### Experiments
We provide the end-to-end experiments for all cascade pipelines used in our paper and corresponding trace files. [Here](experiment.md) are step-by-step instructions on how to execute the experiments.

# Citation
Welcome to cite our work if you find it helpful to your research
```
@misc{ahmad2024diffserveefficientlyservingtexttoimage,
      title={DiffServe: Efficiently Serving Text-to-Image Diffusion Models with Query-Aware Model Scaling}, 
      author={Sohaib Ahmad and Qizheng Yang and Haoliang Wang and Ramesh K. Sitaraman and Hui Guan},
      year={2024},
      eprint={2411.15381},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2411.15381}, 
}
```
