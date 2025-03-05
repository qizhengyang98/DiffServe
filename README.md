# DiffServe
Text-to-image generation using diffusion models has gained increasing popularity due to their ability to produce high-quality, realistic images based on text prompts. However, efficiently serving these models is challenging due to their computation-intensive nature and the variation in query demands. In this paper, we aim to address both problems simultaneously through query-aware model scaling. The core idea is to construct model cascades so that easy queries can be processed by more lightweight diffusion models without compromising image generation quality. Based on this concept, we develop an end-to-end text-to-image diffusion model serving system, DIFFSERVE, which automatically constructs model cascades from available diffusion model variants and allocates resources dynamically in response to demand fluctuations. Our empirical evaluations demonstrate that DIFFSERVE achieves up to 24% improvement in response quality while maintaining 19-70% lower latency violation rates compared to state-of-the-art model serving systems.

# Description
This artifact describes the complete workflow to setup the cluster testbed experiments for DiffServe. We describe how to obtain the code, and then describe installing the dependencies. We explain how to run the experiments and plot the results. 

## Structure
The folder "src" contains all the source codes of DiffServe. Under "src". The folder "traces" contains all workload traces used in our paper and the codes to modify the traces. The folder "logs" contains the logs gathered for each experiment. The folder "plotting" contains the scripts to plot results from the collected logs from the testbed. 

## Dependencies
For hardware dependencies, DiffServe requires a CPU server with 10 cores, 16G RAM, and a cluster with multiple GPU servers with at least 20G vRAM. In our experiments, we used a CPU server for the controller, load_balancer, sink worker, and client processes, and a cluster with 16 A100-80G GPUs for other workers.

For software dependencies, Linux OS, Python=3.8, and several Python packages listed in the *requirements.txt*.

The environment used in the paper: Ubuntu 20.04 LTS, NVIDIA Quadro RTX 8000 GPU.

## Environments and Set-up
Necessary dependencies are listed in the requirements.txt. A Conda environment is recommended for installation.

To set up the environments, firstly clone the repository. Go to the root folder and create a conda environment:
```
conda create -n gmorph python=3.8 ;
conda activate gmorph 
```
Then install all the necessary dependencies:
```
pip install -r requirements.txt ;
cd test_metamorph/transformers/ ;
pip install -e . 
```
Go to the root folder and install GMorph package:
```
cd ../.. ;
pip install metamorph/ 
```
Then all the necessary dependencies should be installed. To do a simple test, go to the folder metamorph/test and run
```
python test.py
```
If the computation graph of models is printed successfully, then installation is done.

## Run benchmarks
### Preparation
To run all the benchmarks and reproduce the results in the experiments, firstly prepare the necessary datasets and pre-trained single-task models.

To prepare them automatically, you can simply run the script *prepare\_ds\_mod.sh*, which downloads all datasets and models and puts them under the correct paths, after installing *gdown* package by
```
pip install gdown
```
Alternatively, you can manually download them from [Google Drive](https://drive.google.com/drive/folders/1Dtvd5eIDeDiseCAwCrj3_wrqjWsy3bq3?usp=sharing) and put them under the corresponding folders:
- Put *datasets.zip* under the root directory and unzip it. There should be four folders under datasets: adience, ESOS, fer2013, VOCDetection;
- Put *salientNet.model*, *salientNet_vgg16.model*, *objectNet.model* under *test_metamorph/scene/pre_models*;
- Put *EmotionNet.model*, *EmotionNet_vgg13.model*, *ageNet.model*, *genderNet.model*, *genderNet_vgg11.model* under *test_metamorph/face/pre_models*;
- Put *age_gender.gz* under *metamorph/data*;
- Put *toy_vgg13.pt* under *metamorph/model*;
- Put *cola.zip*, *sst2.zip*, *multiclass.zip*, *salient.zip* under *test_metamorph/transformer_model* and unzip them.

### Experiments
Then we can Run GMorph for different benchmarks and generate well-trained multi-task models.

There are several shell scripts named *submit_xxx.sh* under the root directory, which are used to evaluate different benchmarks in this experiment. We will execute the shell scripts with proper arguments. The script *figure7table5.sh* is used to reproduce the results in Figure7, and the script *figure8.sh* is used to reproduce the results in Figure8 and Table5.

Under the *benchmark_scripts* folder, there are also separate scripts provided to run all the experiments for each benchmark without manually changing arguments, and which script corresponds to which experiment is written in the script *figure7table5.sh*.

There are some useful configurations on some arguments:
- policy_select: set *SimulatedAnnealing* when testing *GMorph*, set *LCBased* when testing *GMorph w P* and *GMorph w P+R*.
- log_name: the name of the log file, which saves useful intermediate information when GMorph is running.
- acc_drop_thres: the threshold of accuracy drop. 
- enable_filtering_rules: whether or not to enable rule-based filtering. Add this flag when testing *GMorph w P+R*, and remove this flag when testing *GMorph w P*. This flag is useful only when *policy_select=LCBased*.

Other arguments and flags do not need to be changed during evaluations. Note that the arguments of *batch_size* and *num_workers* can be smaller if GPU memory is not enough.

For different benchmark-n, run *submit_bn.sh*, and modify the flags or arguments as shown above. To run experiments without manually changing flags or arguments, go to the *benchmark_scripts* directory and run corresponding scripts. 

### Reproduce results
To reproduce the results shown in Figure7,8 and Table5, run scripts *figure7table5.sh* and *figure8.sh* accordingly. Note that running these scripts can be time-consuming, which basically runs all the experiments for all the benchmarks, so an alternative way is to run each experiment separately given the comments in the scripts.

When the shell script or commands inside are running, a corresponding log file will be generated under *results/log*. The log will record the architecture of the model, the accuracy and latency of the model, and the overall search time at the end of each iteration. Note that since GMorph is based on a random algorithm, the outcomes during the model searching and the final multi-task models may be similar but not exactly the same between different runs. It would be better to run each benchmark multiple times to generate multiple logs to minimize the influence of randomness.

To reproduce the results in Table4, run the shell script *table4.sh*, the latency of all-shared models and multi-task models found by TreeMTL in benchmark 1-4 will be printed. 

To reproduce the results in Table3, run the shell script *table3.sh*, the model architectures found by GMorph will be compiled by both PyTorch and TensorRT automatically, and the results, which are the latency of the models, will be printed. 12G GPU memory is needed for benchmark-6 and 15G is needed for benchmark-7.

# Citation
Welcome to cite our work if you find it helpful to your research
```
@article{gmorph2024,
  title={GMorph: Accelerating Multi-DNN Inference via Model Fusion},
  author={Yang, Qizheng and Yang, Tianyi and Xiang, Mingcan and Zhang, Lijun and Wang, Haoliang and Serafini, Marco and Guan, Hui},
  journal={Proceedings of Nineteenth European Conference on Computer Systems}
  year={2024}
}
```
