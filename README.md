To build package

`conda build .`
`conda install /home/sahmad_umass_edu/.conda/envs/traffic/conda-bld/linux-64/traffic_analysis-1.2.2-py310_0.tar.bz2`

To run on unity:

`srun --pty -p gypsum-1080ti --gres=gpu:1 -t 24:00:00 --mem=32G bash`
`conda activate traffic`
`module load cudnn/cuda11-8.4.1.50`
`module load cuda/11.4.0`

For genderNet, use `module load cuda/11.8.0`

Use conda environment 'traffic' with requirements.txt and python=3.10


Step 1: Pre-process a video file and convert it into a series of images
        that can be read by the traffic analysis pipeline
`python preprocess.py`
Set the desired framerate in the python file

Step 2: Run the traffic analysis pipeline
`python run.py`

Note to self:   Make sure to `pip freeze > requirements.txt` if any
                additional packages are installed


To convert yolov5 models to ONNX:
https://docs.ultralytics.com/yolov5/tutorials/model_export/



In '/work/pi_rsitaram_umass_edu/sohaib/profiling/yolov5':
conda activate traffic
python export.py --weights yolov5s.pt --include onnx --dynamic
