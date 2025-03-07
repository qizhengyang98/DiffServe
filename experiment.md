# Experiment
we provide instructions on how to execute experiments using the scripts in the artifact. 

DiffServe testbed requires a CPU server and multiple GPU servers.
- `CPU`: a CPU server with 10 cores, 16G RAM for the controller, load\_balancer, sink worker, and client processes.
- `GPU`: multiple servers with powerful GPUs (e.g., A100-40G/80G, L40s, etc) to execute diffusion models. The number of servers depends on how many workers are used in the experiments. For the minimum experiments, 4 servers are needed. In the paper, we used 16 servers for 16 workers.

Alternatively, we provide simulated execution in the artifact, which simulates the execution of diffusion models. In this case, any Nvidia GPUs can be used, and multiple workers can be created on a single GPU servers by using `tmux` such that the experiments can be done with fewer GPUs.

## Workflow
In the following steps, {*Step Num.-R*} means steps for doing real execution of diffusion models, while {*Step Num.-S*} means steps for doing simulated execution.
- `Step 1-R`: For preparation, open one terminal on each GPU server and 4 terminals on CPU server.
- `Step 2-R`: In the terminals of the CPU server, run `tmux new -s contr`, `tmux new -s loadb`, `tmux new -s sink`, `tmux new -s client`, respectively.
- `Step 3-R`: Activate the conda environment in each terminal by running `conda activate diffserve`.
- `Step 4-R`: To run experiments of cascade-1, in `tmux contr` terminal, run the script `start_controller.sh` which starts the controller process. Copy the IP address printed in the console and replace the original IP address after `-cip` in `start_worker.sh`.
- `Step 5-R`: In `tmux loadb` terminal, run `start_load_balancer.sh` which starts the load\_balancer process.
- `Step 6-R`: In `tmux sink` terminal, run `start_worker_sink.sh` which starts the sink worker process.
- `Step 7-R`: In each terminal of GPU server, modify the number after `-p` in `start_worker.sh`, then run the script. This number is the port number of each worker. Make sure the number you assign to each worker is unique, and is between [50051, 50066]. Note that if you execute the worker for the first time, diffusion models will be downloaded automatically to `model` folder.
- `Step 8-R`: If all the processes have been set up successfully, there will be logs corresponding to each process under the folder `logs`. The logs include `controller`, `load_balancer`, `worker_{port number}`, and `model_{port number}`.
- `Step 9-R`: Then in `tmux client`, run `start_client.sh` to start the client process, which keeps sending queries in 6 minutes. Modify the flag `-trace` given the number of workers you allocate. Use `1to8qps`, `2to16qps`, `2_5to20qps`, `3to24qps`, `4to32qps` if you have ~4, 8, 10, 12, 16 workers, respectively.
- `Step 10-R`: The Client process will report `"Trace ended"` when it stops sending queries. Then stop all the processes.
- `Step 11-R`: Under folder `logs`, there will be three csv files which contains the end-to-end experiment results. To generate graphs, go to folder `plotting`, and run the script `run_plot_results.sh`.
- `Step 12-R`: Be sure to remove all the log files before starting a new expriment.

For simulated execution,
- `Step 1-S`: For preparation, open one or multiple terminals on each GPU server and 4 terminals on CPU server. The total number of terminals on GPU servers should be equal to the total number of workers you want to allocate.
- `Step 2.1-S`: In the terminals of the CPU server, run `tmux new -s contr`, `tmux new -s loadb`, `tmux new -s sink`, `tmux new -s client`, respectively.
- `Step 2.2-S`: In each terminal of the GPU servers, run `tmux new -s workerX` respectively, where *X* is a number or character. Making sure *X* is unique to each terminal on a single server.
- `Step 2.3-S`: In `start_worker.sh`, add a flag `--do_simulate` at the end of the python command.
- `Step (3-12)-S`: Steps 3-12 are the same as Step 3-R to Step 12-R explained above.

The above steps describe the end-to-end experiment flow of cascade 1. To run experiments for cascade 2 and 3, simply replace the flag `-c sdturbo` to `-c sdxs` and `-c sdxlltn` in all shell scripts under the root folder, then follow the same steps. For cascade 3, it is recommended to use simulated execution when the number of GPUs is less than 16, as the controller may struggle to find a solution due to insufficient available workers.

## Evaluation
The testbed produces log files in the `logs` folder. The logs files contain snapshots of the system at regular intervals, including resource managements, user demands, system capacity, requests served/dropped/late, and confidence thresholds set given the demand changes.

The python script `plotting/plot_results.py` can generate three graphs with the logs: confidence threshold over time, slo violation ratio over time, and FID score over time, which should be similar to the one in Figure 5. The script also prints the average SLO violation ratio and average FID score, which should be similar to those in Figure 6. The results can vary slightly given different hardwares and trace files in use. To generate the graphs, simply modify the flag `--cascade` with [sdturbo, sdxs, sdxlltn] for different cascades, then run the script `run_plot_results.sh`. 
