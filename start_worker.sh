#!/bin/bash

# Start the worker process 
cd src/worker/
python worker.py -cip localhost -p 50051 -c sdturbo --do_simulate