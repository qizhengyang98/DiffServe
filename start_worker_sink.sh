#!/bin/bash

# Start the worker process 
cd src/worker/
python worker.py -cip 10.100.10.25 -p 50100 -c sdturbo --is_sink