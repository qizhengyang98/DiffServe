#!/bin/bash

# Start the worker process 
cd src/worker/
python worker.py -cip localhost -p 50100 -c sdturbo --is_sink