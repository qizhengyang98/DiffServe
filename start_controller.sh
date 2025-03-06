#!/bin/bash

# Get the IP address
IP_ADDRESS=$(hostname -I | awk '{print $1}')

# Use the IP address in a command
echo "The IP address is: $IP_ADDRESS"

export GRB_LICENSE_FILE=$(realpath "gurobi/gurobi.lic")

# Start the controller process 
cd src/controller/
python controller.py -ap 5 -c sdturbo