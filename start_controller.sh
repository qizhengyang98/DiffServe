#!/bin/bash

# Get the IP address
IP_ADDRESS=$(hostname -I | awk '{print $1}')

# Use the IP address in a command
echo "The IP address is: $IP_ADDRESS"

# Start the controller process 
cd src/controller/
python controller.py -ap 5 -c sdturbo