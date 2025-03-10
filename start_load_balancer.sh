#!/bin/bash

# Get the IP address
IP_ADDRESS=$(hostname -I | awk '{print $1}')

# Use the IP address in a command
echo "The IP address is: $IP_ADDRESS"

# Start the load_balancer process 
cd src/load_balancer/
python load_balancer.py -cip localhost -c sdturbo