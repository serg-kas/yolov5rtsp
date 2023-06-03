#!/bin/bash

# Start the first process
./mediamtx &

# Start the second process
python app.py &
#./app.py &

# Wait for any process to exit
#wait -n
wait

# Exit with status of process that exited first
exit $?
