# Description

A test of the multithreading versus multiprocessing libraries of Python, inspired by Dave's Space's video: [threading vs multiprocessing in python](https://www.youtube.com/watch?v=AZnGRKFUU0c)

# Dependencies
* numpy
* pyqtgraph
* PyQt6

# Usage

In thread_v_process_test.py:
* Edit the "instances" variable on line 11 to change how many threads/processes spawn.
* Edit the "test_time" variable on line 12 to change how long each test run takes.
* Edit the "bin_sz" variable on line 14 to control the time step resolution of the data.

Run `python thread_v_process_test.py` to run the program.

# Screenshots

![ ](./images/4instances_1ms_resolution.png)
![ ](./images/16instances_1ms_resolution.png)
![ ](./images/32instances_1ms_resolution.png)

![ ](./videos/4inst_200u_res.mp4)
![ ](./videos/8inst_1ms_res.mp4)
![ ](./videos/16inst_1ms_res.mp4)