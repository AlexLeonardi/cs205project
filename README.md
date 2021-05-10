# cs205project


<!--- Project Web Site
An important piece of your final project is a public web site that describes all the great work you did for your project. The web site serves as the final project report, and needs to describe your complete project. You can use GitHub Pages, or the README file on the GitHub repository, so you can easily refer to the software at the GitHub repository. You should assume the reader has no prior knowledge of your project and has not read your proposal. It should address the following aspects: -->

<!---
- Description of problem and the need for HPC and/or Big Data
- Description of solution and comparison with existing work on the problem
- Description of your model and/or data in detail: where did it come from, how did you acquire it, what does it mean, etc.
- Technical description of the parallel application, programming models, platform and infrastructure
- Links to repository with source code, evaluation data sets and test cases
- Technical description of the software design, code baseline, dependencies, how to use the code, and system and environment needed to - reproduce your tests
- Performance evaluation (speed-up, throughput, weak and strong scaling) and discussion about overheads and optimizations done
- Description of advanced features like models/platforms not explained in class, advanced functions of modules, techniques to mitigate overheads, challenging parallelization or implementation aspects...
- Final discussion about goals achieved, improvements suggested, lessons learnt, future work, interesting insights…
- Citations
Your web page should include screenshots of your software that demonstrate how it functions. You should include a link to your source code.
-->

## Introduction
### Problem Description

In 2013, a team at DeepMind demonstrated the ability of reinforcement learning (RL) to master control policies using Atari 2600 games. [[1]](#1) Prior to this work, most applications of deep learning required vast amounts of labeled taining data making it difficult to both scale the model and apply the model to unsurpervised problems. In an RL context most learning is typically unsupervised, requiring many iterations to maximize a reward or to determine a policy which maximizes a reward. Our project seeks to employ parallelization within the training of a Deep Reinforcement Learning model to speed up the learning process. 

The need for big compute stems from the Deep RL model itself. Deep RL models to not have a finite training set so they can continue to train forever while continuing to approximate an optimal training policy. Of course from a practical perspective infinite training is not realistic so we must instead maximize performance by speeding up the training proccess. By employing HPC we can parallelize the training of the RL model and increase the number of training iterations completed in a fixed amount of time. We can then measure the impact of these iterations by assessing the agent's performance.

In our poject we applied a deep RL model to the game of Tetris. Tetris is a game in which different shaped blocks fall one at a time to be placed on other fallen blocks. Tetris is an ideal choice for our project because it has a finite state space and relatively simple game play which make it simple to programmatically implement. It has a quantifiable score metric which makes it easy to understand agent performance and therefore conducive to reinforcement learning. We can easily reward higher scores and penalize mistakes which limit score potential. Finally, Tetris is a pattern recognition problem. Blocks that fit together reduce the number of holes and enable larger line clears therefore earning more points. Pattern recognition makes Tetris conducive to neural networks which can recognize these simple patterns quickly.


### Approach and Existing Work

In our approach we began by combining the prior work of three main sources:
- An OpenAI gym environment of Tetris called gym-tetris [[2]](#2)
- A github repository which provides a production-ready framework for reinforcement learning with PyTorch [[3]](#3)
- A Stanford publication on playing tetris using deep reinforcement learning [[4]](#4)

We then built upon these sources with knowledge from the course to parallelize our implementation usin OpenMP and MPI.

## Model and Data

## Technical Description
Our repository can be found here https://github.com/AlexLeonardi/cs205project/.

### Programming Model, Platfom, and Infrastructure


### Environmental Information


### Replication Steps

1. git clone https://github.com/AlexLeonardi/cs205project.git -b master
2. wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip
3. sudo apt install unzip
4. unzip libtorch-cxx11-abi-shared-with-deps-1.4.0+cpu.zip
5. sudo apt-get install cmake
6. sudo apt-get install gcc g++
7. [GET TO PYTORCH DIRECTORY]
8. mkdir build
9. cd build
10. cmake -DCMAKE_PREFIX_PATH=~/libtorch ..
11. make -j4
12. sudo apt update
13. sudo apt install python3-pip
14. pip3 install gym
15. pip3 install nes-py
16. pip3 install gym-tetris
17. pip3 install scikit-build
18. sudo apt install libopenmpi-dev
19. pip3 install mpi4py
21. Upload and execute getid_linux on instance from www.roboti.us
22. Submit computer ID on www.roboti.us to receive product key
23. Upload product key to instance as mjkey.txt
24. Sudo cp mjkey.txt /bin/mjkey.txt
25. wget https://www.roboti.us/download/mjpro150_linux.zip
26. mkdir ~/.mujoco
27. cp mjpro150_linux.zip ~/.mujoco/
28. unzip  ~/.mujoco/mjpro150_linux.zip
29. pip3 install --upgrade setuptools pip
30. pip3 install opencv-python
31. pip3 install baselines
32. Add “export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mjpro150/bin” to .bashrc
33. pip3 install mujoco-py==1.50.1.56
34. pip3 install baselines
35. sudo apt install python3-opencv
36. pip3 install msgpack


## Performance Evaluation and Optimizations

## Discussion

## Citations

<a id="1">[1]</a> 
Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D. and Riedmiller, M., 2013. Playing Atari with Deep Reinforcement Learning. arXiv preprint arXiv:1312.5602.

<a id="2">[2]</a> 
GitHub. Kautenja. gym-tetris - An OpenAI Gym interface to Tetris on the NES.. https://github.com/Kautenja/gym-tetris

<a id="3">[3]</a> 
GitHub. Omegastick. Pytorrch-cpp-rl - PyTorch C++ Reinforcement Learning. https://github.com/Omegastick/pytorch-cpp-rl

<a id="4">[4]</a> 
Stavene, M., Pradhan, S., 2016. Playing Tetris with Deep Reinforcement Learning. Stanford.

