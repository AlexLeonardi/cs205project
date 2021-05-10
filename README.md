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

In 2013, a team at DeepMind demonstrated the ability of reinforcement learning (RL) to master control policies using Atari 2600 games. [[1]](#1) Prior to this work, most applications of deep learning requires vast amounts of labeled training data making it difficult to both scale the model and apply the model to unsupervised problems. In an RL context most learning is typically unsupervised, requiring many iterations to maximize a reward or to determine a policy which maximizes a reward. Our project seeks to employ parallelization within the training of a Deep Reinforcement Learning model to speed up the learning process. 

The need for big compute stems from the Deep RL model itself. Deep RL models do not have a finite training set so they can continue to train forever while continuing to approximate an optimal training policy. Of course from a practical perspective infinite training is not realistic, so we must instead maximize performance by speeding up the training process. By employing HPC we can parallelize the training of the RL model and increase the number of training iterations completed in a fixed amount of time. We can then measure the impact of these iterations by assessing the agent's performance.

In our project we applied a deep RL model to the game of Tetris. Tetris is a game in which different shaped blocks fall one at a time to be placed on other fallen blocks. Tetris is an ideal choice for our project because it has a finite state space and relatively simple game play which makes it simple to programmatically implement. It has a quantifiable score metric which makes it easy to understand agent performance and therefore conducive to reinforcement learning. We can easily reward higher scores and penalize mistakes which limit score potential. Finally, Tetris is a pattern recognition problem. Blocks that fit together reduce the number of holes and enable larger line clears therefore earning more points. Pattern recognition makes Tetris conducive to neural networks which can recognize these simple patterns quickly.


### Approach and Existing Work

In our approach we began by combining the prior work of three main sources:
- An OpenAI gym environment of Tetris called gym-tetris [[2]](#2)
- A github repository which provides a production-ready framework for reinforcement learning with PyTorch [[3]](#3)
- A Stanford publication on playing tetris using deep reinforcement learning [[4]](#4)

We then built upon these sources with knowledge from the course to parallelize our implementation using OpenMP and MPI.

[EXPLAIN OPENMP  implementation]

The first parallelization scheme we implemented was OpenMP. Throughout the process of training our Tetris agent, there were several opportunities to introduce OpenMP parallelization. Using PyTorch’s OpenMP backend we could run several Tetris environments simultaneously and parallelize the process of generating a batch. We can control these variables by using num_envs and batch_size, respectively. When it comes to actually accessing OpenMP parallelization, we can use the at::set_num_threads command from PyTorch’s ATen library in conjunction with the standard OMP_NUM_THREADS to specify the number of threads we plan to use, and using at::parallel_for should allow us to mimic the effect of OpenMP’s #pragma parallel for. 

When parallelizing simulations, overhead will likely arise from synchronization since various simulations will take different amounts of time to complete before they can be used to update the network. We expected to see a near-linear speed-up since the simulations will likely account for a large portion of our application’s runtime. We would also expect this speed-up to be scalable to a certain extent since new simulations can be run on the additional threads, but we would expect the marginal speed-up to decrease as thread count increases since the marginal benefit of additional simulations on agent performance decreases depending on how long we have waited since we last updated the network. 

[EXPLAIN MPI  implementation and difficulties]

## Model and Data

We first set up our model by defining the state space, action space, and reward metrics using gym-tetris, a package built on OpenAI Gym. The state space for our rendition of Tetris is a 256x240 RGB image, the action space consists of 6 simple actions (including move left, move right, rotate right, rotate left, move down, and no operation), and the reward is a score that rewards when the agent clears a line and penalizes the height of the Tetris stack.

Then, to train our reinforcement learning agent, we use proximal policy optimization (PPO), an algorithm developed by OpenAI. The PPO schema uses a clipped objective function which searches over a trust region in a method by which gradient descent can be used. PPO also simplifies the typical reinforcement learning by removing the KL penalty from the objective function. Instead, PPO uses an estimated advantage ($\hat{A}$) weighted by the ratio of probabilities, $r$, of achieving certain actions under different policies (parameterized by some theta). PPO provides a balanced approach with relative ease of implementation that minimizes the cost function while ensuring the deviation from the previous policy is relatively small and therefore less stochastic. Additionally, PPO tends to perform relatively well without significant hyperparameter tuning, which makes it an attractive option when one has relatively little time to train an agent. We train our agent in batches, with data collected from multiple simultaneous environments.

We combined this approach with an underlying CNN architecture to read the Tetris game board itself. The CNN convolves over the game board with approximately 1.5 million trainable parameters across 3 convolutional layers and 2 linear layers using ReLU activations.

$$L^{CLIP}(\theta) = \hat{E}_t[min(r_t(\theta)\hat{A}_t,clip(r_t(\theta),1-\varepsilon,1+\varepsilon)\hat{A}_t)]$$
$\theta$ is the policy parameter
$\hat{E}_t$ denotes the empirical expectation over timesteps
$r_t$ is the ratio of the probability under the new and old policies, respectively
$\hat{A}_t$ is the estimated advantage at time $t$
$\varepsilon$ is a hyperparameter, usually 0.1 or 0.2

Since our model seeks to accomplish a reinforcement learning task, there is no need for any external data. All training data will be provided by running simulations of the agent acting on a given state space after choosing an action from the action space based on the agent’s current policy. As we run more simulations, we give the agent more room to learn and encounter novel scenarios, so the load of this problem comes from the computational intensity of running many of these simulations.


## Technical Description
Our repository can be found here https://github.com/AlexLeonardi/cs205project/.

### Programming Environment


- 2 t2.2xlarge instances
- Ubuntu 18.04
- ZMQ Messaging Library

Specifications:
![Specs 1](https://github.com/AlexLeonardi/cs205project/blob/master/images/Screen%20Shot%202021-05-09%20at%2011.43.15%20PM.png)


### Replication Steps


1. git clone https://github.com/AlexLeonardi/cs205project.git -b master
    1. gpu: instead, git clone https://github.com/AlexLeonardi/cs205project.git -b gpu
2. (For GPU: “follow Guide: OpenACC on AWS”)
3. (GPU: CUDA install)
    1. Follow I5, installing cuda 10.2.89 instead of 10.0 or 11.2, i.e. “sudo apt-get install cuda-10.2”
4. wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip
    1. GPU: instead, wget https://download.pytorch.org/libtorch/cu101/libtorch-shared-with-deps-1.4.0.zip
5. sudo apt install unzip
6. unzip libtorch-cxx11-abi-shared-with-deps-1.4.0+cpu.zip
    1. GPU: instead, unzip libtorch-shared-with-deps-1.4.0.zip
7. sudo apt-get install cmake
8. sudo apt-get install gcc g++
9. [GET TO PYTORCH DIRECTORY]
10. (GPU: install libcudnn8)
    1. wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin 
    2. sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    3. sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    4. sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
    5. sudo apt-get update
    6. sudo apt-get install libcudnn8=8.1.1.*-1+cuda10.2
11. sudo apt-get install libcudnn8-dev=8.1.1.*-1+cuda10.2
12. (GPU: sudo apt install nvidia-cuda-toolkit)
13. (GPU: add to ~/.bashrc: )
    1. export PATH=$PATH:/usr/local/cuda/bin export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib:/usr/local/lib export CPLUS_INCLUDE_PATH=/usr/local/cuda/include
14. (GPU: Fix libtorch cuda.cmake file: )
    1. Replace line 148 in libtorch/share/cmake/Caffe2/public/cuda.cmake: file(READ ${CUDNN_INCLUDE_PATH}/cudnn.h CUDNN_HEADER_CONTENTS)
    2. With:
      if(EXISTS ${CUDNN_INCLUDE_PATH}/cudnn_version.h) 
        file(READ ${CUDNN_INCLUDE_PATH}/cudnn_version.h CUDNN_HEADER_CONTENTS)  
      else() 
        file(READ ${CUDNN_INCLUDE_PATH}/cudnn.h CUDNN_HEADER_CONTENTS)
      endif() 
15. mkdir build
16. cd build
17. cmake -DCMAKE_PREFIX_PATH=~/libtorch ..
    1. Gpu: cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2 -DCMAKE_PREFIX_PATH=~/libtorch ..
    2. On GPU, cmake gives a warning message; this can be ignored, since the overlaps causing it don’t lead to runtime errors
18. make -j4
19. sudo apt update
20. sudo apt install python3-pip
21. pip3 install gym
22. pip3 install nes-py
23. pip3 install gym-tetris
24. pip3 install scikit-build
25. sudo apt install libopenmpi-dev
26. pip3 install mpi4py
27. Install Mujoco
    1. Upload and execute getid_linux on instance
    2. Submit computer ID on www.roboti.us to receive product key
    3. Upload product key to instance as mjkey.txt
28. sudo cp mjkey.txt /bin/mjkey.txt
29. wget https://www.roboti.us/download/mjpro150_linux.zip
30. mkdir ~/.mujoco
31. cp mjpro150_linux.zip ~/.mujoco/
32. unzip  ~/.mujoco/mjpro150_linux.zip
33. pip3 install --upgrade setuptools pip
34. pip3 install opencv-python
35. Add “export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mjpro150/bin” to .bashrc (in home directory)
36. pip3 install mujoco-py==1.50.1.56
37. pip3 install baselines
38. sudo apt install python3-opencv
39. pip3 install msgpack

### Running the Program

1. Open two terminals connected to the same instance
2. Move to repository
3. Terminal 1: 
    1. create build directory by running mkdir build 
    2. Move to build repository
    3. Run cmake ..
    4. Run make -j[x] where x is the number of threads used
4. Terminal 2: 
    1. Run python3 ./launch_gym_server.py
5. Terminal 1:
    1. Still in /build run ./example/gym_client
    2. Port output here if desired
6. Edit /example/gym_client.cpp to alter hyperparameters and game

## Performance Evaluation and Optimizations

The objective of our project was to attempt to speed up the training of a deep RL model. In order to measure speed up we sought two main metrics. The first was frames-per-second, a measure built into the gym environment which measured how many frames of the game finished every second. Parallelizing using OpenMP we achieved a tripling in FPS from approximately 10 to 34. 

The second metric we used was holding the execution time off the training process constant. Since one could theoretically train these RL models infinitely, cutting off the training and observing differences in average performance makes more sense from a practical perspective. By observing average performance we can analyze the speed-up provided by weak scaling parallelization. Our analysis of speed-up allows us to determine the increase in the number of training iterations that can be completed in a fixed amount of time from a model trained on a single-core architecture versus a parallel one. 

| **Envs-Threads**     | **Average FPS**    | **Highest Reward (1 hour of training)** |
|------|------|------|
| 1-1 | 11.27 | N/A |
| 2-2 | 19.34       | -20 |
| 4-4   | 30.24   |-19.8   |
| 6-6        |33.83        |-19.5        |
| 8-8        |33.21       |-20        |

<img src="https://github.com/AlexLeonardi/cs205project/blob/master/images/Screen%20Shot%202021-05-10%20at%202.52.16%20PM.png" width="50%" height="50%">
<img src="https://github.com/AlexLeonardi/cs205project/blob/master/images/Screen%20Shot%202021-05-10%20at%202.52.20%20PM.png" width="50%" height="50%">
<img src="https://github.com/AlexLeonardi/cs205project/blob/master/images/Screen%20Shot%202021-05-10%20at%205.21.10%20PM.png" width="50%" height="50%">


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




