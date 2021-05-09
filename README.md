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

In our approach we combined prior work from three main sources:





## Model and Data

## Technical Description (with links to code/data set)

## Performance Evaluation and Optimizations

## Discussion

## Citations

<a id="1">[1]</a> 
Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D. and Riedmiller, M., 2013. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.
