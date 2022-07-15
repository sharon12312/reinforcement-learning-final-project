# IDC – Reinforcement Learning - Final Project - 2022

---

## Getting Started
The main goal of this final project is to summarize the main topics that we have discussed in the course using some practice and theory, and especially the second part of course (Deep RL). \
In this project you will solve several variations of the Highway env.
* The environment: https://github.com/eleurent/highway-env
* Additional documentation: https://highway-env.readthedocs.io

Highway is a collection of environments for autonomous driving and tactical decision-making tasks. Our goal is to help the user car overtake the bot cars on its own in a roadway environment. \
We train the user car with the help of deep reinforcement learning, the reward function will penalize the user car every time it slows down, every time it crashes into bot car and if there are any bot cars in front of it. In the following environments we will use the raw pixels as our state space, therefore, it will allow to train CNN Neural Networks.

---

## Instructions

### Upload Weights  
* Download the given _**weights.zip**_ file and extract it into your google drive.
* Once the folder is set, please perform the "_**Import Weights**_" section code within the google-colab notebook. This function loads the weights folder into your colab hosted runtime.
* Grant permissions by clicking on the "_**Connect to Google Drive**_" button to allow the notebook to access the weights folder.
* The folder should contain:
```
├── ex1_w
│   ├── dqn_weights_easy.h5
│   ├── ddqn_weights_easy.h5
│   ├── a3c_icm_weights.h5
│   └── icm_a3c_icm_weights.h5
├── ex2_w
│   ├── dqn_weights_medium.h5
│   └── ddqn_weights_medium.h5
└── ex3_w
    ├── dqn_weights_medium.h5
    └── ddqn_weights_medium.h5
```

### Run the Agents Evaluations
To evaluate each agent and construct the environment's video, you would require to perform the instructions below:
* Run the "Project Algorithms" section (running 25 cells).
  * _As mentioned above, the "**Import Weights**" section will require you to grant access to your google drive in order to load the models' weights to your local host runtime._
* Run the "**Environments Utils**" section to load the environment's handlers.
* Once it is done, in each exercise (_Highway-Env - Easy, Highway-Env - Medium, and Super Highway Agent_), you should run:
  * The "**Config Level**" section configures the proper environment level.
  * The "**Agent Evaluation**" section (in each algorithm, such as DQN, DDQN, and A3C & ICM), to construct the environment's video for each model.


---
## Authors
* Sharon Mordechai.
* Amit Huli.