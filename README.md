# Chemistry3D: Robotic Interaction Benchmark for Chemistry Experiments 🧪
## Welcome to Chemistry3D
Chemistry3D is an advanced chemistry simulation laboratory leveraging the capabilities of IsaacSim. This platform offers a comprehensive suite of both organic and inorganic chemical experiments, which can be conducted in a simulated environment. One of the distinguishing features of Chemistry3D is its ability to facilitate robotic operations within these experiments. The primary objective of this simulation environment is to provide a robust and reliable testing ground for both experimental procedures and the application of robotics within chemical laboratory settings.

The development of this system aims to integrate chemical experiment simulation with robotic simulation environments, thereby enabling the visualization of chemical experiments while facilitating the simulation of realistic robotic operations. This environment supports the advancement of embodied intelligence for robots, the execution of reinforcement learning tasks, and other robotic operations. Through this integrated environment, we aspire to seamlessly combine chemical experiment simulation with robotic simulation, ultimately creating a comprehensive test environment for the entire chemical experimentation process.

![1](https://github.com/WHU-DOUBLE/Chemistry3D/assets/106065071/06e68194-f25d-4b9a-8688-0222beef818a)

Contact me at eis_hy@whu.edu.cn

* **The first thing you need to do to develop and run demos for robotics operations is to make sure that Issac-Sim is already installed on your operating device.**
* [**Issac—Sim**](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)See more details in this link
* [**Chemistry3D Homepage**](https://www.omni-chemistry.com/#/)See more details in this link
* [**Chemistry3D Document**](https://www.omni-chemistry.com/#/)See more details in this link

# How to Download Chemistry3D

Follow the steps below to download the repository, navigate to the correct directory, and run the demo.

## Step 1: Clone the Repository

First, you need to clone the repository from GitHub. Open your terminal and run the following command:

```bash
git clone https://github.com/WHU-DOUBLE/Chemistry3D.git
```

## Step 2: Navigate to the Project Directory

Once the repository is cloned, navigate to the project directory using the cd command:

```bash
cd Chemistry3D
```

## Step 3: Install Dependencies

If the demo requires any dependencies, you need to install them first. Usually, you can find a requirements.txt file in the repository. You can install the dependencies with:

```bash
pip install -r requirements.txt
```

# Run Chemistry3D Demo
**To run the demo of chemistry experiment, follow these steps:**

```bash
omni_python inorganic_demo.py
omni_python inorganic_demo1.py
omni_python organic_demo.py
```
**:rocket:omni_python is Isaac_Sim's python.sh file, make sure he's added to your system variables!:rocket:**

# Transparency Detection Sim2real
To download the dataset we generate in Isaac-Sim, use the link below:
* [**Transparency Dataset**](https://www.omni-chemistry.com/#/)Click this link for dataset
![Dataset](https://github.com/WHU-DOUBLE/Chemistry3D/assets/106065071/49166b9a-662f-4063-86dd-8bc39a2f5453)

* Chemistry3D integrates with **:rocket:segmentation_models.pytorch:rocket:** if you want to doing vision-based tasks. You can refer to the following documentation.
 [**Transparency Dataset**](https://www.omni-chemistry.com/#/)Click this link for segmentation_models.pytorch
You can also try this command to run our transparency detection demo:
```bash
omni_python Transparent_Grasp/Isaac/train_network.py
```

