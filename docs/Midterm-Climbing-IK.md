# Midterm: Inverse Kinematics for Climbing 
## Prof. Jacky Baltes <jacky.baltes@ntnu.edu.tw>
## Prof. Reinhard Gerndt <r.gerndt@ostfalia.de>
## Due Date: 15th November 2026 - 23:59 Taiwan Time

# Midterm

This project is part of the midterm examination in the course Intelligent Humanoid Robotics at the National Taiwan Normal University (NTNU) in Taiwan.

This project must be done individually.

After the submission of your project, you will demonstrate your knowledge of the material in an oral examination. 
The oral exam will include questions about this project as well as any other material covered in class.

# Introduction

The goal of this midterm is to evaluate the suitablility of various robot kinematics for climbing walls and bouldering tasks. 
See [rock-climbing-robots.github.io](https://rock-climbing-robots.github.io) for more information on rock climbing robots.

In particular, we investigate the kinematics of three popular humanoid robot [https://en.robotis.com/model/page.php?co_id=prd_op3](Robotis-OP3), 
[https://github.com/Rhoban/microban/tree/main](Micoban), and Alex. 

# Setup

We will setup the [https://mujoco.org](Mujoco simulator) with the three robot models as shown in the figure below.

![Alex, Op3, and Microban with Climbing Wall](https://i.postimg.cc/pTwbg0js/world.png)
A basic mujoco world setup to get you started is available at [https://github.com/ciolh007/assignment-climbing-ik.git](https://github.com/ciolh007/assignment-climbing-ik.git).

# Analysis

Implement a program that uses an inverse kinematics algorithm (analytical, Jacobian methods, quadratic methods [https://github.com/Rhoban/placo](https://github.com/Rhoban/placo)).

We will create a voxel space to visualize the reachable space for the kinematics of each robot. 
Create a workspace (box) in front of the robot with suitable dimensions (limited by the arm or leg length of the robot).

Divide this space into small 1cm * 1cm * 1cm cubes. For each cube, set a target location in the center of the cube, calculate a path using inverse kinematics, and check if the path can be executed without collissions using the mujoco simulator. If the ik path is successful, add a cube in a transparent color to the image, otherwise delete the cube.

Repeat this process for each of the four limbs (left hand, right hand, left foot, and right foot).
Use different colors to visualize the cubes of the various limbs.

# Output

Your program should generate an image showing the reachable voxel space in front of each robot.

# Evaluation

You are expected to carefully evaluate the kinematics of the three robots to determine which robot is more
suitable for climbing walls as the one shown in the image above.

Compare the number of voxels that can be reached by the robot at different distances in front of the robot.

# Report

Create a 2 to 5 page report describing your approach to this project. 

# Marking

Due to the increased capabilities of current state of art AI systems and the increased use of these systems by students, the submission of the project will be marked as pass/fail woth 1% of the final grade.

You must demonstrate the capabilties of your system and answer questions about your submission in an oral exam after the submission deadline, which is worth 19% of the final grade.
