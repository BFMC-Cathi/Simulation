***Requirements***

Docker 

Git



***Creating the Set Up***

Clone the project to your desired location:

git clone https://github.com/BFMC-Cathi/Simulation

cd BFMC_workspace



***Build the Docker Image***

docker build -t team_cathi/bfmc_env .



***Run the Container for the First Time***

xhost +local:docker

docker run -it \
  --name bfmc_sim \
  --network=bridge \
  --env DISPLAY=$DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --device /dev/dri:/dev/dri \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --volume $(pwd):/home/ros_dev/BFMC_workspace \
  team_cathi/bfmc_env bash

  
  
***To Run Whenever You Want***

xhost +local:docker

docker start bfmc_sim

docker exec -it bfmc_sim bash
