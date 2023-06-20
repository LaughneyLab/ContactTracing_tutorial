

[![DOI](https://zenodo.org/badge/625036312.svg)](https://zenodo.org/badge/latestdoi/625036312)


# ContactTracing_tutorial

This repository contains a jupyter notebook and scripts to take you through an example ContactTracing analysis.

The code can all be run within a docker image. Commands to get started:
```
WORKDIR=/local/workdir   # set this to directory where you keep your input files
mkdir -p $WORKDIR        # create directory if doesn't exist
cd $WORKDIR
git clone https://github.com/LaughneyLab/ContactTracing_tutorial.git
docker pull docker.io/biohpc/scrna2023
docker run --rm -d -v $WORKDIR:/data -p 8888:8888 docker.io/biohpc/scrna2023 /root/scripts/startJupyter.sh /data
```

This should output a container ID (or you can retrieve it with `docker ps`). You can use this container ID to get the notebook token using the command: `docker exec -it <containerID> jupyter lab list`. This should provide the URL/token you need to log into the jupyter notebook.

Use your browser to go to this URL, or you can go to localhost:8888 and enter the token.

Then, open the notebook, which will be stored in `/data/ContactTracing_tutorial/tutorial.ipynb`. You will need input files including an anndata file and an interactions file -- these are described in more detail in the notebook. You can place the files in $WORKDIR on your machine, and they will be found in /data within in the notebook.

