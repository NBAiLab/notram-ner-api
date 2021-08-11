# NoTram Project NER API

This repository contains an API for using the Named Entity Recognition (NER) model based on [NoTraM](https://github.com/NBAiLab/notram).

The API can be deployed using Docker as follows:
```
$ DEVICE=-1 URN_PATH=<> docker-compose up --build
```
where DEVICE is the index of the GPU to use (currently unsupported, -1 for CPU), and URN_PATH is the path to the folder containing all the URNs.
