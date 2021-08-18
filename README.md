# NoTram Project NER API

This repository contains an API for using the Named Entity Recognition (NER) model based
on [NoTraM](https://github.com/NBAiLab/notram).

The API can be deployed using Docker as follows:

```bash
$ <ENV_VARIABLE_1=VALUE...> docker-compose up --build
```

### Endpoints

- `GET /entities/groups`

  Returns available entity groups.
- `POST /entities/text`

  Returns a list of found entities in a given text.
- `POST /entities/website`

  Returns a list of found entities in a given URL.
- `POST /entities/urn`

  Returns a list of found entities in a given URN.

All the POST methods have options to group and/or filter entities by type, as well as a wait option for task queue.

### Environment variables

- **ENABLE_TASK_QUEUE** ("True" or "False")

  Whether or not to use a task queue to handle requests, default "False".
- **URN_BASE_PATH**

  Path to URN directory (requires a specific structure). Default is None, meaning URN endpoint is disabled.
- **MODEL_PATH**

  Path to model, default "model"
- **DO_BATCHING** ("True" or "False")

  Whether or not to batch inputs, default "False"
- **DEVICE**

  The index of the GPU to use, or -1 for CPU (default)
- **SPLIT_LANG**

  Language code for sentence splitter, default "no". "Disable" will disable sentence splitting (using strided windows
  instead)
