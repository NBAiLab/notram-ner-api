#!/bin/bash

regex="[Tt][Rr][Uu][Ee]"
if echo $ENABLE_TASK_QUEUE | grep -Eq $regex;
then
  echo "Starting celery..."
  celery -A app.tasks worker -P solo --detach
fi

#uvicorn app.main:app --host 0.0.0.0 --port 80