#!/bin/bash

regex="[Tt][Rr][Uu][Ee]"
if echo $ENABLE_TASK_QUEUE | grep -Eq $regex;
then
  echo "Starting celery..."
  celery -A app.tasks worker -P solo --detach
fi
