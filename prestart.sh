#!/usr/bin/env bash

if $ENABLE_TASK_QUEUE;
then
  echo "Starting celery..."
  celery -A app.tasks worker -P solo --detach
fi