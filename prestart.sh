#!/bin/bash

regex="[Tt][Rr][Uu][Ee]"
if echo $ENABLE_TASK_QUEUE | grep -Eq $regex;
then
  echo "Starting celery..."
  celery -A app.tasks worker -P solo --detach
fi

if [[ $MODEL_PATH ]]
then
  if [ "$(ls -A $MODEL_PATH)" ]; then
    echo "Local model found at $MODEL_PATH"
  else
    echo "$MODEL_PATH empty, trying downloading"
    python -c "import transformers as tx; m='$MODEL_PATH'; tx.AutoModel.from_pretrained(m); tx.AutoTokenizer.from_pretrained(m)" 2>&1
  fi
else
  if [ "$(ls -A ./model)" ]; then
    echo "Local model found at ./model"
  else
    echo "[WARNING] No model found"
  fi
fi
