#!/usr/bin/env bash
celery -A app.tasks worker -P solo --detach