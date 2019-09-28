#!/bin/bash

if [ ! -d "/data" ]; then
  mkdir /data
fi

if [ ! -d "/log" ]; then
  mkdir /log
fi

python main.py