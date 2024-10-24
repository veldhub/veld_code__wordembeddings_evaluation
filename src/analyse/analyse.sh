#!/bin/bash

jupyter nbconvert --to script analyse.ipynb
python analyse.py 
rm analyse.py

