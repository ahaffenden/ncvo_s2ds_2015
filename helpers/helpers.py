#! /usr/bin/env python
# Tue Aug 11 08:57:52 BST 2015
# Austin Haffenden

"""
This script contains helper functions
"""

#import ipdb
import os

def ensure_dir(directory):
  """Method that creates a directory if it does not exist"""
  if not os.path.exists(directory):
    os.makedirs(directory)

