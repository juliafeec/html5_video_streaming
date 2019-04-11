#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:08:22 2019

@author: tianqiluke
"""

import boto3
import sys
def download(filename, path, bucket='msds603camera'):
    """Download a file from s3 to download path"""
    ACCESS_KEY = "AKIA2UZ37BVQGUF5O4XB"
    SECRET_KEY = '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'
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)
    download_path = path + filename
    s3.download_file(bucket, filename, download_path)

if __name__ == '__main__':
    filename = sys.argv[1]
    path = sys.argv[2]
    download(filename, path)