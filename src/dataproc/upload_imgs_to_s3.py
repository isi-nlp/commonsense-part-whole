"""
    Does what it says on the filename
"""
import boto3
import csv
import os
from tqdm import tqdm

with open('/home/jamesm/.aws/credentials.csv') as f:
    r = csv.reader(f)
    next(r)
    creds = next(r)
    iam_access = creds[2]
    iam_secret = creds[3]

s3 = boto3.client('s3',
                     aws_access_key_id = iam_access,
                     aws_secret_access_key = iam_secret,
                     region_name='us-west-1'
                     )

bucket = 'commonsense-mturk-images'
BASE_DIR = '/home/jamesm/commonsense-part-whole/data/nouns/vg_imgs3/'
for img_dir in tqdm(os.listdir(BASE_DIR)):
    for fn in os.listdir('%s/%s' % (BASE_DIR, img_dir)):
        s3_dir = '_'.join(img_dir.split())
        res = s3.put_object(Body=open('%s/%s/%s' % (BASE_DIR, img_dir, fn), 'rb'), Bucket=bucket, ContentType='image/png', Key='%s/%s' % (s3_dir, fn), ACL='public-read') 
