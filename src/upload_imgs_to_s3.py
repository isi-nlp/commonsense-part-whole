# coding: utf-8
import boto
import boto3
import csv
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
s3
bucket = 'commonsense-mturk-images'
res = s3.put_object(Body=open('/home/jamesm/commonsense-part-whole/data/nouns/vg_imgs/person_luggage/2317616.png'), Bucket=bucket, ContentType='image/png', Key='2317616')
res = s3.put_object(Body=open('/home/jamesm/commonsense-part-whole/data/nouns/vg_imgs/person_luggage/2317616.png', 'rb'), Bucket=bucket, ContentType='image/png', Key='2317616')
res
res = s3.put_object(Body=open('/home/jamesm/commonsense-part-whole/data/nouns/vg_imgs/person_luggage/2317616.png', 'rb'), Bucket=bucket, ContentType='image/png', Key='2317616.png')
res = s3.put_object(Body=open('/home/jamesm/commonsense-part-whole/data/nouns/vg_imgs/person_luggage/2317616.png', 'rb'), Bucket=bucket, ContentType='image/png', Key='2317616.png', ACL='public-read')
import os
os.listdir()
from tqdm import tqdm
for img_dir in tqdm(os.listdir()):
    for fn in os.listdir(img_dir):
        res = s3.put_object(Body=open('/home/jamesm/commonsense-part-whole/data/nouns/vg_imgs/%s/%s' % (img_dir, fn), 'rb'), Bucket=bucket, ContentType='image/png', Key='%s' % fn, ACL='public-read') 
        
from tqdm import tqdm
for img_dir in tqdm(os.listdir()):
    for fn in os.listdir(img_dir):
        s3_dir = '_'.join(img_dir.split())
        res = s3.put_object(Body=open('/home/jamesm/commonsense-part-whole/data/nouns/vg_imgs/%s/%s' % (img_dir, fn), 'rb'), Bucket=bucket, ContentType='image/png', Key='%s/%s' % (s3_dir, fn), ACL='public-read') 
        
