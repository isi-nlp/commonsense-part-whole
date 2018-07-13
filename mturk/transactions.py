from collections import namedtuple
import csv

import boto3
import numpy as np
import pandas as pd
from tqdm import tqdm


MTURK_URL = 'https://mturk-requester.us-east-1.amazonaws.com'
with open('/home/jamesm/.aws/credentials2.csv') as f:
    r = csv.reader(f)
    next(r)
    creds = next(r)
    iam_access = creds[2]
    iam_secret = creds[3]

mturk = boto3.client('mturk',
                     aws_access_key_id = iam_access,
                     aws_secret_access_key = iam_secret,
                     region_name='us-east-1',
                     endpoint_url = MTURK_URL
                     )

df = pd.read_csv('/home/jamesm/Downloads/Transactions_2018-07-04_to_2018-07-12.csv')
hits = set(df['HIT ID'].unique())
hitinfo = namedtuple('hitinfo', ['hittypeid', 'title', 'description', 'keywords', 'reward', 'creationtime', 'maxassignments', 'assignmentduration', 'autoapprovaldelay', 'expiration'])
hit2info = {}
for hit_id in tqdm(hits):
    if not pd.isnull(hit_id):
        hit = mturk.get_hit(HITId=hit_id)
        HIT = hit['HIT']
        hit2info[hit_id] = hitinfo(HIT['HITTypeId'], HIT['Title'], HIT['Description'], HIT['Keywords'], HIT['Reward'], HIT['CreationTime'], HIT['MaxAssignments'], HIT['AssignmentDurationInSeconds'], HIT['AutoApprovalDelayInSeconds'], HIT['Expiration'])
    
   
with open('transaction_history2.csv', 'w') as of:
    w = csv.writer(of)
    w.writerow(['HITId', 'HITTypeId', 'Title', 'Description', 'Keywords', 'Reward', 'CreationTime', 'MaxAssignments', 'AssignmentDurationInSeconds', 'AutoApprovalDelayInSeconds', 'Expiration', 'AssignmentId', 'WorkerId', 'AssignmentStatus', 'AcceptTime', 'SubmitTime', 'AutoApprovalTime', 'ApprovalTime', 'WorkTimeInSeconds'])
    for i,row in tqdm(df.iterrows()):
        hit_id = row['HIT ID']
        if pd.isnull(hit_id): continue
        info = hit2info[hit_id]
        assgns = mturk.list_assignments_for_hit(HITId=hit_id)
        for assgn in assgns['Assignments']:
            time = (assgn['SubmitTime'] - assgn['AcceptTime']).total_seconds()
            w.writerow([hit_id, info.hittypeid, info.title, info.description, info.keywords, info.reward, str(info.creationtime), info.maxassignments, info.assignmentduration, info.autoapprovaldelay, info.expiration, assgn['AssignmentId'], assgn['WorkerId'], assgn['AssignmentStatus'], str(assgn['AcceptTime']), str(assgn['SubmitTime']), str(assgn['AutoApprovalTime']), str(assgn['ApprovalTime']), time])
