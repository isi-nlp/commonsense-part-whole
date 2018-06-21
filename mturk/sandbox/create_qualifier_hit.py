import boto3
import csv

MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

with open('/home/jamesm/.aws/credentials.csv') as f:
    r = csv.reader(f)
    next(r)
    creds = next(r)
    iam_access = creds[2]
    iam_secret = creds[3]

mturk = boto3.client('mturk',
                     aws_access_key_id = iam_access,
                     aws_secret_access_key = iam_secret,
                     region_name='us-east-1',
                     endpoint_url = MTURK_SANDBOX
                     )

prompt = open('prompt_qualifier.xml').read()
title = "Common sense visual reasoning qualifier 2"
qualifier_hit = mturk.create_hit(
                                 Title = title,
                                 Description = 'View some images and choose the option that best describes the possibility of some statements about an object.',
                                 Keywords = 'images, quick, question answering, qualifier',
                                 Reward = '0.10',
                                 MaxAssignments = 300,
                                 LifetimeInSeconds = 60*60*24*2, #2 days
                                 AssignmentDurationInSeconds = 60*30, #30 minutes
                                 AutoApprovalDelayInSeconds = 60*60*24, #1 day
                                 Question = prompt,
                                 QualificationRequirements = [
                                     {
                                         'QualificationTypeId': '00000000000000000040', #number of HITs approved
                                         'Comparator': 'GreaterThanOrEqualTo',
                                         'IntegerValues': [
                                             500,
                                         ],
                                     },
                                     {
                                         'QualificationTypeId': '000000000000000000L0', #percentage assignments approved
                                         'Comparator': 'GreaterThanOrEqualTo',
                                         'IntegerValues': [
                                             98,
                                         ]
                                     }
                                     ]
                                 )
pw2jjs = {('leg', 'knee'): set(['bent']),
          ('boat', 'top'): set(['open']),
          ('woman', 'shoulder'): set(['beautiful']),
          ('jacket', 'collar'): set(['brown']),
          ('woman', 'chain'): set(['beautiful']),
          ('boat', 'letter'): set(['long', 'wooden']),
          ('bed', 'pillow'): set(['wooden']),
          ('road', 'dirt'): set(['open'])}
with open('hit_batches/qualifier_2.csv', 'w') as of:
    w = csv.writer(of)
    w.writerow(['title', 'url', 'HITid', 'whole', 'part', 'jj'])
    
    for (whole, part), jjs in pw2jjs.items():
        w.writerow([title, qualifier_hit['HIT']['HITGroupId'], qualifier_hit['HIT']['HITId'], whole, part, ';'.join(jjs)])
