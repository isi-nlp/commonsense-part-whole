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

print("I have $" + mturk.get_account_balance()['AvailableBalance'] + " in my Sandbox account")
