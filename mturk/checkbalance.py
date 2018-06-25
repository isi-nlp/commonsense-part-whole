import boto3
import csv, sys

MTURK_URL = 'https://mturk-requester.us-east-1.amazonaws.com'

which = '' if sys.argv[1] == '1' else '2'
with open('/home/jamesm/.aws/credentials%s.csv' % which) as f:
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

print("I have $" + mturk.get_account_balance()['AvailableBalance'] + " in my live account")
