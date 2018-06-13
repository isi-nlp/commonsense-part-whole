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

question = open('prompt.xml').read()



new_hit = mturk.create_hit(
                           Title = 'How likely is the followup statement?',
                           Description = 'Choose the option that best describes the possibility of each followup statement given an initial sentence.',
                           Keywords = 'text, quick, labeling',
                           Reward = '0.15',
                           MaxAssignments = 10000,
                           LifetimeInSeconds = 60*60*24*7*2, #2 weeks
                           AssignmentDurationInSeconds = 60*5, #5 minutes
                           AutoApprovalDelayInSeconds = 60*60*4, #4 hours
                           Question = question,
)

print("A new HIT has been created. You can preview it here:")
print("https://workersandbox.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'])
print("HITID = " + new_hit['HIT']['HITId'] + " (Use to Get Results)")

# Remember to modify the URL above when you're publishing
# HITs to the live marketplace.
# Use: https://worker.mturk.com/mturk/preview?groupId=
