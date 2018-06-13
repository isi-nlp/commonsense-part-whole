import boto3
import csv
import xmltodict

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
                     )

# Use the hit_id previously created
hit_id = 'HIT_ID HERE'

# We are only publishing this task to one Worker
# So we will get back an array with one item if it has been completed

worker_results = mturk.list_assignments_for_hit(HITId=hit_id, AssignmentStatuses=['Submitted'])

if worker_results['NumResults'] > 0:
    for assignment in worker_results['Assignments']:
        xml_doc = xmltodict.parse(assignment['Answer'])

        print("Worker's answer was:")
        if type(xml_doc['QuestionFormAnswers']['Answer']) is list:
            # Multiple fields in HIT layout
            for answer_field in xml_doc['QuestionFormAnswers']['Answer']:
                print("For input field: " + answer_field['QuestionIdentifier'])
                print("Submitted answer: " + answer_field['FreeText'])
        else:
            # One field found in HIT layout
            print("For input field: " + xml_doc['QuestionFormAnswers']['Answer']['QuestionIdentifier'])
            print("Submitted answer: " + xml_doc['QuestionFormAnswers']['Answer']['FreeText'])
else:
    print("No results ready yet")
