import boto3
import csv
import xmltodict

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

def get_response(response, q_id, free_text):
    if q_id == 'possibility':
        if response is None:
            response = free_text
    elif q_id == 'pw-nonsense':
        response = 'pw-nonsense'
    elif q_id == 'pjj-nonsense':
        if response == 'pw-nonsense':
            response = 'both-nonsense'
        else:
            response = 'pjj-nonsense'
    return response

# Use the hit_id previously created
with open('hit_batches/batch_1528749931259.csv') as f:
    with open('hit_results/batch_1528749931259.csv', 'w') as of:
        w = csv.writer(of)
        r = csv.reader(f)
        header = next(r)
        header.extend(['result1', 'result2', 'result3'])
        w.writerow(header)
        for row in r:
            to_write = row[:5]
            hit_id = row[1]
            print(hit_id)

            # We are only publishing this task to one Worker
            # So we will get back an array with one item if it has been completed

            worker_results = mturk.list_assignments_for_hit(HITId=hit_id, AssignmentStatuses=['Submitted'])

            responses = []
            if worker_results['NumResults'] > 0:
                for assignment in worker_results['Assignments']:
                    response = None
                    xml_doc = xmltodict.parse(assignment['Answer'])
                    #import pdb; pdb.set_trace()

                    if type(xml_doc['QuestionFormAnswers']['Answer']) is list:
                        # Multiple fields in HIT layout
                        for answer_field in xml_doc['QuestionFormAnswers']['Answer']:
                            response = get_response(response, answer_field['QuestionIdentifier'], answer_field['FreeText'])
                    else:
                        # One field found in HIT layout
                        answer_field = xml_doc['QuestionFormAnswers']['Answer']
                        response = get_response(response, answer_field['QuestionIdentifier'], answer_field['FreeText'])
                    responses.append(response)
            else:
                print("No results ready yet")

            to_write.extend(responses)
            w.writerow(to_write)
            print()
