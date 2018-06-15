import boto3
import csv
import xmltodict
from collections import defaultdict

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

def update_responses(responses, q_id, free_text):
    ix = int(q_id[-1]) - 1
    if q_id.startswith('response'):
        if responses[ix] is None:
            responses[ix] = free_text
    elif q_id.startswith('wjj-nonsense'):
        responses[ix] = 'wjj-nonsense'
        if responses[ix] == 'pjj-nonsense':
            responses[ix] = 'both-nonsense'
    elif q_id.startswith('pjj-nonsense'):
        responses[ix] = 'pjj-nonsense'
        if responses[ix] == 'wjj-nonsense':
            responses[ix] = 'both-nonsense'
    return responses

with open('hit_batches/batch_1529089597996.csv') as f:
    with open('hit_results/batch_1529089597996.csv', 'w') as of:
        w = csv.writer(of)
        r = csv.reader(f)
        header = next(r)
        header.extend(['result1', 'result2', 'result3'])
        w.writerow(header)
        hit_ids = set()
        for row in r:
            hit_id = row[2]
            jjs = row[-1].split(';')
            print(hit_id)

            worker_results = mturk.list_assignments_for_hit(HITId=hit_id, AssignmentStatuses=['Submitted'])

            responses = defaultdict(set)
            if worker_results['NumResults'] > 0:
                import pdb; pdb.set_trace()
                for assignment in worker_results['Assignments']:
                    assgn_responses = [None] * len(jjs)
                    xml_doc = xmltodict.parse(assignment['Answer'])

                    if type(xml_doc['QuestionFormAnswers']['Answer']) is list:
                        # Multiple fields in HIT layout
                        for i,answer_field in enumerate(xml_doc['QuestionFormAnswers']['Answer']):
                            assgn_responses = update_responses(assgn_responses, answer_field['QuestionIdentifier'], answer_field['FreeText'])
                    else:
                        # One field found in HIT layout
                        answer_field = xml_doc['QuestionFormAnswers']['Answer']
                        assgn_responses = update_responses(assgn_responses, answer_field['QuestionIdentifier'], answer_field['FreeText'])
                for jj, res in zip(jjs, assgn_responses):
                    responses[jj].add(res)
            else:
                print("No results ready yet")
                continue

            for jj, res in zip(jjs, responses):
                to_write = row[:5]
                to_write.extend([jj, *responses[jj]])
                w.writerow(to_write)
            print()
