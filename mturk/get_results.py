import argparse
import boto3
import csv
import xmltodict
from collections import defaultdict

MTURK_URL = 'https://mturk-requester.us-east-1.amazonaws.com'

def update_responses(responses, q_id, free_text):
    ix = int(q_id[-1]) - 1
    if q_id.startswith('response'):
        if responses[ix] is None:
            responses[ix] = free_text
    elif 'nonsense' in q_id:
        responses[ix] = free_text
    return responses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_id", type=str, help="ID for the batch you want results for")
    parser.add_argument("status", choices=['Submitted', 'Approved', 'Rejected'], help="What kind of results you want")
    parser.add_argument("--num-assignments", dest="num_assignments", type=int, default=3, help="how many assignments there were for this batch (default: 3)")
    args = parser.parse_args()

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
                         endpoint_url = MTURK_URL
                         )

    with open('hit_batches/batch_%s.csv' % args.batch_id) as f:
        with open('hit_results/batch_%s.csv' % args.batch_id, 'w') as of:
            w = csv.writer(of)
            r = csv.reader(f)
            header = next(r)
            for i in range(args.num_assignments):
                header.extend(['result%d' % (i+1), 'worker%d' % (i+1)])
            w.writerow(header)
            hit_ids = set()
            for row in r:
                hit_id = row[2]
                jjs = row[-1].split(';')
                print(hit_id)

                worker_results = mturk.list_assignments_for_hit(HITId=hit_id, AssignmentStatuses=[args.status])

                responses = defaultdict(list)
                worker_ids = defaultdict(list)
                if worker_results['NumResults'] > 0:
                    for assignment in worker_results['Assignments']:
                        worker_id = assignment['WorkerId']
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
                            responses[jj].append(res)
                            worker_ids[jj].append(worker_id)
                    #import pdb; pdb.set_trace()
                else:
                    print("No results ready yet")
                    continue

                for jj, res in zip(jjs, responses):
                    to_write = row[:5]
                    to_write.append(jj)
                    for res, worker_id in zip(responses[jj], worker_ids[jj]):
                        to_write.append(res)
                        to_write.append(worker_id)
                    w.writerow(to_write)
                print()
