import argparse
import boto3
import csv
import xmltodict
from collections import defaultdict

MTURK_URL = 'https://mturk-requester.us-east-1.amazonaws.com'

def update_response(responses, q_id, free_text):
    if q_id == 'comment':
        return responses, free_text, None
    triple = tuple(q_id.split('_')[1:])
    if q_id.startswith('response'):
        if responses[triple] == '':
            responses[triple] = free_text
    elif 'nonsense' in q_id:
        responses[triple] = free_text
    return responses, None, triple

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hit_file", type=str, help="path to the qualifier batch")
    parser.add_argument("result_file", type=str, help="path to save the retrieved annotations")
    parser.add_argument("status", choices=['Submitted', 'Approved', 'Rejected'], help="What kind of results you want")
    parser.add_argument("--num-assignments", dest="num_assignments", type=int, default=300, help="how many assignments there were for this batch (default: 300)")
    args = parser.parse_args()

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

    with open(args.hit_file) as f:
        with open(args.result_file, 'w') as of:
            with open('qualifier_comments.csv', 'a') as cf:
                w = csv.writer(of)
                cw = csv.writer(cf)
                r = csv.reader(f)
                header = next(r)
                hit_ids = set()

                row = next(r)
                hit_id = row[2]
                print(hit_id)

                worker_results = mturk.list_assignments_for_hit(HITId=hit_id, AssignmentStatuses=[args.status], MaxResults=100)
                responses = defaultdict(list)
                worker_ids = defaultdict(list)
                while 'NextToken' in worker_results.keys():
                    next_tok = worker_results['NextToken']
                    print("processing %d responses..." % worker_results['NumResults'])
                    comments = []
                    if worker_results['NumResults'] > 0:
                        for assignment in worker_results['Assignments']:
                            worker_id = assignment['WorkerId']
                            response = defaultdict(str)
                            xml_doc = xmltodict.parse(assignment['Answer'])

                            triples = set()
                            for i,answer_field in enumerate(xml_doc['QuestionFormAnswers']['Answer']):
                                response, comment, triple = update_response(response, answer_field['QuestionIdentifier'], answer_field['FreeText'])
                                triples.add(triple)
                                if comment is not None and comment != '':
                                    print(comment)
                                    comments.append((worker_id, comment))
                            for triple in triples:
                                responses[triple].append(response[triple])
                                worker_ids[triple].append(worker_id)
                    else:
                        print("No results ready yet")

                    worker_results = mturk.list_assignments_for_hit(HITId=hit_id, AssignmentStatuses=[args.status], MaxResults=100, NextToken=next_tok)

                for i in range(len(list(responses.values())[0])):
                    header.extend(['result%d' % (i+1), 'worker%d' % (i+1)])
                w.writerow(header)

                for triple in responses.keys():
                    to_write = row[:3]
                    to_write.extend(list(triple))
                    for res, worker_id in zip(responses[triple], worker_ids[triple]):
                        to_write.append(res)
                        to_write.append(worker_id)
                    w.writerow(to_write)
                print()

                for worker, comment in comments:
                    cw.writerow([hit_id, worker, whole, part, comment])

