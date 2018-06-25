import argparse, csv
import pandas as pd
import boto3

MTURK_URL = 'https://mturk-requester.us-east-1.amazonaws.com'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_file", type=str, help="path to file to worker scores")
    parser.add_argument("metric", choices=['kappa', 'kappa_linear', 'kappa_quadratic', 'kappa_two_class', 'spearman'], help="which metric criterion to use")
    parser.add_argument("threshold", type=float, help="workers must be above this threshold of agreement to qualify")
    parser.add_argument("--live", const=True, action='store_const', help="flag to actually run this and give workers the qualification")
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

    df = pd.read_csv(args.metrics_file)
    tot_workers = len(df)
    df = df[df[args.metric] > args.threshold]
    print("number of qualified workers: %d/%d" % (len(df), tot_workers))

    if args.live:
        cont = input('About to make and assign qualifications. You sure? (say yes if so) > ')
        if cont == 'yes':
            qual = mturk.create_qualification_type(
                                 Name='commonsense-visual-qualifier',
                                 QualificationTypeStatus='Active',
                                 Description='Invited to participate in the full common sense visual reasoning study'
                                 )
            qual_id = qual['QualificationType']['QualificationTypeId']
            for worker in df['worker_id']:
                mturk.associate_qualification_with_worker(QualificationTypeId=qual_id, WorkerId=worker)
        else:
            print("Quitting")
