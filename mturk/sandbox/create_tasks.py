import boto3
import bs4
import csv, html, random, time
from tqdm import tqdm
import xmltodict

from collections import defaultdict

MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

def make_hit(example, full, form_str, w):
    form = "<![CDATA[\n" + form_str + "\n\n]]>" #dumb hack to make the unparse() output valid
    full['HTMLQuestion']['HTMLContent'] = form
    prompt = html.unescape(xmltodict.unparse(full))
    new_hit = mturk.create_hit(
                               Title = 'How likely is the followup statement?', #TODO: remove the number
                               Description = 'Choose the option that best describes the possibility of each followup statement given an initial sentence.',
                               Keywords = 'text, quick, labeling',
                               Reward = '0.01',
                               MaxAssignments = 3,
                               LifetimeInSeconds = 60*60*24, #1 day
                               AssignmentDurationInSeconds = 60*5, #5 minutes
                               AutoApprovalDelayInSeconds = 60*60*4, #4 hours
                               Question = prompt
    )
    w.writerow(["https://workersandbox.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'], new_hit['HIT']['HITId'], *example])

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
                     endpoint_url = MTURK_SANDBOX #TODO: update url
                     )

print("reading examples")
examples = [row for row in csv.reader(open('/home/jamesm/commonsense/data/adjectives/pw-jj-candidates-v2.csv'))]
#parse examples into (whole, jj) : {parts} lookup
wjj2parts = defaultdict(set)
for whole, part, jj in examples:
    wjj2parts[(whole, jj)].add(part)

with open('/home/jamesm/commonsense/mturk/sandbox/hit_batches/batch_%d.csv' % round(time.time()*1000), 'w') as of:
    w = csv.writer(of)
    w.writerow(['url', 'HITId', 'whole', 'part', 'jj'])

    print("making HITs...")
    #iterate in random order
    for num_ex,((whole, jj), parts) in tqdm(enumerate(sorted(wjj2parts.items(), key=lambda x: random.random()))):
        prompt = open('prompt.xml').read()
        full = xmltodict.parse(prompt)
        soup = bs4.BeautifulSoup(full['HTMLQuestion']['HTMLContent'], 'lxml')
        form = soup.body.form

        #TODO: remove this
        if num_ex > 25:
            break
        #update parse with whole, jj
        initial_sentence = soup.new_tag('h4')
        #basic logic for a vs. an
        det = 'an' if jj[0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
        initial_sentence.string = 'Initial sentence: There is %s %s %s.' % (det, jj, whole)
        form.findChildren('div')[0].h4.replace_with(initial_sentence)

        num_in_hit = 0
        for i, part in enumerate(parts):
            #replace followup sentence
            followup_sent = soup.new_tag('p')
            followup_sent.string = "The %s's %s is %s." % (whole, part, jj)
            div = form.findChildren('div')[num_in_hit+1]
            div.p.replace_with(followup_sent)

            #also replace checkbox labels
            new_label = soup.new_tag('label')
            new_label.string = "I don't think a \"%s\" can have a \"%s\"." % (whole, part)
            div.findChildren('label')[0].replace_with(new_label)
            #new_label = soup.new_tag('label')
            #new_label.string = "I don't think a \"%s\" can be \"%s\"." % (part, jj)
            #div.findChildren('label')[1].replace_with(new_label)

            num_in_hit += 1

            if num_in_hit >= 3:
                # we have three followups, create the HIT
                make_hit((whole, part, jj), full, str(soup), w)
                num_in_hit = 0
        #we're done, make a HIT with any remaining
        if num_in_hit > 0:
            #remove extra divs
            for i in range(3, num_in_hit, -1):
                form.findChildren('div')[i].decompose()
            make_hit((whole, part, jj), full, str(soup), w)

# Remember to modify the URL above when you're publishing
# HITs to the live marketplace.
# Use: https://worker.mturk.com/mturk/preview?groupId=
