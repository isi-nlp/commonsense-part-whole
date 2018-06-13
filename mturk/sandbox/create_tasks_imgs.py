import boto3
import bs4
import csv, html, os, random, time
from tqdm import tqdm
import xmltodict

from collections import defaultdict

MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

def make_hit(example, full, form_str, w):
    form = "<![CDATA[\n" + form_str + "\n\n]]>" #dumb hack to make the unparse() output valid
    full['HTMLQuestion']['HTMLContent'] = form
    prompt = html.unescape(xmltodict.unparse(full))
    new_hit = mturk.create_hit(
                               Title = 'Common sense visual reasoning 2', #TODO: remove the number
                               Description = 'View some images and choose the option that best describes the possibility of some statements about an object.',
                               Keywords = 'images, quick, question answering',
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
examples = [row for row in csv.reader(open('/home/jamesm/commonsense/data/adjectives/sample_50_pws.csv'))]
#parse examples into (whole, part) : {jjs} lookup
pw2jjs = defaultdict(set)
for whole, part, jj in examples:
    pw2jjs[(whole, part)].add(jj)

with open('/home/jamesm/commonsense/mturk/sandbox/hit_batches/batch_%d.csv' % round(time.time()*1000), 'w') as of:
    w = csv.writer(of)
    w.writerow(['url', 'HITId', 'whole', 'part', 'jj'])

    print("making HITs...")
    #iterate in random order
    for num_ex,((whole, part), jjs) in tqdm(enumerate(sorted(pw2jjs.items(), key=lambda x: random.random()))):
        prompt = open('prompt_imgs.xml').read()
        full = xmltodict.parse(prompt)
        soup = bs4.BeautifulSoup(full['HTMLQuestion']['HTMLContent'], 'lxml')
        form = soup.body.form

        #replace images
        pw = '_'.join([whole, part])
        new_srcs = [fn for fn in os.listdir('/home/jamesm/commonsense/data/nouns/vg_imgs/%s/' % pw) if fn.endswith('png')]
        for i,img in enumerate(form.findChildren('div')[0].findChildren('img')):
            img['src'] = '/home/jamesm/commonsense/data/nouns/vg_imgs/%s/%s' % (pw, new_srcs[0])

        #TODO: remove this
        if num_ex > 1:
            break
        #update parse with whole, jj
        initial_sentence = soup.new_tag('h4')
        #basic logic for a vs. an
        #det = 'an' if jj[0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
        #initial_sentence.string = 'Each of the three pictures below has a %s, outlined with a blue box. Notice that each %s also has a %s. The %s is outlined with a red box.' % (whole, whole, part, part)
        initial_sentence.string = form.findChildren('div')[0].h4.text.replace('WHOLE', whole).replace('PART', part)
        form.findChildren('div')[0].h4.replace_with(initial_sentence)

        num_in_hit = 0
        for i, jj in enumerate(jjs):
            #replace followup sentence
            followup_sent = soup.new_tag('h4')
            div = form.findChildren('div')[num_in_hit+1]
            #followup_sent.string = "Now consider a new %s. If I told you that the %s is %s, which of the following is true?" % (whole, whole, jj)
            followup_sent.string = div.h4.text.replace('WHOLE', whole).replace('ADJECTIVE', jj)
            div.h4.replace_with(followup_sent)

            #also replace radio button labels
            for label in div.findChildren('label'):
                old_str = label.text
                new_label = soup.new_tag('label')
                new_label.string = old_str.replace('WHOLE', whole).replace('PART', part).replace('ADJECTIVE', jj)
                label.replace_with(new_label)

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
