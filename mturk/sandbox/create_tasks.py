import argparse, csv, html, os, random, sys, time

import boto3
import bs4
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm
import xmltodict

from collections import defaultdict

MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

def replace_template(soup, span, whole=None, part=None, jj=None):
    new_span = soup.new_tag('span')
    if 'WHOLE' in span.text:
        new_span.attrs['class'] = 'whole'
        new_span.string = span.text.replace('WHOLE', whole)
    if 'PART' in span.text:
        new_span.attrs['class'] = 'part'
        new_span.string = span.text.replace('PART', part)
    if 'ADJECTIVE' in span.text:
        new_span.attrs['class'] = 'jj'
        new_span.string = span.text.replace('ADJECTIVE', jj)
    return new_span

def reload_template(whole, part):
    prompt = open('prompt.xml').read()
    full = xmltodict.parse(prompt)
    soup = bs4.BeautifulSoup(full['HTMLQuestion']['HTMLContent'], 'lxml')
    form = soup.body.form

    #update parse with whole, jj
    for span in form.findChildren('div')[2].h4.findChildren('span'):
        new_span = replace_template(soup, span, whole, part)
        span.replace_with(new_span)
    return prompt, full, soup, form

def make_hit(whole, part, jjs, full, form_str, w, title, dryrun):
    form = "<![CDATA[\n" + form_str + "\n\n]]>" #dumb hack to make the unparse() output valid
    full['HTMLQuestion']['HTMLContent'] = form
    prompt = html.unescape(xmltodict.unparse(full))
    title = 'Common sense reasoning %s' % title #TODO: remove the number
    if not dryrun:
        new_hit = mturk.create_hit(
                                   Title = title,
                                   Description = 'Choose the option that best describes the possibility of each followup statement given an initial sentence.',
                                   Keywords = 'text, quick, labeling',
                                   Reward = '0.02',
                                   MaxAssignments = 3,
                                   LifetimeInSeconds = 60*60*24, #1 day
                                   AssignmentDurationInSeconds = 60*5, #5 minutes
                                   AutoApprovalDelayInSeconds = 60*60*4, #4 hours
                                   Question = prompt,
                                   #QualificationRequirements = [
                                   #  {
                                   #      'QualificationTypeId': '00000000000000000040', #number of HITs approved
                                   #      'Comparator': 'GreaterThanOrEqualTo',
                                   #      'IntegerValues': [
                                   #          100,
                                   #      ],
                                   #  },
                                   #  {
                                   #      'QualificationTypeId': '000000000000000000L0', #percentage assignments approved
                                   #      'Comparator': 'GreaterThanOrEqualTo',
                                   #      'IntegerValues': [
                                   #          98,
                                   #      ]
                                   #  }
                                   #]
        )
        w.writerow(["https://workersandbox.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'], new_hit['HIT']['HITId'], whole, part, ';'.join(jjs)])
    else:
        with open("page_%s_%s.html" % (whole, part), 'w') as of:
            of.write(form_str)
        w.writerow([title, "https://workersandbox.mturk.com/mturk/preview?groupId=1", '2', whole, part, ';'.join(jjs)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate_file', type=str, help="Filename of candidate triples to read from")
    parser.add_argument('title', type=str, help="The part of the task title that goes after 'Common sense reasoning'")
    parser.add_argument('--max-hits', default=10, dest='max_hits', type=int, help="maximum number of HITs to make")
    parser.add_argument('--max-pws', default=100000, dest='max_pws', type=int, help="maximum number of part-wholes to make HITs from")
    parser.add_argument('--dry-run', dest='dry_run', action='store_const', const=True, help='flag to not actually make HITs')
    parser.add_argument('--jjs-per-hit', dest='jjs_per_hit', type=int, default=3, help="maximum number of questions per HIT (jj's per part-whole)")
    args = parser.parse_args()

    ############# AWS STUFF #################
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
    #########################################

    print("reading examples")
    examples = [row for row in csv.reader(open('/home/jamesm/commonsense-part-whole/data/candidates/%s' % (args.candidate_file)))]
    lem = WordNetLemmatizer()
    #parse examples into (whole, jj) : {parts} lookup
    pw2jjs = defaultdict(set)
    for whole, part, jj in examples:
        whole_lem = lem.lemmatize(whole.replace(' ', '_')).replace('_', ' ')
        part_lem = lem.lemmatize(part.replace(' ', '_')).replace('_', ' ')
        pw2jjs[(whole_lem, part_lem)].add(jj)

    wholes = set([w for w,_ in pw2jjs.keys()])
    parts = set([p for _,p in pw2jjs.keys()])
    num_triples = sum([len(jjs) for jjs in pw2jjs.values()])
    print("parts: %d, wholes: %d, triples: %d" % (len(parts), len(wholes), num_triples))

    with open('/home/jamesm/commonsense-part-whole/mturk/sandbox/hit_batches/batch_%d.csv' % round(time.time()*1000), 'w') as of:
        w = csv.writer(of)
        w.writerow(['title', 'url', 'HITId', 'whole', 'part', 'jj'])

        print("making HITs...")
        num_hits = 0
        #iterate in random order
        for num_ex,((whole, part), jjs) in tqdm(enumerate(sorted(pw2jjs.items(), key=lambda x: random.random()))):
            if num_hits >= args.max_hits:
                break
            if whole == part or '.' in whole:
                print("whole, part: %s, %s. skipping... " % (whole, part))
            if len(jjs) == 0: continue
            if num_ex > args.max_pws:
                break
            
            whole = whole.replace('_', ' ')
            part = part.replace('_', ' ')
            prompt, full, soup, form = reload_template(whole, part)

            num_in_hit = 0
            hit_jjs = []
            l_i = 0
            for i, jj in enumerate(jjs):
                #replace followup sentence
                div = form.findChildren('div', {'class': 'question'})[num_in_hit]
                for span in div.p.findChildren('span'):
                    new_span = replace_template(soup, span, whole, part, jj)
                    span.replace_with(new_span)

                #also replace radio button labels
                for label in div.findChildren('label'):
                    label.findPreviousSibling().attrs['id'] = str(l_i)
                    label.attrs['for'] = str(l_i)
                    for span in label.findChildren('span'):
                        new_span = replace_template(soup, span, whole, part, jj)
                        span.replace_with(new_span)
                    l_i += 1

                hit_jjs.append(jj)
                num_in_hit += 1

                if num_in_hit >= args.jjs_per_hit:
                    # we have three followups, create the HIT
                    #remove extra divs
                    for i in range(10-1, num_in_hit-1, -1):
                        form.findChildren('div', {'class': 'question'})[i].decompose()
                    make_hit(whole, part, hit_jjs, full, str(soup), w, args.title, args.dry_run)
                    num_hits += 1
                    num_in_hit = 0
                    hit_jjs = []
                    if num_hits % 100 == 0:
                        print("num hits: %d" % num_hits)
                    #reload the template
                    prompt, full, soup, form = reload_template(whole, part)
                if num_hits >= args.max_hits:
                    break
            if num_hits >= args.max_hits:
                break
            #we're done, make a HIT with any remaining
            if num_in_hit > 0:
                #remove extra divs
                for i in range(10-1, num_in_hit-1, -1):
                    form.findChildren('div', {'class': 'question'})[i].decompose()
                make_hit(whole, part, hit_jjs, full, str(soup), w, args.title, args.dry_run)
                num_hits += 1
                if num_hits % 100 == 0:
                    print("num hits: %d" % num_hits)
        print("num hits: %d" % num_hits)

    # Remember to modify the URL above when you're publishing
    # HITs to the live marketplace.
    # Use: https://worker.mturk.com/mturk/preview?groupId=
