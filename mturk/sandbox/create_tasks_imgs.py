import argparse
import boto3
import bs4
import csv, html, os, random, sys, time
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm
import xmltodict

from collections import defaultdict

MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

def reload_template(whole, part):
    prompt = open('prompt_imgs.xml').read()
    full = xmltodict.parse(prompt)
    soup = bs4.BeautifulSoup(full['HTMLQuestion']['HTMLContent'], 'lxml')
    form = soup.body.form

    #replace images
    pw = '_'.join([whole.replace(' ', '_'), part.replace(' ', '_')])
    #can get filenames from local b/c they were uploaded with the same names
    try:
        new_srcs = [fn for fn in os.listdir('/home/jamesm/commonsense-part-whole/data/nouns/vg_imgs3/%s/' % pw) if fn.endswith('png')]
    except:
        print("dir not found: %s" % pw)
        return None, None, None, None

    for i,img in enumerate(form.findChildren('div')[0].findChildren('img')):
        img['src'] = 'https://s3-us-west-1.amazonaws.com/commonsense-mturk-images/%s/%s' % (pw, new_srcs[i])

    initial_sentence = soup.new_tag('h4')
    #basic logic for a vs. an
    #det = 'an' if part[0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
    initial_sentence.string = form.findChildren('div')[0].h4.text.replace('WHOLE', whole).replace('PART', part)
    form.findChildren('div')[0].h4.replace_with(initial_sentence)
    return prompt, full, soup, form

def make_hit(whole, part, jjs, full, form_str, w, title, dryrun):
    form = "<![CDATA[\n" + form_str + "\n\n]]>" #dumb hack to make the unparse() output valid
    full['HTMLQuestion']['HTMLContent'] = form
    prompt = html.unescape(xmltodict.unparse(full))
    title = 'Common sense visual reasoning %s' % title #TODO: remove the number
    if not dryrun:
        new_hit = mturk.create_hit(
                                   Title = title,
                                   Description = 'View some images and choose the option that best describes the possibility of some statements about an object.',
                                   Keywords = 'images, quick, question answering',
                                   Reward = '0.01',
                                   MaxAssignments = 3,
                                   LifetimeInSeconds = 60*60*24, #1 day
                                   AssignmentDurationInSeconds = 60*5, #5 minutes
                                   AutoApprovalDelayInSeconds = 60*60*4, #4 hours
                                   Question = prompt
        )
        w.writerow([title, "https://workersandbox.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'], new_hit['HIT']['HITId'], whole, part, ';'.join(jjs)])
    else:
        w.writerow([title, "https://workersandbox.mturk.com/mturk/preview?groupId=1", '2', whole, part, ';'.join(jjs)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate_file', type=str, help="Filename of candidate triples to read from")
    parser.add_argument('title', type=str, help="The part of the task title that goes after 'Common sense visual reasoning'")
    parser.add_argument('--max-hits', default=10, dest='max_hits', type=int, help="maximum number of HITs to make")
    parser.add_argument('--max-pws', default=100000, dest='max_pws', type=int, help="maximum number of part-wholes to make HITs from")
    parser.add_argument('--dry-run', dest='dry_run', action='store_const', const=True, help='flag to not actually make HITs')
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
    s3 = boto3.client('s3',
                      aws_access_key_id = iam_access,
                      aws_secret_access_key = iam_secret,
                      region_name='us-east-1',
                      )
    #########################################

    print("reading examples")
    examples = [row for row in csv.reader(open('/home/jamesm/commonsense-part-whole/data/candidates/%s' % (args.candidate_file)))]
    lem = WordNetLemmatizer()
    #parse examples into (whole, part) : {jjs} lookup
    pw2jjs_raw = defaultdict(set)
    for whole, part, jj in examples:
        whole_lem = lem.lemmatize(whole.replace(' ', '_')).replace('_', ' ')
        part_lem = lem.lemmatize(part.replace(' ', '_')).replace('_', ' ')
        pw2jjs_raw[(whole_lem, part_lem)].add(jj)

    #just lemmatize everything
    #pw2jjs = defaultdict(set)
    #for whole, part, jj in examples:
    #    whole_lem = lem.lemmatize(whole.replace(' ', '_'))
    #    part_lem = lem.lemmatize(part.replace(' ', '_'))
    #    if (whole_lem, part_lem) in pw2jjs_raw:
    #        pw2jjs[(whole_lem, part_lem)].update(pw2jjs_raw[(whole, part)])
    #    elif (whole, part_lem) in pw2jjs_raw:
    #        pw2jjs[(whole, part_lem)].update(pw2jjs_raw[(whole, part)])
    #    elif (whole_lem, part) in pw2jjs_raw:
    #        pw2jjs[(whole_lem, part)].update(pw2jjs_raw[(whole, part)])
    #    else:
    #        pw2jjs[(whole, part)].update(pw2jjs_raw[(whole, part)])
    pw2jjs = pw2jjs_raw

    wholes = set([w for w,_ in pw2jjs.keys()])
    parts = set([p for _,p in pw2jjs.keys()])
    num_triples = sum([len(jjs) for jjs in pw2jjs.values()])
    print("parts: %d, wholes: %d, triples: %d" % (len(parts), len(wholes), num_triples))

    with open('/home/jamesm/commonsense-part-whole/mturk/sandbox/hit_batches/batch_%d.csv' % round(time.time()*1000), 'w') as of:
        w = csv.writer(of)
        w.writerow(['title', 'url', 'HITId', 'whole', 'part', 'jj'])

        print("making HITs...")
        #iterate in random order
        num_hits = 0
        for num_ex,((whole, part), jjs) in tqdm(enumerate(sorted(pw2jjs.items(), key=lambda x: random.random()))):
            if num_hits >= args.max_hits:
                break
            if whole == part or '.' in whole:
                print("whole, part: %s, %s. skipping..." % (whole, part))
                continue
            if len(jjs) == 0: continue
 
            whole = whole.replace('_', ' ')
            part = part.replace('_', ' ')
            prompt, full, soup, form = reload_template(whole, part)
            if prompt is None:
                continue

            #TODO: remove this
            if num_ex > args.max_pws:
                break

            num_in_hit = 0
            hit_jjs = []
            for i, jj in enumerate(jjs):
                #replace followup sentence
                followup_sent = soup.new_tag('h4')
                div = form.findChildren('div')[num_in_hit+1]
                followup_sent.string = div.h4.text.replace('WHOLE', whole).replace('ADJECTIVE', jj)
                div.h4.replace_with(followup_sent)

                #also replace radio button labels
                for label in div.findChildren('label'):
                    old_str = label.text
                    new_label = soup.new_tag('label')
                    new_label.string = old_str.replace('WHOLE', whole).replace('PART', part).replace('ADJECTIVE', jj)
                    label.replace_with(new_label)

                hit_jjs.append(jj)
                num_in_hit += 1

                if num_in_hit >= 3:
                    # we have three followups, create the HIT
                    make_hit(whole, part, hit_jjs, full, str(soup), w, args.title, args.dry_run)
                    num_hits += 1
                    num_in_hit = 0
                    hit_jjs = []
                    if num_hits % 100 == 0:
                        print("num hits: %d" % num_hits)
                    #reload the template
                    prompt, full, soup, form = reload_template(whole, part)
                    if prompt is None:
                        continue
                if num_hits >= args.max_hits:
                    break
            if num_hits >= args.max_hits:
                break
            #we're done, make a HIT with any remaining
            if num_in_hit > 0:
                #remove extra divs
                for i in range(3, num_in_hit, -1):
                    form.findChildren('div')[i].decompose()
                make_hit(whole, part, hit_jjs, full, str(soup), w, args.title, args.dry_run)
                num_hits += 1
                if num_hits % 100 == 0:
                    print("num hits: %d" % num_hits)

# Remember to modify the URL above when you're publishing
# HITs to the live marketplace.
# Use: https://worker.mturk.com/mturk/preview?groupId=
