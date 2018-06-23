import boto3
import bs4
import csv, html, sys
import xmltodict

MTURK_URL = 'https://mturk-requester.us-east-1.amazonaws.com'

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

prompt = open('prompt_qualifier_causal.xml').read()
full = xmltodict.parse(prompt)
soup = bs4.BeautifulSoup(full['HTMLQuestion']['HTMLContent'], 'lxml')
form = soup.body.form

#make it possible to check radio box by clicking on text
l_i = 0
for div in form.findChildren('div', {'class', 'question'}):
    for label in div.findChildren('label'):
        label.findPreviousSibling().attrs['id'] = str(l_i)
        label.attrs['for'] = str(l_i)
        l_i += 1

#transform back into html
form = "<![CDATA[\n" + str(soup) + "\n\n]]>" #dumb hack to make the unparse() output valid
full['HTMLQuestion']['HTMLContent'] = form
prompt = html.unescape(xmltodict.unparse(full))

title = "Common sense visual reasoning - 9 questions"
qualifier_hit = mturk.create_hit(
                                 Title = title,
                                 Description = 'View some images and choose the option that best describes the possibility of some statements about an object.',
                                 Keywords = 'images, quick, question answering, qualifier',
                                 Reward = '0.10',
                                 MaxAssignments = 300,
                                 LifetimeInSeconds = 60*60*24*5, #5 days
                                 AssignmentDurationInSeconds = 60*30, #30 minutes
                                 AutoApprovalDelayInSeconds = 60*60*4, #4 hours
                                 Question = prompt,
                                 QualificationRequirements = [
                                     {
                                         'QualificationTypeId': '00000000000000000040', #number of HITs approved
                                         'Comparator': 'GreaterThanOrEqualTo',
                                         'IntegerValues': [
                                             500,
                                         ],
                                     },
                                     {
                                         'QualificationTypeId': '000000000000000000L0', #percentage assignments approved
                                         'Comparator': 'GreaterThanOrEqualTo',
                                         'IntegerValues': [
                                             98,
                                         ]
                                     }
                                     ]
                                 )
pw2jjs = {('leg', 'knee'): set(['bent']),
          ('boat', 'top'): set(['open']),
          ('woman', 'shoulder'): set(['beautiful']),
          ('jacket', 'collar'): set(['brown']),
          ('woman', 'chain'): set(['beautiful']),
          ('boat', 'letter'): set(['long', 'wooden']),
          ('bed', 'pillow'): set(['wooden']),
          ('road', 'dirt'): set(['open'])}
with open('hit_batches/qualifier_final.csv', 'w') as of:
    w = csv.writer(of)
    w.writerow(['title', 'url', 'HITid', 'whole', 'part', 'jj'])
    
    for (whole, part), jjs in pw2jjs.items():
        w.writerow([title, qualifier_hit['HIT']['HITGroupId'], qualifier_hit['HIT']['HITId'], whole, part, ';'.join(jjs)])
