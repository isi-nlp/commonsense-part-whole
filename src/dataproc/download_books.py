from io import BytesIO
import os, time, zipfile
import bs4
import requests

BASE_DIR = '/home/jamesm/commonsense-part-whole/data/gutenberg/www.gutenberg.org/robot'
for f in os.listdir(BASE_DIR):
    html = open('%s/%s' % (BASE_DIR, f)).read()
    soup = bs4.BeautifulSoup(html, 'lxml')
    for i,p in enumerate(soup.body.findChildren('p')):
        if i >= len(soup.body.findChildren('p')) - 1: break
        link = p.findChildren('a')[0].attrs['href']
        res = requests.get(link)
        zf = zipfile.ZipFile(BytesIO(res.content))
        fname = zf.namelist()[0]
        print("writing %s" % fname)
        with open(fname, 'w') as of:
            of.write(str(zf.read(fname)))
        #respect robots.txt...
        time.sleep(5)
        
