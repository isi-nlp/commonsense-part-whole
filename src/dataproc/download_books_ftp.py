import os, re
from ftplib import FTP

BASE_DIR = '/home/jamesm/commonsense-part-whole/data/gutenberg/www.gutenberg.org/robot'

def crawl(dr, n_bks):
    #import pdb; pdb.set_trace()
    subdirs = ftp.nlst(dr)
    for sd in subdirs:
        if sd.endswith('.txt'):
            #found a book, download it
            fname = sd.split('/')[-1]
            if not os.path.exists('%s/%s' % (BASE_DIR, fname)):
                print("new book: retrieving and writing")
                dummy = []
                book = ftp.retrlines('RETR %s' % sd, dummy.append)
                with open('%s/%s' % (BASE_DIR, fname), 'w') as of:
                    of.write("\n".join(dummy))
            n_bks += 1
            if n_bks % 1 == 0:
                print("just wrote %s. %d books retrieved..." % (fname, n_bks))
        elif 'old' not in sd and not sd.endswith('.zip'):
            subsubdirs = ftp.nlst(sd)
            if not (len(subsubdirs) == 1 and subsubdirs[0] == sd):
                #recurse
                n_bks = crawl(sd, n_bks)
    return n_bks

if __name__ == "__main__":
    ftp = FTP('ftp.gutenberg.readingroo.ms')
    ftp.login("anonymous", "james-crawling")

    #get base level dirs
    dirs = []
    for d in ftp.nlst('gutenberg'):
        if re.match('gutenberg/\d', d):
            dirs.append(d)

    #crawl directory structure
    n_bks = 0
    for d in dirs:
        n_bks = crawl(d, n_bks)
