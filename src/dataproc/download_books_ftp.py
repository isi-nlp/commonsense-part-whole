import os, re
import ftplib
from ftplib import FTP

BASE_DIR = '/home/jamesm/commonsense-part-whole/data/gutenberg/www.gutenberg.org/robot'

def crawl(dr, n_bks):
    print(dr)
    path_digits = dr.split('/')[1:]
    #skip previously scraped books
    if path_digits[0] < '2':
        return n_bks
    elif path_digits[0] == '2':
        if len(path_digits) > 1:
            if path_digits[1] < '6':
                return n_bks
            elif path_digits[1] == '6':
                if len(path_digits) > 2:
                    if path_digits[2] < '2':
                        return n_bks
                    elif path_digits[2] == '9':
                        if len(path_digits) > 3:
                            if path_digits[3] <= '5':
                                return n_bks

    subdirs = ftp.nlst(dr)
    for sd in subdirs:
        if sd.endswith('.txt'):
            #found a book, download it
            fname = sd.split('/')[-1]
            if not os.path.exists('%s/%s' % (BASE_DIR, fname)):
                print("new book: retrieving and writing")
                dummy = []
                try:
                    book = ftp.retrlines('RETR %s' % sd, dummy.append)
                    with open('%s/%s' % (BASE_DIR, fname), 'w') as of:
                        of.write("\n".join(dummy))
                except ftplib.Error:
                    print("ERROR: %s" % sd)
                    with open('errors.txt', 'a') as ef:
                        ef.write('%s\n' % sd)
            n_bks += 1
            if n_bks % 1 == 0:
                print("%d books retrieved..." % (n_bks))
        elif 'old' not in sd and not sd.endswith('.zip') and 'images' not in sd and 'mp3' not in sd \
         and 'm4b' not in sd and 'ogg' not in sd and 'spx' not in sd:
            subsubdirs = ftp.nlst(sd)
            if not (len(subsubdirs) == 1 and subsubdirs[0] == sd):
                #recurse
                n_bks = crawl(sd, n_bks)
    return n_bks

if __name__ == "__main__":
    FTP.maxline = 16384 #some lines are long
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
