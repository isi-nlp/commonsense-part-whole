#!/usr/bin/env python3
# boilerplate code by Jon May (jonmay@isi.edu)
import argparse
import sys
import codecs
if sys.version_info[0] == 2:
  from itertools import izip
else:
  izip = zip
from collections import defaultdict as dd
import re
import os.path
import gzip
import tempfile
import shutil
import atexit

scriptdir = os.path.dirname(os.path.abspath(__file__))


reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')


def prepfile(fh, code):
  if type(fh) is str:
    fh = open(fh, code)
  ret = gzip.open(fh.name, code if code.endswith("t") else code+"t") if fh.name.endswith(".gz") else fh
  if sys.version_info[0] == 2:
    if code.startswith('r'):
      ret = reader(fh)
    elif code.startswith('w'):
      ret = writer(fh)
    else:
      sys.stderr.write("I didn't understand code "+code+"\n")
      sys.exit(1)
  return ret

def addonoffarg(parser, arg, dest=None, default=True, help="TODO"):
  ''' add the switches --arg and --no-arg that set parser.arg to true/false, respectively'''
  group = parser.add_mutually_exclusive_group()
  dest = arg if dest is None else dest
  group.add_argument('--%s' % arg, dest=dest, action='store_true', default=default, help=help)
  group.add_argument('--no-%s' % arg, dest=dest, action='store_false', default=default, help="See --%s" % arg)

class Dep:
  """ minor processing of dependency line """
  def __init__(self, text):
    # storing complete text, head word, processed dependencies, head word pos in ngram, total n-gram count
    self.text = text
    toks = text.strip().split('\t')
    assert(len(toks) >= 4)
    self.head = toks[0]
    self.headid = -1
    self.deps = []
    for index, dep in enumerate(toks[1].split()):
      deptok = DepTok(dep)
      if deptok.index == -1:
        self.headid = index
      self.deps.append(deptok)
    assert(self.headid > -1)
    self.count = int(toks[2])
    # counts by year currently not processed

  def findmod(self, posfilter=None, depfilter=None):
    """ find the direct modifiers of the head word, optionally filtered by pos tag and/or deplabel. Return as (mod, postag, deplabel) tuple """
    for dep in self.deps:
      if dep.index == self.headid:
        if (posfilter is None or posfilter == dep.postag) and (depfilter is None or depfilter == dep.deplabel):
          yield (dep.word, dep.postag, dep.deplabel)

class DepTok:
  """ named struct for dependency token. head indexes are stored 0-based (even though the text is 1-based) for more convenient access """
  def __init__(self, text):
    # from https://docs.google.com/document/d/14PWeoTkrnKk9H8_7CfVbdvuoFZ7jYivNTkBX2Hj7qLw/edit
    # "each token format is “word/pos-tag/dep-label/head-index”. The word field can contain any
    # non-whitespace character.  The other fields can contain any non-whitespace character except for ‘/’."
    #
    # thus, parse from the back and rejoin word with '/'

    toks = text.split('/')
    assert(len(toks) >=4)
    self.index = int(toks[-1])-1
    self.deplabel = toks[-2]
    self.postag = toks[-3]
    self.word = '/'.join(toks[0:-3])


def main():
  parser = argparse.ArgumentParser(description="find heads and their modifiers in google dependency ngrams",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  addonoffarg(parser, 'debug', help="debug mode", default=False)
  parser.add_argument("--infile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input file")
  parser.add_argument("--pos", type=str, default=None, help="filter by pos type")
  parser.add_argument("--dep", type=str, default=None, help="filter by dependency type")
  parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")




  try:
    args = parser.parse_args()
  except IOError as msg:
    parser.error(str(msg))

  workdir = tempfile.mkdtemp(prefix=os.path.basename(__file__), dir=os.getenv('TMPDIR', '/tmp'))

  def cleanwork():
    shutil.rmtree(workdir, ignore_errors=True)
  if args.debug:
    print(workdir)
  else:
    atexit.register(cleanwork)


  infile = prepfile(args.infile, 'r')
  outfile = prepfile(args.outfile, 'w')


  for line in infile:
    dep = Dep(line)
    for tup in dep.findmod(posfilter=args.pos, depfilter=args.dep):
      outfile.write("{}\t{}\t{}\n".format(dep.head, tup[0], dep.count))

if __name__ == '__main__':
  main()
