# Data processing scripts

## 0. Download the data you need.

- [Google n-grams](http://storage.googleapis.com/books/ngrams/books/datasetsv2.html) - scroll down to "English fiction" and download all 1-grams files.

- [Brysbaert concreteness ratings](http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt)

- [Visual Genome](http://visualgenome.org/api/v0/api_home.html) Download relationships, version 1.4.

- [Conceptnet 5.5](https://github.com/commonsense/conceptnet5/wiki/Downloads) 

- [SemEval 2010 Task 8](http://www.kozareva.com/downloads.html)

- [Wikidata](https://dumps.wikimedia.org/wikidatawiki/entities/) -> 20180612 -> wikidata-20180612-truthy-BETA.nt.bz2. But replace 20180612 with whatever the most recent date is.

- [Preprocessed dependency n-grams](https://drive.google.com/drive/folders/10Nbp2t6wzBDR0_8XqpGcBiQV_f15z2fl) (google drive link)

- [SNLI](https://nlp.stanford.edu/projects/snli/) (1.0) and [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) (0.9)

- [MSCOCO captions](http://cocodataset.org/#download) (2017 Train/val annotations)

- [GloVe](https://nlp.stanford.edu/projects/glove/) Common crawl - 840B tokens, 300d vectors
  
## 1. Prepare nouns from each dataset

- To prepare wikidata, run `process_wikidata.sh`, then `load_wikidata_pws.py`

- To prepare conceptnet, run `filter_conceptnet.py`

- To prepare semeval, run `semeval_pws.py`

- To prepare Google unigram noun counts for frequency filtering, run `noun_counts.py` (this will use all your cores)

- **Visual**: To gather part-wholes from visual genome, limited to those instances occurring in at least 3 distinct images, run `vg_candidates.py`

- **Non-visual:** Then, to use all these sources to generate part-whole candidates, run `python part_whole_candidates.py no-pwkb fiction-1950 ../../data/nouns/allnonvis.csv --noun-freq 50000 --union-concrete --no-concrete-filter`

  - This will gather all part-wholes from wikidata, conceptnet, and semeval, and then filter parts and wholes by noun frequency in Google n-grams, where the counts are restricted to those from fiction books since 1950 only. 

  - This script has some outdated parts but should work as advertised for this command.

## 2. Attach adjectives

- **Visual**: Run `python adj_candidates.py ../../data/nouns/vg_candidates.csv ../../data/adjectives/vis.csv`

- **Non-visual**: Run `python adj_candidates.py ../../data/nouns/allnonvis.csv ../../data/adjectives/nonvis.csv`

This will find the 5 most common adjectives attaching to the whole from Google n-grams and visual genome, filter out those not seen applying to the part, limit superfluous colors, and write the output.

For reverse direction data, call `reverse_adj_candidates.py` with similar arguments.

## 3. Mechanical Turk

To launch the main annotation task, these are the main steps:

- Retrieve 3 images for each part-whole from VG api. We need to store them on S3 as hosting for the AMT task. This can be done with `retrieve_images_for_candidates.py` - you will probably need to update the input/output locations

- Use `upload_images_to_s3.py` to upload.

- Use `../../mturk/create_tasks_imgs.py`, which uses `prompt_imgs.xml` as a template, to create HITs for the visual task. Make sure the Reward and MaxAssignments are set how you want them in `make_hit`. Set the `--max-hits` input arg to how many tasks to make at once. 

## 4. MTurk post-processing

- Use `compile_annotations.py` to throw out triples with nonsense annotations, select the label from annotations, and split into train/dev/test with given percentages, randomly by triple

- Similarly, use `compile_split_by{pw,pj}.py` to do the same but split by part-whole or part-adjective, respectively.

- Use `plot_distr.sh` to generate the label distribution plot.

## 5. Embeddings

- First, `cat` together all the train/dev/test files into one big file in `data/annotated/` called `full_all.csv`

- Run `elmo_vocab.sh` to create a vocabulary file and get ELMo embeddings for each individual word. (requires `allennlp`)

- Run `{glove,conceptnet}_embeds.py` to get GloVe /ConceptNet vectors for each word in the vocab.

## 6. Mining sentences (

### 6a. Whole-adjective, part-adjective sentences

This case is easy and common enough to just run spaCy to validate attachment in sentences where the "[adjective] [noun]" (or "[noun] [adjective]") appears in sequence just like that. We use Project Gutenberg, Gigaword, Image captions from MSCOCO, and SNLI/MultiNLI sentences as candidate sentences.

- Get the gutenberg data from an FTP mirror using `download_books_ftp.py`. This will probably take a day or so. 

- Use `find_sentences_{gutenberg,gigaword,nli,captions}.py` to gather sentences from each dataset. Each one, besides captions, takes cpu count as an input argument. 

### 6b. Part-whole sentences

Here, we have to use syntax to do some filtering down to sentences with patterns that reflect the part-whole relationship. Thus we are limited to using parsed corpora - Annotated Gigaword and a parsed version of the UMBC Webbased corpus.

- First, run `preproc.sh` to unzip and convert to `tgrep` format (uses all cores).

- Run `mine_{of,poss}_sentences.sh` to filter to sentences that have a syntactic "[noun] of [det] [noun]" or "[noun] 's [noun]" pattern, respectively. 

- Run `postproc_pw_sentences.py` to select from the filtered sentences those that have our desired part-whole nouns in them. 

- Run `compile_pw_sents.py` to combine the outputs from both Gigaword and UMBC.

### 6c. Combine

- Use `select_found_sentences.py` to combine the whole-adjective, part-whole, and part-adjective sentences into pairs of at most 5 per triple. 
  - This will have to be slightly modified to just return the whole-adjective premise and a constructed hypothesis to re-create the truly entailment snli-style data, I think.

- Run `split_and_label_snli_style.py` do write csv's from the sentence pairs, split into train/dev/test

- Run `tag_sentences.py` to write sentence pairs in JSON lines format, with extra pre-processing as needed for the DIIN model (parts of speech, token match features)

