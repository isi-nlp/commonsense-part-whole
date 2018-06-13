import csv, json, os

from io import BytesIO 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as PIL_Image                                 
import requests
from tqdm import tqdm

with open('../data/nouns/sample_50.tsv') as f:
    r = csv.reader(f, delimiter='\t')
    #header
    #next(r)
    for row in tqdm(r):
        pw = '_'.join(row[:2])
        if not os.path.isdir('../data/nouns/vg_imgs/%s' % pw):
            os.mkdir('../data/nouns/vg_imgs/%s' % pw)
        for bbox_json in row[-3:]:
            bbox = json.loads(bbox_json)
            img_id = bbox['id']
            res = requests.get('http://visualgenome.org/api/v0/images/%d' % img_id)
            try:
                im_url = res.json()['url']
                im_res = requests.get(im_url)
                img = PIL_Image.open(BytesIO(im_res.content))
                #draw bounding boxes
                plt.imshow(img)
                ax = plt.gca()
                ax.add_patch(Rectangle((bbox['px'], bbox['py']),
                                      bbox['pw'],
                                      bbox['ph'],
                                      fill=False,
                                      edgecolor='red',
                                      linewidth=2))
                ax.add_patch(Rectangle((bbox['wx'], bbox['wy']),
                                      bbox['ww'],
                                      bbox['wh'],
                                      fill=False,
                                      edgecolor='blue',
                                      linewidth=2))
                fig = plt.gcf()
                fig.savefig('../data/nouns/vg_imgs/%s/%d.png' % (pw, img_id))
                plt.clf()
            except Exception as e:
                print(str(e))
                import pdb; pdb.set_trace()
