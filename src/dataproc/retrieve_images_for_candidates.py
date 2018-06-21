"""
    Download example images for each part-whole relation and draw their bounding boxes
"""
import csv, json, os, shutil

from io import BytesIO 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as PIL_Image                                 
import requests
from tqdm import tqdm

with open('../../data/nouns/vg_min_3.tsv') as f:
    r = csv.reader(f, delimiter='\t')
    #header
    next(r)
    for i,row in tqdm(enumerate(r)):
        pw = '_'.join(row[:2])
        if not os.path.isdir('../../data/nouns/vg_imgs2/%s' % pw):
            print("no %s images found, downloading..." % pw)
            os.mkdir('../../data/nouns/vg_imgs3/%s' % pw)
            for bbox_json in row[-3:]:
                bbox = json.loads(bbox_json)
                img_id = bbox['id']
                if i == 0:
                    print(pw, img_id)
                res = requests.get('http://visualgenome.org/api/v0/images/%d' % img_id)
                try:
                    im_url = res.json()['url']
                    im_res = requests.get(im_url)
                    img = PIL_Image.open(BytesIO(im_res.content))
                    #draw bounding boxes
                    #plt.imshow(img)
                    #ax = plt.gca()
                    fig = plt.figure(frameon=False)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
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
                    #fig = plt.gcf()
                    ax.imshow(img)
                    plt.tick_params(labelbottom=False, labelleft=False)
                    fig.savefig('../../data/nouns/vg_imgs3/%s/%d.png' % (pw, img_id), bbox_inches='tight', pad_inches=0)
                    plt.clf()
                    plt.close(fig)
                except Exception as e:
                    print(str(e))
                    import pdb; pdb.set_trace()
        else:
            #we already have images for this pw
            shutil.copytree('../../data/nouns/vg_imgs2/%s' % pw, '../../data/nouns/vg_imgs3/%s' % pw)
