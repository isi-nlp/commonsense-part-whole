import sys
import requests
from PIL import Image as PIL_Image                                 
from io import BytesIO 
import matplotlib.pyplot as plt

if __name__ == "__main__":
    res = requests.get('http://visualgenome.org/api/v0/images/%s' % sys.argv[1])
    im_res = requests.get(res.json()['url'])
    img = PIL_Image.open(BytesIO(im_res.content))
    plt.imshow(img)
    plt.show()
