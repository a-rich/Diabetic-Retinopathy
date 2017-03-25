import time
import itertools
from PIL import Image
from os import listdir
from multiprocessing import Pool, Lock


def processImage(args):
  image = Image.open(source + "/" + args[1])
  dim = args[0].split('x')
  image = image.resize((int(dim[0]), int(dim[1])), Image.ANTIALIAS)
  image.save('./sample/resized/' + args[0] + '/' + args[1])

if __name__ == "__main__":
  source = "./sample/originals"
  new_size = '4096x4096'
  pool = Pool()
  start = time.time()
  pool.map(processImage, zip(itertools.repeat(new_size), listdir(source)))
  print(time.time() - start)
