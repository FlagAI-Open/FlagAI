#!/usr/bin/env python

import torch
from PIL import Image
from os.path import basename, join
from time import time
from clip_txt import txt2vec
from clip_img import img2vec

COST = None


def inference(img, tmpl_kind_li):
  global COST
  image = Image.open(img)
  begin = time()
  image_features = img2vec(image)
  if COST is not None:
    COST += (time() - begin)

  for tmpl, kind_li in tmpl_kind_li:
    li = []
    for i in kind_li:
      li.append(tmpl % i)

    begin = time()
    text_features = txt2vec(li)
    if COST is not None:
      COST += (time() - begin)

    with torch.no_grad():
      text_probs = (image_features @ text_features.T).softmax(dim=-1)

    for kind, p in zip(kind_li, text_probs.cpu().numpy()[0].tolist()):
      p = round(p * 10000)
      if p:
        print("  %s %.1f%%" % (kind, p / 100))
  return


if __name__ == "__main__":
  from misc.config import IMG_DIR
  from glob import glob
  li = glob(join(IMG_DIR, '*.jpg'))
  # 预热，py.compile 要第一次运行才编译
  inference(li[0],
            (('a photo of %s', ('cat', 'rat', 'dog', 'man', 'woman')), ))
  COST = 0
  for i in li:
    print("\n* " + basename(i))
    inference(i, (('a photo of %s', ('cat', 'rat', 'dog', 'man', 'woman')),
                  ('一张%s的图片', ('猫', '老鼠', '狗', '男人', '女人'))))
  print('\ncost %2.fms' % (1000 * COST))
