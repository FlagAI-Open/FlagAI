import webdataset as wds
import torch 
from itertools import islice
from PIL import Image
import io
# url = "/share/projset/laion400m/laion400m-full-release/img_data/laion400m-dat-release/12740.tar"

# dataset = wds.WebDataset(url)

# for sample in islice(dataset, 0, 3):
#     for key, value in sample.items():
#         if key == "jpg":
#             # print(repr(value))
#             image_data = repr(value)
#             image = Image.open(io.BytesIO(image_data))
#             image.show()
#             break
#         # print(key, repr(value)[:50])
#     print()
# import pdb;pdb.set_trace()
# dataset = wds.WebDataset(url).shuffle(1000).decode("torchrgb").to_tuple("jpg;png", "json")

# dataloader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=1)

# for inputs, outputs in dataloader:
#     print(inputs, outputs)
#     break

import pandas as pd 
from glob import glob 
import requests
path = "/home/yanzhaodong/anhforth/data/train-00000-of-00001-6f24a7497df494ae.parquet"

def download_from_path(path, output_dir='/home/yanzhaodong/anhforth/data/images/'):
    df = pd.read_parquet(path)
    df = df.loc[df['TEXT'].str.contains('flower') |  df['TEXT'].str.contains('plant') | 
                    df['TEXT'].str.contains('vegetation') | df['TEXT'].str.contains('garden') | df['TEXT'].str.contains('floral')]
    df = df.sort_values(by=["AESTHETIC_SCORE"], ascending=False)
    df = df[:2000]
    urls = list(df["URL"])
    from tqdm import trange
    # url = "https://us.123rf.com/450wm/grandfailure/grandfailure1601/grandfailure160100013/50661747-woman-in-flower-fields-next-to-red-castle-and-mountain-illustration-painting.jpg?ver=6"
    for i in trange(len(urls)):
        url = urls[i]
        try:
            response = requests.get(url, verify=False, timeout=10)
        except OSError or urllib3.exceptions.NewConnectionError:
            pass
        if response.status_code:
            fp = open(output_dir+str(i)+'.png', 'wb')
            fp.write(response.content)
            fp.close()
# download_from_path(path)
import pdb; pdb.set_trace()
glob('/home/yanzhaodong/anhforth/data/images/*.png')

# files = glob("/share/projset/laion400m/laion400m-full-release/img_data/laion400m-dat-release/*.parquet")
# from tqdm import tqdm 
# for f in tqdm(files):
#     df = pd.read_parquet(f)
#     df = df.loc[df['caption'].str.contains('flower')]

# import pdb;pdb.set_trace()
# df = pd.read_parquet("/share/projset/laion400m/laion400m-full-release/img_data/laion400m-dat-release/12740.parquet")
# df = df.loc[df['caption'].str.contains('flower')]
# import pdb;pdb.set_trace()