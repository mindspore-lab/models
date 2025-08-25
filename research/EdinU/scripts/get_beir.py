# download retrieval datasets from BEIR
from beir import util
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# get some arguana
dataset = 'arguana'
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
data_path = util.download_and_unzip(url, '../data')
print("Dataset downloaded here: {}".format(data_path))

# get quora
dataset = 'quora'
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
data_path = util.download_and_unzip(url, '../data')
print("Dataset downloaded here: {}".format(data_path))