# downloads all the necessary tokenizers from the BPEMB Package
from bpemb import BPEmb
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
langs = ['ha','am','te']

for lang in langs:
    bpemb = BPEmb(lang=lang, vs=25000, dim=100)
    print('Downloaded BPEMB Tokeniser for language:', lang)

