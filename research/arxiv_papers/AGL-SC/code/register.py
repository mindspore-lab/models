import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in [
    "Yelp2018",
    "Movielens1M",
    "Movielens100K",
    "AmazonElectronics",
    "100K",
]:
    dataset = dataloader.Loader(path="../data/" + world.dataset)


print("===========config================")
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print("===========end===================")

MODELS = {"mf": model.PureMF, "lgn": model.LightGCN}
