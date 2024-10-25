import numpy as np
import os,sys
np.random.seed(64) 

def rand_sample(dataset_path, sample_fraction = 0.01):
    input_path = dataset_path + 'cleaned/' + 'train.txt'
    output_path = dataset_path + 'cleaned/' + 'sample.txt'

    with open(input_path,'r') as f:
        data = f.readlines()

    data = np.array(data)
    idx = np.sort(np.random.choice(len(data),int(len(data)*sample_fraction),replace=False))
    sample_data = data[idx]

    with open(output_path,'w') as f:
        for line in sample_data:
            f.write(line)

    print('sampled dataset now exits in ',output_path)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        DATASET_PATH = sys.argv[1]
        rand_sample(DATASET_PATH)
    else:
        DATASET_PATH = sys.argv[1]
        SAMPLE_FRACTION = sys.argv[2]
        rand_sample(DATASET_PATH,SAMPLE_FRACTION)