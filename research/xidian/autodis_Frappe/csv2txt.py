import csv

csv_file_path = 'data/origin_data/train.csv'

txt_file_path = 'data/origin_data/train.txt'

with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    next(csv_reader)
    
    with open(txt_file_path, 'w', newline='') as txt_file:
        for row in csv_reader:
            txt_file.write('\t'.join(row) + '\n')