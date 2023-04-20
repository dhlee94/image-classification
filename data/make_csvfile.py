import os
import numpy as np
import argparse
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, required=True, help='Image path')
parser.add_argument('--train_ratio', type=float, default=0.7, help='ratio of train dataset(Sum of train, valid, test ratio is one)')
parser.add_argument('--valid_ratio', type=float, default=0.2, help='ratio of valid dataset(Sum of train, valid, test ratio is one)')
parser.add_argument('--test_ratio', type=float, default=0.1, help='ratio of test dataset(Sum of train, valid, test ratio is one)')
parser.add_argument('--save_path', type=str, required=True, help='Save csv path')

def write_and_save(image_lists, label_lists, outpath):
    data = {'image':image_lists, 'label':label_lists}
    df = pd.DataFrame.from_dict(data)
    df.to_csv(outpath)

def main():
    args = parser.parse_args()
    class_lists = os.listdir(args.image_path)
    for idx, classes in enumerate(class_lists):
        image_lists = [os.path.join(args.image_path, classes, images) for images in os.listdir(os.path.join(args.image_path, classes))]
        label_lists = [idx for _ in os.listdir(os.path.join(args.image_path, classes))]
        
    for idx, classes in enumerate(class_lists):
        print(f'Class : {classes} Label : {idx}')

    data_indices = np.arange(0, len(image_lists))
    train_idx = np.random.choice(len(data_indices), int(len(data_indices)*args.train_ratio), replace=False)
    valid_idx = np.random.choice(list(set(data_indices)-set(train_idx)), int(len(data_indices)*args.valid_ratio), replace=False)
    test_idx = np.random.choice(list(set(data_indices)-set(train_idx)-set(valid_idx)), int(len(data_indices)*args.test_ratio), replace=False)

    image_lists = np.array(image_lists)
    label_lists = np.array(label_lists)

    write_and_save(image_lists=image_lists[train_idx], label_lists=label_lists[train_idx], outpath=os.path.join(args.save_path, 'Train_data.csv'))
    write_and_save(image_lists=image_lists[valid_idx], label_lists=label_lists[valid_idx], outpath=os.path.join(args.save_path, 'Valid_data.csv'))
    write_and_save(image_lists=image_lists[test_idx], label_lists=label_lists[test_idx], outpath=os.path.join(args.save_path, 'Test_data.csv'))

if __name__ == '__main__':
    main()

