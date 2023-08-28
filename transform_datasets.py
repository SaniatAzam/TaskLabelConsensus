'''
By using this tool you agree to acknowledge the original datasets and to check their terms and conditions.
Some data providers may require authentication, filling forms, etc.
We include a link to the original source of each dataset in our repository, please cite the appropriate sources in your work.
'''

import os
import pandas as pd
import re
import platform
import csv
from itertools import islice

def recursive_walk(folder, delimeter):
    for folderName, subfolders, filenames in os.walk(folder):
        dest = os.path.join(os.getcwd(), folderName.split('data-raw')[0]) + 'transformed_dataset.csv'
        if folderName == 'binary-classification' + delimeter + 'Crowdsourced Amazon Sentiment' + delimeter + 'data-raw':
            processCrowdsourcedAmazonSentimentDataset(filenames, folderName)
        elif folderName == 'binary-classification' + delimeter + 'Sentiment popularity - AMT' + delimeter + 'data-raw':
            processSentiment(filenames, folderName, dest)
        elif folderName == 'multi-class-classification' + delimeter + 'Weather Sentiment - AMT' + delimeter + 'data-raw':
            processSentiment(filenames, folderName, dest)
        else:
            for subfolder in subfolders:
                recursive_walk(subfolder, delimeter)




def processGoldAndLabelFiles(filenames, folderName, dest, dataset):
    if dataset == 'Toloka':
        filenames = sorted(filenames, reverse=True)
    else:
        filenames = sorted(filenames)
    gt_dict = {}
    columns = ['workerID', 'taskID', 'response', 'goldLabel', 'taskContent']
    df = pd.DataFrame([], columns=columns)
    for file in filenames:
        if file == 'gold.txt' or file == 'golden_labels.tsv':
            with open(os.path.join(folderName, file)) as fp:
                rows = ( line.split('\t') for line in fp )
                gt_dict = { row[0]:row[1] for row in rows }
        if file == 'labels.txt' or file == 'crowd_labels.tsv':
            with open(os.path.join(folderName, file)) as fp:
                for line in fp:
                    label_list = re.split(r'\t+', line)
                    goldLabel = None
                    if gt_dict.get(label_list[1]) != None:
                        goldLabel = gt_dict.get(label_list[1]).strip()
                    row = [label_list[0], label_list[1], label_list[2].strip(), goldLabel, None]
                    dfRow = pd.DataFrame([row], columns=columns)
                    df = df.append(dfRow)
    df.to_csv(dest, index=None, header=True)

def processWithSeperateText(filenames, folderName, file1, file2, dest, isTextExist):
    if file2 == 'rte1.tsv':
        filenames = sorted(filenames)
    else:
        filenames = sorted(filenames, reverse=True)
    columns = ['workerID', 'taskID', 'response', 'goldLabel', 'taskContent']
    df = pd.DataFrame([], columns=columns)
    for file in filenames:
        with open(os.path.join(folderName, file)) as fp:
            next(fp)
            for line in fp:
                label_list = re.split(r'\t+', line)
                if file == file1:
                    row = [label_list[1], label_list[2], label_list[3], label_list[4].strip(), None]
                    dfRow = pd.DataFrame([row], columns=columns)
                    df = df.append(dfRow)
                if file == file2 and isTextExist:
                    if file2 == 'rte1.tsv':
                        df.loc[df['taskID'] == label_list[0], ['taskContent']] = label_list[3]
                    else:
                        df.loc[df['taskID'] == label_list[0], ['taskContent']] = label_list[4]
    df.to_csv(dest, index=None, header=True)



def processSentiment(filenames, folderName, dest):
    columns = ['workerID', 'taskID', 'response', 'goldLabel', 'taskContent', 'timeSpent']
    df = pd.DataFrame([], columns=columns)
    for file in sorted(filenames):
        with open(os.path.join(folderName, file)) as fp:
            for line in fp:
                label_list = re.split(r',', line)
                row = [label_list[0], label_list[1], label_list[2], label_list[3], None, label_list[4]]
                dfRow = pd.DataFrame([row], columns=columns)
                # df = df.append(dfRow)
                df = pd.concat([df, dfRow], ignore_index=True)
    df.to_csv(dest, index=None, header=True)
    




def processCrowdsourcedAmazonSentimentDataset(filenames, folderName):
    columns = ['workerID', 'taskID', 'response', 'goldLabel', 'taskContent']
    for file in sorted(filenames):
        for i in range(2):
            df = pd.DataFrame([], columns=columns)
            with open(os.path.join(folderName, file), encoding="utf8") as csv_file:
                filePostfix = 'is_book'
                responseIndex = 14
                goldenIndex = 23
                if (i == 1):
                    filePostfix = 'is_negative'
                    responseIndex = 15
                    goldenIndex = 25
                dest = os.path.join(os.getcwd(),
                                    folderName.split('data-raw')[0]) + 'transformed_dataset_' + filePostfix + '.csv'
                next(csv_file)
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    row = [line[9], line[0], line[responseIndex], line[goldenIndex], line[27]]
                    dfRow = pd.DataFrame([row], columns=columns)
                    # df = df.append(dfRow)
                    df = pd.concat([df, dfRow], ignore_index=True)
                df.to_csv(dest, index=None, header=True)




if __name__ == '__main__':
    # get the current path
    path = '.'

    # define the delimeter based on the operating system
    delimeter = '/'
    if platform.system() == 'Windows':
        delimeter = '\\'

    # walk through the directories and transform all the datasets
    recursive_walk(path, delimeter)




