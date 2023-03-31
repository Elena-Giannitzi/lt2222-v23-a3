#Importig the necessary librariries

import os #i'm using pathlib instead
import sys
import argparse 
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
import re
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
# Whatever other imports you need

def sender(base_dir) :
    emailSender = []
    listOfSenders = []

    for sender_path in base_dir.iterdir():
        if sender_path.is_dir():
            emailSender.append(str(sender_path.name))
            for mail in sender_path.iterdir():
                if mail.is_file():
                    with mail.open("r", encoding='utf-8') as m:
                        for line in m:
                            text = line.replace('\n', ' ')
    #                         print(text)
                            if text.startswith("X-From"):
                                text = text.removeprefix("X-From: ")
                                text = re.sub('</[^>]+>', '', text)
                                listOfSenders.append(text.strip())
                                

    return listOfSenders

def mail(base_dir):
    emails = []
    for sender_path in base_dir.iterdir():
        if sender_path.is_dir():
            for mail in sender_path.iterdir():
                if mail.is_file():
                    with mail.open("r", encoding='utf-8') as m:
                        found_message = False
                        email_body = ""
                        for line in m:
                            if "X-FileName" in line:
                                if "-----Original Message-----" in line:
                                    break
                                found_message = True
                            elif found_message:
                                email_body += line
                        emails.append(email_body)
    return emails 

direc = "/scratch/lt2222-v23/enron_sample/"
senderX = sender(Path(direc))
mailX = mail(Path("/scratch/lt2222-v23/enron_sample/"))
# print(senderX)

vectorizer = CountVectorizer(lowercase=False)
X_sender = vectorizer.fit_transform(senderX)
X_mail = vectorizer.transform(mailX)
y = ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()
    
    print("Reading {}...".format(args.inputdir)) #choose a path directory
    # Do what you need to read the documents here.
    senderX = sender(Path(args.inputdir))
    mailX = mail(Path(args.inputdir))
    
    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    X_sender = vectorizer.transform(senderX)
    X_mail = vectorizer.transform(mailX)
    X = hstack([X_sender, X_mail])
    y = ...
    df = pd.DataFrame(X.toarray())
    df1 = pd.concat([df,pd.Series(y,name='label')],axis=1)

    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.
    trainX, testX, trainY, testY = train_test_split(df1.iloc[:,:-1],df1.iloc[:,-1], test_size=args.testsize/100, random_state=42)
    train = pd.concat([trainX,trainY],axis=1)
    test = pd.concat([testX,testY],axis=1)
    train.to_csv('Train.csv', index=False)
    test.to_csv('Test.csv', index=False)

    print("Done!")
    print("Please check your folder for the csv files. They should be in there under the names 'Train.csv''Test.csv'.")