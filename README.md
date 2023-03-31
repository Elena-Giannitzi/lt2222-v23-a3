# LT2222 V23 Assignment 3

Put any documentation here including any answers to the questions in the 
assignment on Canvas.

# Description

I used the jupyter notebook in the mltgpu terminal in order to complete and run the provided code from the forked repository. 
In order to run the code and see the results you need to run it as any oython file in the terminal (I do suggest the mltgpu server), like python a3_features.py.
Then you would need to add the right path directory in order for the code to access the Enron files, like so: /scratch/lt2222-v23/enron_sample/
In the path above, I have already included the name of the file (enron_sample) that contains the data. You can replace it with the name of the file that cotains the mails form the Enron company as you have it named in your own directory. Then, you would need the number of disired dimensions - how many dimensions would you like your dataset to turn into. The recommeneded one is 20, but you can try a different number. And also you could add the size of the test set you would like to have. The last part is optional but recommended. Usually we divide the train and test into 80% and 20% respectively. So, in the end the calling of the code should look something like this: 

python a3_features.py /scratch/lt2222-v23/enron_sample dataset 30 --test 20  (Note: in this case the dimensions of the dataset are 30)

python a3_features.py /scratch/lt2222-v23/enron_sample dataset 20 --test 20 (Note: in this case the dimensions of the dataset are 20)

# PART 4 Of Final Assignment

The Enron corpus is a large corpus of electronic (email) conversations between the employees of the Enron company, which bankrupted in 2001. The release of these emails
led to the investigation and litigation on the currently known as the "Enron scandal". To be more precise, more than 500.000 emails and email attachments were released to the public. Even though the corpus is mostly used for NLP research as a dataset, it still raises a wide variety of ethical issues concerning privacy and consent violation and, of course, the appropriate use of data. I think, even more concerning is the fact that no one of the employees gave any explicit consent for the use of this corpus and its data. It raises questions about the ethical responsibility of the people who will oversee them and research upon them. 

Another issue, that we should consider, is how the leak of the corpus could harm and violate the personal lives of the people whose names are included in the data. The leak to the public of personal and sensitive information could harm these individuals' reputation and cause psychological problems, like anxiety, distress, despair, depression, etc., which could later lead to more "dangerous" incidents.  

Overall, the Enron corpus serves as an example for research all around the world and the ethical concerns when handling "sensitive" data that were not originated for research purposes. It also highlights that other companies and organizations which collect data need to be more transparent and "serious" when collecting and using data.
