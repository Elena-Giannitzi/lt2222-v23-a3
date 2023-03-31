# LT2222 V23 Assignment 3

Put any documentation here including any answers to the questions in the 
assignment on Canvas.

# PART 4 Of Final Assignment



# PART 5 of Final Assignemnt

I used the jupyter notebook in the mltgpu terminal in order to complete and run the provided code from the forked repository. 
In order to run the code and see the results you need to run it as any oython file in the terminal (I do suggest the mltgpu server), like python a3_features.py.
Then you would need to add the right path directory in order for the code to access the Enron files, like so: /scratch/lt2222-v23/enron_sample/
In the path above, I have already included the name of the file (enron_sample) that contains the data. You can replace it with the name of the file that cotains the mails form the Enron company as you have it named in your own directory. Then, you would need the number of disired dimensions - how many dimensions would you like your dataset to turn into. The recommeneded one is 20, but you can try a different number. And also you could add the size of the test set you would like to have. The last part is optional but recommended. Usually we divide the train and test into 80% and 20% respectively. So, in the end the calling of the code should look something like this: 
# python a3_features.py /scratch/lt2222-v23/enron_sample dataset 30 --test 20  (Note: in this case the dimensions of the dataset are 30)
# python a3_features.py /scratch/lt2222-v23/enron_sample dataset 20 --test 20 (Note: in this case the dimensions of the dataset are 20)

