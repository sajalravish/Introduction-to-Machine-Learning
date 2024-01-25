Name: Sajal Ravish
Student ID: 3037732765

How to reproduce my results:
1. Ensure that you have downloaded all necessary libraries (numpy, sklearn, matplotlib, and pandas)
2. Navigate to your terminal. First cd into the "scripts" folder. Then, run "python svm.py" or "python3 svm.py"
   This will allow you to run my code for ALL homework problems 3-7. Please be aware that this can take up to
   ~22 minutes (may vary depending on your computer) because it takes time to compute the hyperparameter tuning
   and k-fold cross-validation from scratch (questions 5 and 6, respectively).
3. If you don't want to recalculate the optimal values of the hyperparameter C when running svm.py to save time
   (in other words, you just want to produce the .csv files for Kaggle submission), then COMMENT OUT lines
   158-191 and also COMMENT OUT lines 195-238. In this way, you can skip over the step of recalculating the optimal
   C value and can skip straight to producing the .csv files. This also saves a significant amount of time in 
   running svm.py, cutting down the time to approximately 9 minutes.

Note on file structure:
Within the scripts folder, I have 8 files I contributed to:
- svm.py: This file contains the majority of my code. It is where I trained my SVM models on the MNIST and Spam
          datasets and solved the majority of problems 3-7 on Homework 1.
- training-results.txt: This file contains the training and validation accuracies of my SVM models on the MNIST and Spam
          datasets (from problem 4 on Homework 1)
- c-values.txt: This file contains the training and validation accuracies of 15 potential C-values, as well as information
          on which C-value was ultimately the most optimal for my SVM model on the MNIST dataset (from problem 5 on Homework 1)
- ck-values.txt: This file contains the training and validation accuracies of 14 potential C-values, as well as information
          on which C-value was ultimately the most optimal for my SVM model on the Spam dataset (from problem 6 on Homework 1)
- mnist-submission.csv: This file contains the predictions for the test sets for the MNIST dataset, and the predications
          are formatted such that you can submit the file to Kaggle. Running svm.py will create this file.
- spam-submission.csv: This file contains the predictions for the test sets for the Spam dataset, and the predications
          are formatted such that you can submit the file to Kaggle. Running svm.py will create this file.
- featurize.py: I edited this file to add new features to improve the classification abilities of my Spam SVM model.
- problem-2e.py: This file contains a script that you can run to solve Homework 1 Problem 2(e).