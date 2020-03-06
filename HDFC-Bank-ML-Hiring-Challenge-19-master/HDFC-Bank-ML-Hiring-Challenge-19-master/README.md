# HDFC-Bank-ML-Hiring-Challenge-19
##Problem Statement:-

Use case A:


The task is to build a banking behavioral scorecard model for internal customers through a user's liability account and predict the credit risk. These are low transacting customers. The definition that is used for the target variable is every 30+ or X+ days delinquent twice in forward 12 months.

The implementations of the model are as follows:

    1)Pre-approving customers for cross-selling other products of a bank
    2)Decide the mode of treatment of customers and specify it as an input for credit underwriting
    (Note: You can decide the customer treatment through a graded risk profile of customers.)

What is a Banking behavioral scorecard?

Banking behavioral scorecard is a model that is maintained for a customer based on his liability transactions. Liability transactions are transactions that are transacted by an internal customer of a bank. Internal customers of a bank are the customers who have a savings account (SA) with the bank. 

How can a customer repay a loan?

Customer pays the loan in equal monthly installments (EMIs). Loans get paid through post-dated cheques. Customer also has an option to pay through the Electronic Clearing System (ECS) technique or standing instructions to debit the user's HDFC Bank account with the EMI amount.

What is the meaning of customer risk profile?

Customer risk profile means the probability of the customer defaulting on his EMI payment.

Use Case B: 

    1)Describe your approach that you have used to build the model and your reasons for the same
    2)Describe model segmentation (if any)
    3)Describe how can your model help banks

## Data description

     1.Total number of variables: 2395
     2.Col1 Variable: ID 
     3.Col2 Variable: Target Variable 
     4.Col_3 To Col_2397 are anonymized features

The dataset consists of the following types of files:​​

     Train.csv: 17521 * 2395

     Test.csv:  20442 * 2394

   Sample_submission.csv: Sample format of expected predictions in the .csv format

The sources of the data that are used for the model are as follows:

    1 Banking information of the customer.
    2 Customer savings account or Current Account transaction information for a certain period
    3 Create additional variables or use any additional alternate sources of information for building the model
   
# Submission format

Use Case A: Submit the submissions in a .csv format according to the format that is described in the Sample submission file. 

     Col1,Col2
     RIGD58ZWD,0
     RIH660YDS,1
     RIH660Q96,0
     RIYDO15W1,1

## Evaluation criteria

The evaluation criteria for the two use cases are defined as follows:

Use Case A: Submissions are evaluated based on the F1_score technique. It uses the predicted probability and the observed target with the weighted average. 

# F1 SCORE:- 86.68
# Rank 14 
