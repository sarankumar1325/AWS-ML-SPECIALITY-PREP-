import { Question } from '../types';

export const EXAM_DURATION_MINUTES = 130;
export const PASSING_SCORE_PERCENTAGE = 72;

export const questions: Question[] = [
  {
    id: 'q1',
    text: "A large mobile network operating company is building a machine learning model to predict customers who are likely to unsubscribe from the service. The company plans to offer an incentive for these customers as the cost of churn is far greater than the cost of the incentive. The model produces a confusion matrix after evaluating on a test dataset of 100 customers. Based on the model evaluation results, why is this a viable model for production?",
    type: 'MCQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'The model is 86% accurate and the cost incurred by the company as a result of false negatives is less than the false positives.' },
      { id: 'b', text: 'The precision of the model is 86%, which is less than the accuracy of the model.' },
      { id: 'c', text: 'The model is 86% accurate and the cost incurred by the company as a result of false positives is less than the false negatives.' },
      { id: 'd', text: 'The precision of the model is 86%, which is greater than the accuracy of the model.' }
    ],
    correctAnswerIds: ['c'],
    explanation: "For churn prediction, false negatives (missing customers who will churn) are more costly than false positives (offering incentives to customers who wouldn't churn). The 86% accuracy combined with the lower cost of false positives makes this model viable for production."
  },
  {
    id: 'q2',
    text: "A Machine Learning Specialist is designing a system for improving sales for a company. The objective is to use the large amount of information the company has on users' behavior and product preferences to predict which products users would like based on the users' similarity to other users. What should the Specialist do to meet this objective?",
    type: 'MCQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'Build a content-based filtering recommendation engine with Apache Spark ML on Amazon EMR' },
      { id: 'b', text: 'Build a collaborative filtering recommendation engine with Apache Spark ML on Amazon EMR.' },
      { id: 'c', text: 'Build a model-based filtering recommendation engine with Apache Spark ML on Amazon EMR' },
      { id: 'd', text: 'Build a combinative filtering recommendation engine with Apache Spark ML on Amazon EMR' }
    ],
    correctAnswerIds: ['b'],
    explanation: "Collaborative filtering is the appropriate approach for predicting user preferences based on similarity to other users. Apache Spark ML on Amazon EMR provides the tools needed to build and scale this type of recommendation system."
  },
  {
    id: 'q3',
    text: "A Mobile Network Operator is building an analytics platform to analyze and optimize a company's operations using Amazon Athena and Amazon S3. The source systems send data in .CSV format in real time. The Data Engineering team wants to transform the data to the Apache Parquet format before storing it on Amazon S3. Which solution takes the LEAST effort to implement?",
    type: 'MCQ',
    domain: 'Data Engineering',
    options: [
      { id: 'a', text: 'Ingest .CSV data using Apache Kafka Streams on Amazon EC2 instances and use Kafka Connect S3 to serialize data as Parquet' },
      { id: 'b', text: 'Ingest .CSV data from Amazon Kinesis Data Streams and use Amazon Glue to convert data into Parquet.' },
      { id: 'c', text: 'Ingest .CSV data using Apache Spark Structured Streaming in an Amazon EMR cluster and use Apache Spark to convert data into Parquet.' },
      { id: 'd', text: 'Ingest .CSV data from Amazon Kinesis Data Streams and use Amazon Kinesis Data Firehose to convert data into Parquet.' }
    ],
    correctAnswerIds: ['d'],
    explanation: "Amazon Kinesis Data Firehose can automatically convert incoming data from CSV to Parquet format before delivering it to S3. This serverless solution requires the least operational effort compared to managing EMR clusters or writing custom streaming applications."
  },
  {
    id: 'q4',
    text: "A city wants to monitor its air quality to address the consequences of air pollution. A Machine Learning Specialist needs to forecast the air quality in parts per million of contaminates for the next 2 days in the city. As this is a prototype, only daily data from the last year is available. Which model is MOST likely to provide the best results in Amazon SageMaker?",
    type: 'MCQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'Use the Amazon SageMaker k-Nearest-Neighbors (kNN) algorithm on the single time series consisting of the full year of data with a predictor_type of regressor.' },
      { id: 'b', text: 'Use Amazon SageMaker Random Cut Forest (RCF) on the single time series consisting of the full year of data.' },
      { id: 'c', text: 'Use the Amazon SageMaker Linear Learner algorithm on the single time series consisting of the full year of data with a predictor_type of regressor.' },
      { id: 'd', text: 'Use the Amazon SageMaker Linear Learner algorithm on the single time series consisting of the full year of data with a predictor_type of classifier.' }
    ],
    correctAnswerIds: ['c'],
    explanation: "For time series forecasting with a single year of daily data, the Linear Learner algorithm with a regressor type is most appropriate. RCF is for anomaly detection, kNN is not ideal for time series, and this is a regression problem (predicting continuous values) not classification."
  },
  {
    id: 'q5',
    text: "A Data Engineer needs to build a model using a dataset containing customer credit card information. How can the Data Engineer ensure the data remains encrypted and the credit card information is secure?",
    type: 'MCQ',
    domain: 'Security',
    options: [
      { id: 'a', text: 'Use a custom encryption algorithm to encrypt the data and store the data on an Amazon SageMaker instance in a VPC. Use the SageMaker DeepAR algorithm to randomize the credit card numbers.' },
      { id: 'b', text: 'Use an IAM policy to encrypt the data on the Amazon S3 bucket and Amazon Kinesis to automatically discard credit card numbers and insert fake credit card numbers.' },
      { id: 'c', text: 'Use an Amazon SageMaker launch configuration to encrypt the data once it is copied to the SageMaker instance in a VPC. Use the SageMaker principal component analysis (PCA) algorithm to reduce the length of the credit card numbers.' },
      { id: 'd', text: 'Use AWS KMS to encrypt the data on Amazon S3 and Amazon SageMaker, and redact the credit card numbers from the customer data with AWS Glue.' }
    ],
    correctAnswerIds: ['d'],
    explanation: "AWS KMS provides managed encryption for data at rest in S3 and SageMaker. AWS Glue can be used to redact or mask sensitive data like credit card numbers. This follows AWS security best practices for handling PII (Personally Identifiable Information)."
  },
  {
    id: 'q6',
    text: "A Machine Learning Specialist is using an Amazon SageMaker notebook instance in a private subnet of a corporate VPC. The ML Specialist has important data stored on the Amazon SageMaker notebook instance's Amazon EBS volume, and needs to take a snapshot of that EBS volume. However, the ML Specialist cannot find the Amazon SageMaker notebook instance's EBS volume or Amazon EC2 instance within the VPC. Why is the ML Specialist not seeing the instance visible in the VPC?",
    type: 'MCQ',
    domain: 'SageMaker',
    options: [
      { id: 'a', text: 'Amazon SageMaker notebook instances are based on the EC2 instances within the customer account, but they run outside of VPCs.' },
      { id: 'b', text: 'Amazon SageMaker notebook instances are based on the Amazon ECS service within customer accounts.' },
      { id: 'c', text: 'Amazon SageMaker notebook instances are based on EC2 instances running within AWS service accounts.' },
      { id: 'd', text: 'Amazon SageMaker notebook instances are based on AWS ECS instances running within AWS service accounts.' }
    ],
    correctAnswerIds: ['c'],
    explanation: "SageMaker notebook instances are managed resources that run in AWS service accounts, not in the customer's account. This is why you cannot see the underlying EC2 instances or EBS volumes directly in your VPC console."
  },
  {
    id: 'q7',
    text: "A Machine Learning Specialist is building a model that will perform time series forecasting using Amazon SageMaker. The Specialist has finished training the model and is now planning to perform load testing on the endpoint so they can configure Auto Scaling for the model variant. Which approach will allow the Specialist to review the latency, memory utilization, and CPU utilization during the load test?",
    type: 'MCQ',
    domain: 'SageMaker',
    options: [
      { id: 'a', text: 'Review SageMaker logs that have been written to Amazon S3 by leveraging Amazon Athena and Amazon QuickSight to visualize logs as they are being produced.' },
      { id: 'b', text: 'Generate an Amazon CloudWatch dashboard to create a single view for the latency, memory utilization, and CPU utilization metrics that are outputted by Amazon SageMaker.' },
      { id: 'c', text: 'Build custom Amazon CloudWatch Logs and then leverage Amazon ES and Kibana to query and visualize the log data as it is generated by Amazon SageMaker.' },
      { id: 'd', text: 'Send Amazon CloudWatch Logs that were generated by Amazon SageMaker to Amazon ES and use Kibana to query and visualize the log data.' }
    ],
    correctAnswerIds: ['b'],
    explanation: "Amazon SageMaker automatically emits metrics like latency, CPU utilization, and memory utilization to Amazon CloudWatch. Creating a CloudWatch dashboard provides the most straightforward way to visualize and monitor these metrics in real-time during load testing."
  },
  {
    id: 'q8',
    text: "A manufacturing company has structured and unstructured data stored in an Amazon S3 bucket. A Machine Learning Specialist wants to use SQL to run queries on this data. Which solution requires the LEAST effort to be able to query this data?",
    type: 'MCQ',
    domain: 'Data Analytics',
    options: [
      { id: 'a', text: 'Use AWS Data Pipeline to transform the data and Amazon RDS to run queries.' },
      { id: 'b', text: 'Use AWS Glue to catalogue the data and Amazon Athena to run queries.' },
      { id: 'c', text: 'Use AWS Batch to run ETL on the data and Amazon Aurora to run the queries.' },
      { id: 'd', text: 'Use AWS Lambda to transform the data and Amazon Kinesis Data Analytics to run queries.' }
    ],
    correctAnswerIds: ['b'],
    explanation: "AWS Glue can crawl and catalog both structured and unstructured data in S3, making it queryable. Amazon Athena is a serverless interactive query service that allows you to analyze data in S3 using standard SQL, requiring minimal setup compared to other solutions."
  },
  {
    id: 'q9',
    text: "A Machine Learning Specialist is developing a custom video recommendation model for an application. The dataset used to train this model is very large with millions of data points and is hosted in an Amazon S3 bucket. The Specialist wants to avoid loading all of this data onto an Amazon SageMaker notebook instance because it would take hours to move and will exceed the attached 5 GB Amazon EBS volume on the notebook instance. Which approach allows the Specialist to use all the data to train the model?",
    type: 'MCQ',
    domain: 'SageMaker',
    options: [
      { id: 'a', text: 'Load a smaller subset of the data into the SageMaker notebook and train locally. Confirm that the training code is executing and the model parameters seem reasonable. Initiate a SageMaker training job using the full dataset from the S3 bucket using Pipe input mode.' },
      { id: 'b', text: 'Launch an Amazon EC2 instance with an AWS Deep Learning AMI and attach the S3 bucket to the instance. Train on a small amount of the data to verify the training code and hyperparameters. Go back to Amazon SageMaker and train using the full dataset' },
      { id: 'c', text: 'Use AWS Glue to train a model using a small subset of the data to confirm that the data will be compatible with Amazon SageMaker. Initiate a SageMaker training job using the full dataset from the S3 bucket using Pipe input mode.' },
      { id: 'd', text: 'Load a smaller subset of the data into the SageMaker notebook and train locally. Confirm that the training code is executing and the model parameters seem reasonable. Launch an Amazon EC2 instance with an AWS Deep Learning AMI and attach the S3 bucket to train the full dataset.' }
    ],
    correctAnswerIds: ['a'],
    explanation: "Using Pipe input mode allows SageMaker training jobs to stream data directly from S3 during training, avoiding the need to download the entire dataset. Testing with a subset locally first ensures the code works before scaling to the full dataset."
  },
  {
    id: 'q10',
    text: "A Machine Learning Specialist has completed a proof of concept for a company using a small data sample, and now the Specialist is ready to implement an end-to-end solution in AWS using Amazon SageMaker. The historical training data is stored in Amazon RDS. Which approach should the Specialist use for training a model using that data?",
    type: 'MCQ',
    domain: 'SageMaker',
    options: [
      { id: 'a', text: 'Write a direct connection to the SQL database within the notebook and pull data in' },
      { id: 'b', text: 'Push the data from Microsoft SQL Server to Amazon S3 using an AWS Data Pipeline and provide the S3 location within the notebook.' },
      { id: 'c', text: 'Move the data to Amazon DynamoDB and set up a connection to DynamoDB within the notebook to pull data in.' },
      { id: 'd', text: 'Move the data to Amazon ElastiCache using AWS DMS and set up a connection within the notebook to pull data in for fast access.' }
    ],
    correctAnswerIds: ['b'],
    explanation: "Moving data from RDS to S3 is the standard approach for SageMaker training. AWS Data Pipeline or AWS DMS can handle this migration. SageMaker works best with data stored in S3, which provides the scalability and performance needed for training jobs."
  },
  {
    id: 'q11',
    text: "A Machine Learning Specialist receives customer data for an online shopping website. The data includes demographics, past visits, and locality information. The Specialist must develop a machine learning approach to identify the customer shopping patterns, preferences, and trends to enhance the website for better service and smart recommendations. Which solution should the Specialist recommend?",
    type: 'MCQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'Latent Dirichlet Allocation (LDA) for the given collection of discrete data to identify patterns in the customer database.' },
      { id: 'b', text: 'A neural network with a minimum of three layers and random initial weights to identify patterns in the customer database.' },
      { id: 'c', text: 'Collaborative filtering based on user interactions and correlations to identify patterns in the customer database.' },
      { id: 'd', text: 'Random Cut Forest (RCF) over random subsamples to identify patterns in the customer database.' }
    ],
    correctAnswerIds: ['c'],
    explanation: "Collaborative filtering is the ideal approach for recommendation systems as it identifies patterns based on user interactions and correlations between similar users. It works well with demographic, visit history, and locality data to predict shopping preferences."
  },
  {
    id: 'q12',
    text: "A Machine Learning Specialist is working with a large company to leverage machine learning within its products. The company wants to group its customers into categories based on which customers will and will not churn within the next 6 months. The company has labeled the data available to the Specialist. Which machine learning model type should the Specialist use to accomplish this task?",
    type: 'MCQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'Linear regression' },
      { id: 'b', text: 'Classification' },
      { id: 'c', text: 'Clustering' },
      { id: 'd', text: 'Reinforcement learning' }
    ],
    correctAnswerIds: ['b'],
    explanation: "This is a supervised learning problem with labeled data predicting a binary outcome (churn or not churn). Classification models are designed for this type of task. Linear regression is for predicting continuous values, clustering is unsupervised learning, and reinforcement learning is for sequential decision-making."
  },
  {
    id: 'q13',
    text: "A graph from a forecasting model for testing a time series shows that the predicted values closely follow the actual values, capturing both the overall upward/downward movement and the recurring patterns at regular intervals. Considering the graph only, which conclusion should a Machine Learning Specialist make about the behavior of the model?",
    type: 'MCQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'The model predicts both the trend and the seasonality well' },
      { id: 'b', text: 'The model predicts the trend well, but not the seasonality.' },
      { id: 'c', text: 'The model predicts the seasonality well, but not the trend.' },
      { id: 'd', text: 'The model does not predict the trend or the seasonality well.' }
    ],
    correctAnswerIds: ['a'],
    explanation: "When a forecasting model's predicted values closely follow actual values throughout the time series, capturing both overall directional movement (trend) and recurring patterns (seasonality), it indicates good performance on both aspects."
  },
  {
    id: 'q14',
    text: "A company wants to classify user behavior as either fraudulent or normal. Based on internal research, a Machine Learning Specialist would like to build a binary classifier based on two features: age of account and transaction month. The class distribution for these features shows non-linearly separable patterns. Based on this information, which model would have the HIGHEST accuracy?",
    type: 'MCQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'Long short-term memory (LSTM) model with scaled exponential linear unit (SELU)' },
      { id: 'b', text: 'Logistic regression' },
      { id: 'c', text: 'Support vector machine (SVM) with non-linear kernel' },
      { id: 'd', text: 'Single perceptron with tanh activation function' }
    ],
    correctAnswerIds: ['c'],
    explanation: "For non-linearly separable data, SVM with a non-linear kernel (like RBF) can effectively create complex decision boundaries. Logistic regression and single perceptron are linear classifiers that would struggle with non-linear patterns. LSTM is overkill for this simple 2-feature problem."
  },
  {
    id: 'q15',
    text: "A Machine Learning Specialist at a company sensitive to security is preparing a dataset for model training. The dataset is stored in Amazon S3 and contains Personally Identifiable Information (PII). The dataset must be accessible from a VPC only and must not traverse the public internet. How can these requirements be satisfied?",
    type: 'MCQ',
    domain: 'Security',
    options: [
      { id: 'a', text: 'Create a VPC endpoint and apply a bucket access policy that restricts access to the given VPC endpoint and the VPC.' },
      { id: 'b', text: 'Create a VPC endpoint and apply a bucket access policy that allows access from the given VPC endpoint and an Amazon EC2 instance.' },
      { id: 'c', text: 'Create a VPC endpoint and use Network Access Control Lists (NACLs) to allow traffic between only the given VPC endpoint and an Amazon EC2 instance.' },
      { id: 'd', text: 'Create a VPC endpoint and use security groups to restrict access to the given VPC endpoint and an Amazon EC2 instance' }
    ],
    correctAnswerIds: ['a'],
    explanation: "A VPC endpoint (gateway endpoint for S3) provides private connectivity to S3 without going through the public internet. Restricting bucket access with a policy that only allows traffic through the specific VPC endpoint ensures data never traverses the public internet."
  },
  {
    id: 'q16',
    text: "During mini-batch training of a neural network for a classification problem, a Data Scientist notices that training accuracy oscillates. What is the MOST likely cause of this issue?",
    type: 'MCQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'The class distribution in the dataset is imbalanced.' },
      { id: 'b', text: 'Dataset shuffling is disabled.' },
      { id: 'c', text: 'The batch size is too big.' },
      { id: 'd', text: 'The learning rate is very high.' }
    ],
    correctAnswerIds: ['d'],
    explanation: "A high learning rate causes the optimizer to take large steps, potentially overshooting the optimal weights and bouncing around the minimum. This results in oscillating training accuracy. Reducing the learning rate is the standard fix for this issue."
  },
  {
    id: 'q17',
    text: "An employee found a video clip with audio on a company's social media feed. The language used in the video is Spanish. English is the employee's first language, and they do not understand Spanish. The employee wants to do a sentiment analysis. What combination of services is the MOST efficient to accomplish the task?",
    type: 'MCQ',
    domain: 'AI Services',
    options: [
      { id: 'a', text: 'Amazon Transcribe, Amazon Translate, and Amazon Comprehend' },
      { id: 'b', text: 'Amazon Transcribe, Amazon Comprehend, and Amazon SageMaker seq2seq' },
      { id: 'c', text: 'Amazon Transcribe, Amazon Translate, and Amazon SageMaker Neural Topic Model (NTM)' },
      { id: 'd', text: 'Amazon Transcribe, Amazon Translate and Amazon SageMaker BlazingText' }
    ],
    correctAnswerIds: ['a'],
    explanation: "Amazon Transcribe converts Spanish audio to text, Amazon Translate translates the Spanish text to English, and Amazon Comprehend performs sentiment analysis on the English text. This is the most efficient serverless solution requiring no model training."
  },
  {
    id: 'q18',
    text: "A Machine Learning Specialist is packaging a custom ResNet model into a Docker container so the company can leverage Amazon SageMaker for training. The Specialist is using Amazon EC2 P3 instances to train the model and needs to properly configure the Docker container to leverage the NVIDIA GPUs. What does the Specialist need to do?",
    type: 'MCQ',
    domain: 'SageMaker',
    options: [
      { id: 'a', text: 'Bundle the NVIDIA drivers with the Docker image.' },
      { id: 'b', text: 'Build the Docker container to be NVIDIA-Docker compatible.' },
      { id: 'c', text: 'Organize the Docker container\'s file structure to execute on GPU instances.' },
      { id: 'd', text: 'Set the GPU flag in the Amazon SageMaker CreateTrainingJob request body.' }
    ],
    correctAnswerIds: ['b'],
    explanation: "NVIDIA-Docker provides support for GPUs in Docker containers by including the necessary CUDA libraries and toolkit. Building the container to be NVIDIA-Docker compatible ensures it can leverage GPU resources on P3 instances. SageMaker automatically handles GPU allocation."
  },
  {
    id: 'q19',
    text: "A Machine Learning Specialist is building a logistic regression model that will predict whether or not a person will order a pizza. The Specialist is trying to build the optimal model with an ideal classification threshold. What model evaluation technique should the Specialist use to understand how different classification thresholds will impact the model's performance?",
    type: 'MCQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'Receiver operating characteristic (ROC) curve' },
      { id: 'b', text: 'Misclassification rate' },
      { id: 'c', text: 'Root Mean Square Error (RMSE)' },
      { id: 'd', text: 'L1 norm' }
    ],
    correctAnswerIds: ['a'],
    explanation: "The ROC curve plots the true positive rate against the false positive rate at various classification thresholds. This allows the Specialist to evaluate trade-offs and select the optimal threshold based on business requirements."
  },
  {
    id: 'q20',
    text: "An interactive online dictionary wants to add a widget that displays words used in similar contexts. A Machine Learning Specialist is asked to provide word features for the downstream nearest neighbor model powering the widget. What should the Specialist do to meet these requirements?",
    type: 'MCQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'Create one-hot word encoding vectors.' },
      { id: 'b', text: 'Produce a set of synonyms for every word using Amazon Mechanical Turk.' },
      { id: 'c', text: 'Create word embedding vectors that store edit distance with every other word.' },
      { id: 'd', text: 'Download word embeddings pre-trained on a large corpus.' }
    ],
    correctAnswerIds: ['d'],
    explanation: "Pre-trained word embeddings (like Word2Vec, GloVe, or fastText) capture semantic relationships between words based on context from large corpora. These embeddings enable finding words with similar contexts through nearest neighbor search, which is exactly what the dictionary widget needs."
  },
  {
    id: 'q21',
    text: "A Machine Learning Specialist is configuring Amazon SageMaker so multiple Data Scientists can access notebooks, train models, and deploy endpoints. To ensure the best operational performance, the Specialist needs to be able to track how often the Scientists are deploying models, GPU and CPU utilization on the deployed SageMaker endpoints, and all errors that are generated when an endpoint is invoked. Which services are integrated with Amazon SageMaker to track this information? (Choose two.)",
    type: 'MSQ',
    domain: 'SageMaker',
    options: [
      { id: 'a', text: 'AWS CloudTrail' },
      { id: 'b', text: 'AWS Health' },
      { id: 'c', text: 'AWS Trusted Advisor' },
      { id: 'd', text: 'Amazon CloudWatch' },
      { id: 'e', text: 'AWS Config' }
    ],
    correctAnswerIds: ['a', 'd'],
    explanation: "AWS CloudTrail logs API calls to track deployment activities like creating endpoints. Amazon CloudWatch collects metrics (CPU/GPU utilization) and logs errors from SageMaker endpoints. These two services provide comprehensive operational monitoring for SageMaker."
  },
  {
    id: 'q22',
    text: "A retail chain has been ingesting purchasing records from its network of 20,000 stores to Amazon S3 using Amazon Kinesis Data Firehose. To support training an improved machine learning model, training records will require new but simple transformations, and some attributes will be combined. The model needs to be retrained daily. Given the large number of stores and the legacy data ingestion, which change will require the LEAST amount of development effort?",
    type: 'MCQ',
    domain: 'Data Engineering',
    options: [
      { id: 'a', text: 'Require that the stores to switch to capturing their data locally on AWS Storage Gateway for loading into Amazon S3, then use AWS Glue to do the transformation.' },
      { id: 'b', text: 'Deploy an Amazon EMR cluster running Apache Spark with the transformation logic, and have the cluster run each day on the accumulating records in Amazon S3, outputting new/transformed records to Amazon S3.' },
      { id: 'c', text: 'Spin up a fleet of Amazon EC2 instances with the transformation logic, have them transform the data records accumulating on Amazon S3, and output the transformed records to Amazon S3.' },
      { id: 'd', text: 'Insert an Amazon Kinesis Data Analytics stream downstream of the Kinesis Data Firehose stream that transforms raw record attributes into simple transformed values using SQL.' }
    ],
    correctAnswerIds: ['d'],
    explanation: "Kinesis Data Analytics with SQL is serverless and can perform simple transformations on streaming data with minimal development effort. It integrates seamlessly with Kinesis Data Firehose, requiring no infrastructure management."
  },
  {
    id: 'q23',
    text: "A Machine Learning Specialist is building a convolutional neural network (CNN) that will classify 10 types of animals. The Specialist has built a series of layers in a neural network that will take an input image of an animal, pass it through a series of convolutional and pooling layers, and then finally pass it through a dense and fully connected layer with 10 nodes. The Specialist would like to get an output from the neural network that is a probability distribution of how likely it is that the input image belongs to each of the 10 classes. Which function will produce the desired output?",
    type: 'MCQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'Dropout' },
      { id: 'b', text: 'Smooth L1 loss' },
      { id: 'c', text: 'Softmax' },
      { id: 'd', text: 'Rectified linear units (ReLU)' }
    ],
    correctAnswerIds: ['c'],
    explanation: "The Softmax function transforms the output layer's logits into a probability distribution where all values sum to 1. It's the standard activation function for the final layer of multi-class classification problems."
  },
  {
    id: 'q24',
    text: "A Machine Learning Specialist trained a regression model, but the first iteration needs optimizing. The Specialist needs to understand whether the model is more frequently overestimating or underestimating the target. What option can the Specialist use to determine whether it is overestimating or underestimating the target value?",
    type: 'MCQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'Root Mean Square Error (RMSE)' },
      { id: 'b', text: 'Residual plots' },
      { id: 'c', text: 'Area under the curve' },
      { id: 'd', text: 'Confusion matrix' }
    ],
    correctAnswerIds: ['b'],
    explanation: "Residual plots display the difference between predicted and actual values. If residuals are predominantly positive, the model underestimates; if predominantly negative, it overestimates. Random distribution indicates good fit."
  },
  {
    id: 'q25',
    text: "A company wants to classify user behavior as either fraudulent or normal. Based on internal research, a Machine Learning Specialist would like to build a binary classifier based on two features: age of account and transaction month. The class distribution for these features shows complex, non-linear boundaries. Based on this information, which model would have the HIGHEST recall with respect to the fraudulent class?",
    type: 'MCQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'Decision tree' },
      { id: 'b', text: 'Linear support vector machine (SVM)' },
      { id: 'c', text: 'Naive Bayesian classifier' },
      { id: 'd', text: 'Single Perceptron with sigmoidal activation function' }
    ],
    correctAnswerIds: ['a'],
    explanation: "Decision trees can capture complex, non-linear decision boundaries by creating splits in the feature space. This flexibility allows them to achieve high recall for minority classes like fraud, especially when the class distribution shows complex patterns."
  },
  {
    id: 'q26',
    text: "A Machine Learning Specialist kicks off a hyperparameter tuning job for a tree-based ensemble model using Amazon SageMaker with Area Under the ROC Curve (AUC) as the objective metric. This workflow will eventually be deployed in a pipeline that retrains and tunes hyperparameters each night to model click-through on data that goes stale every 24 hours. With the goal of decreasing the amount of time it takes to train these models, and ultimately to decrease costs, the Specialist wants to reconfigure the input hyperparameter range(s). Which visualization will accomplish this?",
    type: 'MCQ',
    domain: 'SageMaker',
    options: [
      { id: 'a', text: 'A histogram showing whether the most important input feature is Gaussian.' },
      { id: 'b', text: 'A scatter plot with points colored by target variable that uses t-Distributed Stochastic Neighbor Embedding (t-SNE) to visualize the large number of input variables in an easier-to-read dimension.' },
      { id: 'c', text: 'A scatter plot showing the performance of the objective metric over each training iteration.' },
      { id: 'd', text: 'A scatter plot showing the correlation between maximum tree depth and the objective metric.' }
    ],
    correctAnswerIds: ['d'],
    explanation: "A scatter plot showing the correlation between hyperparameters (like max tree depth) and the objective metric helps identify optimal ranges. If the metric plateaus beyond certain values, you can restrict the hyperparameter search space, reducing training time and costs."
  },
  {
    id: 'q27',
    text: "A Machine Learning Specialist is creating a new natural language processing application that processes a dataset comprised of 1 million sentences. The aim is to then run Word2Vec to generate embeddings of the sentences and enable different types of predictions. Here is an example from the dataset: 'The quck BROWN FOX jumps over the lazy dog.' Which of the following are the operations the Specialist needs to perform to correctly sanitize and prepare the data in a repeatable manner? (Choose three.)",
    type: 'MSQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'Perform part-of-speech tagging and keep the action verb and the nouns only.' },
      { id: 'b', text: 'Normalize all words by making the sentence lowercase.' },
      { id: 'c', text: 'Remove stop words using an English stopword dictionary.' },
      { id: 'd', text: 'Correct the typography on "quck" to "quick".' },
      { id: 'e', text: 'One-hot encode all words in the sentence.' },
      { id: 'f', text: 'Tokenize the sentence into words.' }
    ],
    correctAnswerIds: ['b', 'c', 'f'],
    explanation: "For Word2Vec preprocessing: (1) Normalize by making text lowercase for consistency, (2) Remove stop words (common words like 'the', 'a') as they don't add semantic value, and (3) Tokenize to split sentences into individual words. These are standard NLP preprocessing steps."
  },
  {
    id: 'q28',
    text: "A company is using Amazon Polly to translate plaintext documents to speech for automated company announcements. However, company acronyms are being mispronounced in the current documents. How should a Machine Learning Specialist address this issue for future documents?",
    type: 'MCQ',
    domain: 'AI Services',
    options: [
      { id: 'a', text: 'Convert current documents to SSML with pronunciation tags.' },
      { id: 'b', text: 'Create an appropriate pronunciation lexicon.' },
      { id: 'c', text: 'Output speech marks to guide in pronunciation.' },
      { id: 'd', text: 'Use Amazon Lex to preprocess the text files for pronunciation' }
    ],
    correctAnswerIds: ['b'],
    explanation: "A pronunciation lexicon allows you to define how specific words or acronyms should be pronounced by Amazon Polly. You can upload this lexicon and apply it to all future syntheses, providing a centralized and maintainable solution."
  },
  {
    id: 'q29',
    text: "An insurance company is developing a new device for vehicles that uses a camera to observe drivers' behavior and alert them when they appear distracted. The company created approximately 10,000 training images in a controlled environment that a Machine Learning Specialist will use to train and evaluate machine learning models. During the model evaluation, the Specialist notices that the training error rate diminishes faster as the number of epochs increases and the model is not accurately inferring on the unseen test images. Which of the following should be used to resolve this issue? (Choose two.)",
    type: 'MSQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'Add vanishing gradient to the model.' },
      { id: 'b', text: 'Perform data augmentation on the training data.' },
      { id: 'c', text: 'Make the neural network architecture complex.' },
      { id: 'd', text: 'Use gradient checking in the model.' },
      { id: 'e', text: 'Add L2 regularization to the model.' }
    ],
    correctAnswerIds: ['b', 'e'],
    explanation: "This describes overfitting: low training error but high test error. Data augmentation (rotations, flips, etc.) increases training data diversity. L2 regularization adds a penalty on large weights, preventing the model from memorizing training examples. Both combat overfitting."
  },
  {
    id: 'q30',
    text: "When submitting Amazon SageMaker training jobs using one of the built-in algorithms, which common parameters MUST be specified? (Choose three.)",
    type: 'MSQ',
    domain: 'SageMaker',
    options: [
      { id: 'a', text: 'The training channel identifying the location of training data on an Amazon S3 bucket.' },
      { id: 'b', text: 'The validation channel identifying the location of validation data on an Amazon S3 bucket.' },
      { id: 'c', text: 'The IAM role that Amazon SageMaker can assume to perform tasks on behalf of the users.' },
      { id: 'd', text: 'Hyperparameters in a JSON array as documented for the algorithm used.' },
      { id: 'e', text: 'The Amazon EC2 instance class specifying whether training will be run using CPU or GPU.' },
      { id: 'f', text: 'The output path specifying where on an Amazon S3 bucket the trained model will persist.' }
    ],
    correctAnswerIds: ['a', 'c', 'f'],
    explanation: "For Word2Vec preprocessing: (1) Normalize by making text lowercase for consistency, (2) Remove stop words (common words like 'the', 'a') as they don't add semantic value, and (3) Tokenize to split sentences into individual words. These are standard NLP preprocessing steps."
  },
  {
    id: 'q28',
    text: "A company is using Amazon Polly to translate plaintext documents to speech for automated company announcements. However, company acronyms are being mispronounced in the current documents. How should a Machine Learning Specialist address this issue for future documents?",
    type: 'MCQ',
    domain: 'AI Services',
    options: [
      { id: 'a', text: 'Convert current documents to SSML with pronunciation tags.' },
      { id: 'b', text: 'Create an appropriate pronunciation lexicon.' },
      { id: 'c', text: 'Output speech marks to guide in pronunciation.' },
      { id: 'd', text: 'Use Amazon Lex to preprocess the text files for pronunciation' }
    ],
    correctAnswerIds: ['b'],
    explanation: "A pronunciation lexicon allows you to define how specific words or acronyms should be pronounced by Amazon Polly. You can upload this lexicon and apply it to all future syntheses, providing a centralized and maintainable solution."
  },
  {
    id: 'q29',
    text: "An insurance company is developing a new device for vehicles that uses a camera to observe drivers' behavior and alert them when they appear distracted. The company created approximately 10,000 training images in a controlled environment that a Machine Learning Specialist will use to train and evaluate machine learning models. During the model evaluation, the Specialist notices that the training error rate diminishes faster as the number of epochs increases and the model is not accurately inferring on the unseen test images. Which of the following should be used to resolve this issue? (Choose two.)",
    type: 'MSQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'Add vanishing gradient to the model.' },
      { id: 'b', text: 'Perform data augmentation on the training data.' },
      { id: 'c', text: 'Make the neural network architecture complex.' },
      { id: 'd', text: 'Use gradient checking in the model.' },
      { id: 'e', text: 'Add L2 regularization to the model.' }
    ],
    correctAnswerIds: ['b', 'e'],
    explanation: "This describes overfitting: low training error but high test error. Data augmentation (rotations, flips, etc.) increases training data diversity. L2 regularization adds a penalty on large weights, preventing the model from memorizing training examples. Both combat overfitting."
  },
  {
    id: 'q30',
    text: "When submitting Amazon SageMaker training jobs using one of the built-in algorithms, which common parameters MUST be specified? (Choose three.)",
    type: 'MSQ',
    domain: 'SageMaker',
    options: [
      { id: 'a', text: 'The training channel identifying the location of training data on an Amazon S3 bucket.' },
      { id: 'b', text: 'The validation channel identifying the location of validation data on an Amazon S3 bucket.' },
      { id: 'c', text: 'The IAM role that Amazon SageMaker can assume to perform tasks on behalf of the users.' },
      { id: 'd', text: 'Hyperparameters in a JSON array as documented for the algorithm used.' },
      { id: 'e', text: 'The Amazon EC2 instance class specifying whether training will be run using CPU or GPU.' },
      { id: 'f', text: 'The output path specifying where on an Amazon S3 bucket the trained model will persist.' }
    ],
    correctAnswerIds: ['a', 'c', 'f'],
    explanation: "Required parameters for SageMaker training jobs: (1) Training channel with S3 data location, (2) IAM role for permissions to access resources, and (3) Output path to save the trained model artifacts. Validation data, hyperparameters, and instance type are typically specified but not mandatory."
  },
  {
    id: 'q31',
    text: "A monitoring service generates 1 TB of scale metrics record data every minute. A Research team performs queries on this data using Amazon Athena. The queries run slowly due to the large volume of data, and the team requires better performance.\nHow should the records be stored in Amazon S3 to improve query performance?",
    type: 'MCQ',
    domain: 'Data Engineering',
    options: [
      { id: 'a', text: 'CSV files' },
      { id: 'b', text: 'Parquet files' },
      { id: 'c', text: 'Compressed JSON' },
      { id: 'd', text: 'RecordIO' }
    ],
    correctAnswerIds: ['b'],
    explanation: "Parquet is a columnar storage format that provides better query performance for analytics workloads. It allows Athena to scan only the columns needed for each query, compresses data efficiently, and stores schema metadata, making it ideal for large-scale data analytics."
  },
  {
    id: 'q32',
    text: "A Machine Learning Specialist is working with a media company to perform classification on popular articles from the company's website. The company is using random forests to classify how popular an article will be before it is published. Given the dataset containing a Day_Of_Week column, the Specialist wants to convert this column to binary values.\nWhat technique should be used to convert this column to binary values?",
    type: 'MCQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'Binarization' },
      { id: 'b', text: 'One-hot encoding' },
      { id: 'c', text: 'Tokenization' },
      { id: 'd', text: 'Normalization transformation' }
    ],
    correctAnswerIds: ['b'],
    explanation: "One-hot encoding converts categorical variables (like days of the week) into binary vectors where each category becomes a separate column with a 0 or 1 value. This is the standard technique for handling categorical data in machine learning models."
  },
  {
    id: 'q33',
    text: "A gaming company has launched an online game where people can start playing for free, but they need to pay if they choose to use certain features. The company needs to build an automated system to predict whether or not a new user will become a paid user within 1 year. The company has gathered a labeled dataset from 1 million users. The training dataset consists of 1,000 positive samples (from users who ended up paying within 1 year) and 999,000 negative samples (from users who did not use any paid features). Each data sample consists of 200 features including user age, device, location, and play patterns. Using this dataset for training, the Data Science team trained a random forest model that converged with over 99% accuracy on the training set. However, the prediction results on a test dataset were not satisfactory.\nWhich of the following approaches should the Data Science team take to mitigate this issue? (Choose two.)",
    type: 'MSQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'Add more deep trees to the random forest to enable the model to learn more features.' },
      { id: 'b', text: 'Include a copy of the samples in the test dataset in the training dataset.' },
      { id: 'c', text: 'Generate more positive samples by duplicating the positive samples and adding a small amount of noise to the duplicated data.' },
      { id: 'd', text: 'Change the cost function so that false negatives have a higher impact on the cost value than false positives.' },
      { id: 'e', text: 'Change the cost function so that false positives have a higher impact on the cost value than false negatives.' }
    ],
    correctAnswerIds: ['c', 'd'],
    explanation: "This is a severe class imbalance problem (1000:999000). Oversampling the minority class by duplicating with noise helps balance the dataset. Additionally, weighting false negatives higher makes the model prioritize catching the rare paid users, which is the business-critical class."
  },
  {
    id: 'q34',
    text: "A Data Scientist is developing a machine learning model to predict future patient outcomes based on information collected about each patient and their treatment plans. The model should output a continuous value as its prediction. The data available includes labeled outcomes for a set of 4,000 patients. The study was conducted on a group of individuals over the age of 65 who have a particular disease that is known to worsen with age. Initial models have performed poorly. While reviewing the underlying data, the Data Scientist notices that, out of 4,000 patient observations, there are 450 where the patient age has been input as 0. The other features for these observations appear normal compared to the rest of the sample population.\nHow should the Data Scientist correct this issue?",
    type: 'MCQ',
    domain: 'Data Science',
    options: [
      { id: 'a', text: 'Drop all records from the dataset where age has been set to 0.' },
      { id: 'b', text: 'Replace the age field value for records with a value of 0 with the mean or median value from the dataset' },
      { id: 'c', text: 'Drop the age feature from the dataset and train the model using the rest of the features.' },
      { id: 'd', text: 'Use k-means clustering to handle missing features' }
    ],
    correctAnswerIds: ['b'],
    explanation: "The age value of 0 is likely a data entry error or placeholder for missing data. Replacing these with the mean or median value (imputation) preserves the dataset size and maintains the statistical distribution better than dropping records or the entire feature."
  },
  {
    id: 'q35',
    text: "A Data Science team is designing a dataset repository where it will store a large amount of training data commonly used in its machine learning models. As Data Scientists may create an arbitrary number of new datasets every day, the solution has to scale automatically and be cost-effective. Also, it must be possible to explore the data using SQL.\nWhich storage scheme is MOST adapted to this scenario?",
    type: 'MCQ',
    domain: 'Data Engineering',
    options: [
      { id: 'a', text: 'Store datasets as files in Amazon S3.' },
      { id: 'b', text: 'Store datasets as files in an Amazon EBS volume attached to an Amazon EC2 instance.' },
      { id: 'c', text: 'Store datasets as tables in a multi-node Amazon Redshift cluster.' },
      { id: 'd', text: 'Store datasets as global tables in Amazon DynamoDB.' }
    ],
    correctAnswerIds: ['a'],
    explanation: "Amazon S3 is the most cost-effective and scalable solution for storing large amounts of training data. It automatically scales, offers tiered storage pricing, and integrates with services like Athena for SQL-based data exploration without requiring managed infrastructure."
  },
  {
    id: 'q36',
    text: "A Machine Learning Specialist deployed a model that provides product recommendations on a company's website. Initially, the model was performing very well and resulted in customers buying more products on average. However, within the past few months, the Specialist has noticed that the effect of product recommendations has diminished and customers are starting to return to their original habits of spending less. The Specialist is unsure of what happened, as the model has not changed from its initial deployment over a year ago.\nWhich method should the Specialist try to improve model performance?",
    type: 'MCQ',
    domain: 'Model Deployment',
    options: [
      { id: 'a', text: 'The model needs to be completely re-engineered because it is unable to handle product inventory changes.' },
      { id: 'b', text: 'The model\'s hyperparameters should be periodically updated to prevent drift.' },
      { id: 'c', text: 'The model should be periodically retrained from scratch using the original data while adding a regularization term to handle product inventory changes' },
      { id: 'd', text: 'The model should be periodically retrained using the original training data plus new data as product inventory changes.' }
    ],
    correctAnswerIds: ['d'],
    explanation: "This is a case of concept/data drift - the model was trained on old data that doesn't reflect current product inventory and customer preferences. Periodically retraining with fresh data that includes recent product changes will help the model adapt to current patterns."
  },
  {
    id: 'q37',
    text: "A Machine Learning Specialist working for an online fashion company wants to build a data ingestion solution for the company's Amazon S3-based data lake. The Specialist wants to create a set of ingestion mechanisms that will enable future capabilities comprised of: Real-time analytics, Interactive analytics of historical data, Clickstream analytics, Product recommendations\nWhich services should the Specialist use?",
    type: 'MCQ',
    domain: 'Data Engineering',
    options: [
      { id: 'a', text: 'AWS Glue as the data catalog; Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for real-time data insights; Amazon Kinesis Data Firehose for delivery to Amazon ES for clickstream analytics; Amazon EMR to generate personalized product recommendations' },
      { id: 'b', text: 'Amazon Athena as the data catalog: Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for near-real-time data insights; Amazon Kinesis Data Firehose for clickstream analytics; AWS Glue to generate personalized product recommendations' },
      { id: 'c', text: 'AWS Glue as the data catalog; Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for historical data insights; Amazon Kinesis Data Firehose for delivery to Amazon ES for clickstream analytics; Amazon EMR to generate personalized product recommendations' },
      { id: 'd', text: 'Amazon Athena as the data catalog; Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for historical data insights; Amazon DynamoDB streams for clickstream analytics; AWS Glue to generate personalized product recommendations' }
    ],
    correctAnswerIds: ['a'],
    explanation: "AWS Glue catalogs data for Athena queries. Kinesis Data Streams handles real-time ingestion, Kinesis Data Analytics provides real-time insights, Kinesis Data Firehose delivers to Amazon ES for clickstream search, and Amazon EMR provides the compute power for generating recommendation models at scale."
  },
  {
    id: 'q38',
    text: "A company is observing low accuracy while training on the default built-in image classification algorithm in Amazon SageMaker. The Data Science team wants to use an Inception neural network architecture instead of a ResNet architecture.\nWhich of the following will accomplish this? (Choose two.)",
    type: 'MSQ',
    domain: 'SageMaker',
    options: [
      { id: 'a', text: 'Customize the built-in image classification algorithm to use Inception and use this for model training.' },
      { id: 'b', text: 'Create a support case with the SageMaker team to change the default image classification algorithm to Inception.' },
      { id: 'c', text: 'Bundle a Docker container with TensorFlow Estimator loaded with an Inception network and use this for model training.' },
      { id: 'd', text: 'Use custom code in Amazon SageMaker with TensorFlow Estimator to load the model with an Inception network, and use this for model training.' },
      { id: 'e', text: 'Download and apt-get install the inception network code into an Amazon EC2 instance and use this instance as a Jupyter notebook in Amazon SageMaker.' }
    ],
    correctAnswerIds: ['c', 'd'],
    explanation: "SageMaker supports custom models through Docker containers (bring your own container) or custom code with framework estimators. Both approaches allow you to use Inception architecture instead of the default ResNet. The built-in algorithm cannot be customized, and AWS support won't change defaults."
  },
  {
    id: 'q39',
    text: "A Machine Learning Specialist built an image classification deep learning model. However, the Specialist ran into an overfitting problem in which the training and testing accuracies were 99% and 75%, respectively.\nHow should the Specialist address this issue and what is the reason behind it?",
    type: 'MCQ',
    domain: 'Machine Learning',
    options: [
      { id: 'a', text: 'The learning rate should be increased because the optimization process was trapped at a local minimum.' },
      { id: 'b', text: 'The dropout rate at the flatten layer should be increased because the model is not generalized enough.' },
      { id: 'c', text: 'The dimensionality of dense layer next to the flatten layer should be increased because the model is not complex enough.' },
      { id: 'd', text: 'The epoch number should be increased because the optimization process was terminated before it reached the global minimum.' }
    ],
    correctAnswerIds: ['b'],
    explanation: "A large gap between training accuracy (99%) and test accuracy (75%) indicates overfitting - the model memorized the training data. Increasing dropout randomly disables neurons during training, forcing the network to learn more robust, generalized features that transfer better to unseen data."
  },
  {
    id: 'q40',
    text: "A Machine Learning team uses Amazon SageMaker to train an Apache MXNet handwritten digit classifier model using a research dataset. The team wants to receive a notification when the model is overfitting. Auditors want to view the Amazon SageMaker log activity report to ensure there are no unauthorized API calls.\nWhat should the Machine Learning team do to address the requirements with the least amount of code and fewest steps?",
    type: 'MCQ',
    domain: 'SageMaker',
    options: [
      { id: 'a', text: 'Implement an AWS Lambda function to log Amazon SageMaker API calls to Amazon S3. Add code to push a custom metric to Amazon CloudWatch. Create an alarm in CloudWatch with Amazon SNS to receive a notification when the model is overfitting.' },
      { id: 'b', text: 'Use AWS CloudTrail to log Amazon SageMaker API calls to Amazon S3. Add code to push a custom metric to Amazon CloudWatch. Create an alarm in CloudWatch with Amazon SNS to receive a notification when the model is overfitting.' },
      { id: 'c', text: 'Implement an AWS Lambda function to log Amazon SageMaker API calls to AWS CloudTrail. Add code to push a custom metric to Amazon CloudWatch. Create an alarm in CloudWatch with Amazon SNS to receive a notification when the model is overfitting.' },
      { id: 'd', text: 'Use AWS CloudTrail to log Amazon SageMaker API calls to Amazon S3. Set up Amazon SNS to receive a notification when the model is overfitting' }
    ],
    correctAnswerIds: ['b'],
    explanation: "AWS CloudTrail is a managed service that automatically logs API calls to S3 without custom code. CloudWatch with custom metrics and alarms handles the overfitting notification. This solution requires minimal code compared to implementing Lambda functions for logging."
  }
];
