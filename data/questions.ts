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
  }
];
