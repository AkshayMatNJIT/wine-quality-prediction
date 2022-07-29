# wine-quality-prediction

EMR is used to run the Apache spark and runs on a cluster of EC2 instances. Amazon EMR (previously called Amazon Elastic MapReduce) is a managed cluster platform that simplifies running big data frameworks, such as Apache Hadoop and Apache Spark, on AWS to process and analyze vast amounts of data. Using these frameworks and related open-source projects, you can process data for analytics purposes and business intelligence workloads. Amazon EMR also lets you transform and move large amounts of data into and out of other AWS data stores and databases, such as Amazon Simple Storage Service (Amazon S3) and Amazon DynamoDB. PySpark is used to train and test the model. RandomForest is used to evaluate the individual models and take a collective decision using voting. Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. Training.py file is used to train the model. The model then will be pushed to S3 bucket. S3 bucket is required to have a common repository which every worker node can access. Dockerfile is used to generate the docker image, which is uploaded to docker hub. This image is used to create docker container for testing the model.

Perform the following steps:

Create a EMR in AWS with 1 master and 4 slave
Log in to master EC2
Run dependency.sh, it will install all the required dependency
All the required CSV files are uploaded in S3 under source-data folder
Run Training.py as "spark-submi Training.py" - This will train and save the model in S3 under data-output and prints the F1 score
Log in to the docker hub repo
In the master run docker to build a container - "sudo docker run --name pa2validate-container cs62/pa2validate-aws". This will take the repository from the Docker hub and create the container.
The container runs the validationData.csv and prints Test Error Link to docker hub image: [DockerHub](https://hub.docker.com/r/akshaymutha611/akshay-docker)
