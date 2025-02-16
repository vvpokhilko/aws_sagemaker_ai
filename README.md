# Project: Fine-Tuning and Deploying a Hugging Face Model for Text Classification on AWS SageMaker
## **Project Overview**
In this project, we fine-tuned and deployed a **DistilBERT** model from **Hugging Face** 🤗 for **news classification** using **AWS SageMaker AI**. The project involved training the model on a **custom news dataset** stored in an **S3 bucket**, conducting **exploratory data analysis (EDA) in Jupyter Lab**, and deploying the trained model as an API endpoint using **AWS Lambda and API Gateway**.

## **Key Steps**
### **1. Data Preparation and Exploration**
* Collected a **news corpus dataset** and uploaded it to an **AWS S3 bucket** for storage.
* Conducted **Exploratory Data Analysis (EDA)** in **Jupyter Lab** to understand data distribution, label balance, and text preprocessing needs.
* Tokenized the dataset using **DistilBERT tokenizer** from **Hugging Face** 🤗 and converted it into the required format for training.

### **2. Model Training and Fine-Tuning**
* Loaded the **pretrained DistilBERT model** from Hugging Face.
* Configured the **training parameters**, including batch size, learning rate, and number of epochs.
* Fine-tuned the model on the **news classification dataset** using **AWS SageMaker AI**.
* Used **CrossEntropyLoss** as the loss function and **Adam optimizer** for parameter updates.
* Tracked **training loss** and **accuracy** at regular intervals for model performance monitoring.

### **3. Model Evaluation and Deployment**
* Evaluated the model’s performance on a **validation dataset** using accuracy and loss metrics.
* Saved the **fine-tuned model** and tokenizer in an **S3 bucket**.
* Created a **SageMaker model instance** for inference.

### **4. Building the API for Model Inference**
* Developed an **AWS Lambda function** to serve predictions from the trained model.
* Configured **API Gateway** to expose the Lambda function as an **HTTP endpoint**.
* Enabled **POST requests** to send input text data and receive model predictions.

### **5. Testing and Deployment**
* Deployed the API endpoint and tested it with real-world news articles.
* Ensured low-latency predictions and validated model accuracy on unseen data.
* Successfully created a fully functional **end-to-end machine learning pipeline for news classification**.

## **Technologies Used**
* **Machine Learning Framework**: Hugging Face 🤗, Transformers, PyTorch
* **Cloud Services**: AWS SageMaker, S3, Lambda, API Gateway
* **Data Processing & Analysis**: Pandas, NumPy, Matplotlib, Jupyter Lab
* **Deployment Tools**: AWS Lambda, API Gateway

This project demonstrates how to **fine-tune a transformer-based model, deploy it on AWS**, and **serve it as an API**, making it accessible for real-time text classification. 🚀
