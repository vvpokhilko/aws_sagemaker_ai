import json  # Library for handling JSON data
import boto3  # AWS SDK for Python to interact with AWS services, including SageMaker

def lambda_handler(event, context):
    """
    AWS Lambda function to classify a news headline using a deployed SageMaker text classification model.
    """
    # Create a SageMaker runtime client to communicate with the deployed model
    sagemaker_runtime = boto3.client('sagemaker-runtime')

    # Parse the incoming event body (assuming it's a JSON payload)
    body = json.loads(event['body'])  # Extract request body
    
    # Extract the headline text from the request payload
    headline = body['query']['headline']  # Example: "New vaccines discovered"
    
    # Define the name of the deployed SageMaker inference endpoint
    endpoint_name = 'multiclass-text-classification-endpoint-deployment'
    
    # Format the input as a JSON payload to match the model's expected format
    payload = json.dumps({"inputs": headline})

    # Invoke the SageMaker endpoint with the input payload
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,  # Specify the deployed model endpoint
        ContentType='application/json',  # Define the content type as JSON
        Body=payload  # Send the input headline for classification
    )    
    
    # Parse the response from the model
    result = json.loads(response['Body'].read().decode())

    # Return the classification result as a JSON response
    return {
        'statusCode': 200,  # HTTP status code indicating success
        'body': json.dumps(result)  # Return the model's classification output
    }