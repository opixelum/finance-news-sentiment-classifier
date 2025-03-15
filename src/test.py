import json
import boto3


def test_bert_endpoint(endpoint_name, text=None, batch=False):
    """
    Test the deployed BERT model endpoint with sample data.

    Args:
        endpoint_name (str): The name of the SageMaker endpoint
        text (str, optional): Text to classify. If None, uses default text
        batch (bool, optional): Whether to test with batch input
    """
    # Initialize SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime')

    # Prepare input data
    if batch:
        if text:
            input_data = {
                'inputs': [text, "This is another example sentence.", "A third sample for testing."]
            }
        else:
            input_data = {
                'inputs': [
                    "I really enjoyed this product, it works great!",
                    "The service was terrible and the staff was rude.",
                    "This is neither good nor bad, just average."
                ]
            }
        print(f"Testing endpoint with batch of {len(input_data['inputs'])} texts...")
    else:
        if text:
            input_data = {'inputs': text}
        else:
            input_data = {'inputs': "I really enjoyed this product, it works great!"}

    # Invoke the endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(input_data)
    )

    # Parse and display the response
    result = json.loads(response['Body'].read().decode())
    print("\nEndpoint Response:")
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    endpoint = "finance-news-sentiments-classifier"
    text = "Nike sales have increased"
    test_bert_endpoint(endpoint, text, False)