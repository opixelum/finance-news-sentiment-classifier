import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# Define global variables for model and tokenizer
tokenizer = None
model = None


def model_fn(model_dir):
    """
    Load the model and tokenizer from the specified directory.
    This function is called by SageMaker when the container starts up.

    Args:
        model_dir (str): The directory where model artifacts are stored

    Returns:
        tuple: The loaded model and tokenizer
    """

    # Load tokenizer and model from the saved artifacts
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

    # Put model in evaluation mode
    model.eval()

    return model, tokenizer


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input.

    Args:
        request_body (str): The request payload
        request_content_type (str): The content type of the request

    Returns:
        dict: Dictionary containing the input text to process
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}. Use 'application/json'")


def predict_fn(input_data, model_and_tokenizer):
    """
    Apply model to the input data.

    Args:
        input_data (dict): Input data in the format {"inputs": "text to classify"} or {"inputs": ["text1", "text2", ...]}
        model_and_tokenizer (tuple): Tuple containing the model and tokenizer

    Returns:
        dict: Model predictions
    """
    model, tokenizer = model_and_tokenizer

    # Process single text or batch of texts
    if isinstance(input_data.get('inputs'), list):
        texts = input_data['inputs']
    else:
        texts = [input_data['inputs']]

    # Tokenize inputs
    encoded_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512
    )

    # Get model predictions
    with torch.no_grad():
        outputs = model(**encoded_inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)

    # Convert predictions to a list of dictionaries
    predictions = []
    for i, probs in enumerate(probabilities):
        # Get the predicted class and its probability
        predicted_class = torch.argmax(probs).item()
        confidence = probs[predicted_class].item()

        # Add other class probabilities
        class_probs = {str(j): round(prob.item(), 4) for j, prob in enumerate(probs)}

        predictions.append({
            'text': texts[i],
            'predicted_class': predicted_class,
            'confidence': round(confidence, 4),
            'probabilities': class_probs
        })

    return {'predictions': predictions if len(predictions) > 1 else predictions[0]}


def output_fn(prediction, response_content_type):
    """
    Serialize the prediction output.

    Args:
        prediction (dict): The prediction result from predict_fn
        response_content_type (str): The content type of the response

    Returns:
        str: Serialized predictions
    """
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}. Use 'application/json'")