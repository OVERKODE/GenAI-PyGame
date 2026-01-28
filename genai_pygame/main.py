from transformers import pipeline

def create_model(model_name: str):
    return_model = pipeline(
        "text-generation",
        model=model_name
    )

    return return_model

def generate_response(model, message: str, max_tokens: int, temperature: float):
    response = model(
        message,
        max_new_tokens=max_tokens,
        temperature=temperature,
        return_full_text=False
    )

    return response[0]["generated_text"]