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

class Session:
    def __init__(self, model, system_prompt=None, max_history=10):
        self.model = model
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.history = []
    
    def add_player_message(self, message):
        self.history.append({"role" : "player", "message" : message})
    
    def add_model_message(self, message):
        self.history.append({"role" : "model", "message" : message})

    def reset(self):
        self.history = []
    
    def build_prompt(self):
        prompt = ""

        if self.system_prompt:
            prompt += f"System: {self.system_prompt}\n\n"
        
        for msg in self.history:
            if msg["role"] == "player":
                prompt += f"Player: {msg['message']}\n\n"
            if msg["role"] == "model":
                prompt += f"Model: {msg['message']}\n\n"
        
        prompt += "Model: "

        return prompt