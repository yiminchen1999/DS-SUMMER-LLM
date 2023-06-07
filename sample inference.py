import json
import requests

API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
headers = {"Authorization": "Bearer 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"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = query({
    "inputs": "Learning Analytics is?",
})


print(output[0]['generated_text'])
