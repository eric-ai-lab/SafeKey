import asyncio
import time
from openai import AzureOpenAI

endpoint_mini = "your-endpoint"
model_name_mini = "your-model-name"
deployment_mini = "your-deployment-name"
subscription_key_mini = "your-subscription-key"
api_version_mini = "your-api-version"

client_mini = AzureOpenAI(
    api_version=api_version_mini,
    azure_endpoint=endpoint_mini,
    api_key=subscription_key_mini,
)


endpoint = "your-endpoint"
subscription_key =  "your-subscription-key"
deployment ="your-deployment-name"
api_version = 'your-api-version'

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Function to run in parallel
def get_response(prompt):
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=40,
            temperature=0,
            top_p=1,
            model=deployment
        )
        return response.choices[0].message.content
    except:
        print("content filter")
        return "None"


def get_response_mini(prompt):
    try:
        response = client_mini.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=40,
            temperature=0,
            top_p=1,
            model=deployment_mini
        )
        return response.choices[0].message.content
    except:
        print("content filter")
        return "None"


# Async wrapper
async def async_batch(prompts):
    tasks = [asyncio.to_thread(get_response, prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)


async def async_batch_mini(prompts):
    tasks = [asyncio.to_thread(get_response_mini, prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)