# container: scipy
# needs: network, GROQ_API_KEY
# Install once with: pip install groq python-dotenv
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # take environment variables from .env.
api_key = os.getenv("GROQ_API_KEY")
if api_key is None:
    raise ValueError("GROQ_API_KEY environment variable not set.")

# region: chat
client = Groq(api_key=api_key)
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Why is the weather in Ottawa colder than Tampa?",
        }
    ],
    model="openai/gpt-oss-20b",  # Groq retires models often
)
print(chat_completion.choices[0].message.content)
# endregion
