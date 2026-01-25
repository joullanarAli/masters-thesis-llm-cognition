# import google.generativeai as genai
from google import genai

# genai.configure(api_key="AIzaSyBpi1qhOGSe55GadvgnVtSvjEfomK683js")
# model = genai.GenerativeModel('gemini-1.5-flash')
# response = model.generate_content("Write a Python function to reverse a string.")
# print(response.text)

# Configure the client with the new SDK
client = genai.Client(api_key="AIzaSyBpi1qhOGSe55GadvgnVtSvjEfomK683js")

# Choose a model from the table above
model = "gemini-3-flash-preview"

# Generate content
response = client.models.generate_content(
    model=model,
    contents="Write a Python function to reverse a string."
)

print(response.text)

