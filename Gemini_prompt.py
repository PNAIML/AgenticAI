from google import genai

content=input("ask you AI:-")

client = genai.Client(
    api_key="AQ.Ab8RN6LB2SqtwYnwaFok1UIbYmzsfRPFp-j8w229ydliguR5XA"   # Replace with your API key
)

response = client.models.generate_content(
    model="gemini-flash-latest",
    contents=content
)

print(response.text)
