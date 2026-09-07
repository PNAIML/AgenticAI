from google import genai

# 1. Hardcode your API key string here directly as a variable
api_key = "AQ.Ab8RN6LB2SqtwYnwaFok1UIbYmzsfRPFp-j8w229ydliguR5XA"

# 2. Check if you accidentally left it empty
if not api_key:
    raise ValueError("Please paste your API key string into the api_key variable.")

# 3. Pass the defined variable to the client
client = genai.Client(api_key=api_key)

print("="*60)
print("Gemini Chat")
print("Type exit to quit")
print("="*60)

while True:
    prompt = input("\nYou : ")
    if prompt.lower() == "exit":
        break
    
    try:
        # Use a modern model name like gemini-2.0-flash or gemini-1.5-flash
        response = client.models.generate_content(
            model="gemini-flash-latest", 
            contents=prompt
        )
        print("\nGemini :", response.text)
    except Exception as e:
        print(f"\nAn error occurred: {e}")

