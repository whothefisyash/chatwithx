from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Access your API keys
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
