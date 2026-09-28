from dotenv import load_dotenv
from graph.graph import app

load_dotenv()  

if __name__ == "__main__":
    result = app.invoke(input = {"question": "What is prompt engineering?"})
    print(result["generation"])