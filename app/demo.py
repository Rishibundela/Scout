
from langchain_google_genai import ChatGoogleGenerativeAI



def main():
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
    response = llm.invoke("Write a 100 words essay on langchain")
    print(response)
    print("-"*30)
    print(response.content)
    print("-"*30)
    print(response.content[0])
    print("-"*30)
    print(response.content[0]["text"])

if __name__ == "__main__":
    main()