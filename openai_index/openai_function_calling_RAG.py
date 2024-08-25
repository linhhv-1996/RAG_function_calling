from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import json, os 

# Configure API key for OpenAI
os.environ['OPENAI_API_KEY'] = "sk-..."
client = OpenAI()

# Define function to retrieve information about employees
def about_employee(info):
    """
    Provides information about the company's management team and employees
    
    Args:
        info (str): Information to search about employees
    
    Returns:
        str: JSON string containing information about employees
    """
    
    # Load the employee information database
    db = FAISS.load_local("openai_index\employee_index", OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization = True)
    
    # Search for similar information in the database
    results = db.similarity_search(info, k=1)  # Get the top result (k=1)
    
    # Convert the results to JSON format
    docs = [{"content": doc.page_content} for doc in results]
    docs_string = json.dumps(docs, ensure_ascii=False)

    return docs_string

# Define function to retrieve information about products
def about_products(info):
    """
    Provides information about the products available at the company
    
    Args:
        info (str): Information to search about products
    
    Returns:
        str: JSON string containing information about products
    """
    
    # Load the product information database
    db = FAISS.load_local("openai_index\products_index", OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization = True)
    
    # Search for similar information in the database
    results = db.similarity_search(info, k=3)  # Get the top 3 results (k=3)
    
    # Convert the results to JSON format
    docs = [{"content": doc.page_content} for doc in results]
    docs_string = json.dumps(docs, ensure_ascii=False)

    return docs_string

# Define function to retrieve product reviews
def reviews_search(info):
    """
    Provides user reviews of the company's products
    
    Args:
        info (str): Information to search about reviews
    
    Returns:
        str: JSON string containing reviews about products
    """
    
    # Load the product reviews database
    db = FAISS.load_local("openai_index\\reviews_index", OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization = True)
    
    # Search for similar information in the database
    results = db.similarity_search(info, k=5)  # Get the top 5 results (k=5)
    
    # Convert the results to JSON format
    docs = [{"content": doc.page_content} for doc in results]
    docs_string = json.dumps(docs, ensure_ascii=False)

    return docs_string

# List of functions available for the virtual assistant
available_functions = {
    "about_employee": about_employee,
    "about_products": about_products,
    "reviews_search": reviews_search
}

# Define tools for the virtual assistant
tools = [
    {
        "type": "function",
        "function": {
            "name": "about_employee",
            "description": "Provides information about the company's management team and employees that you need to know.",
            "parameters": {
                "type": "object",
                "properties": {
                    "info": {
                        "type": "string",
                        "description": "Information you need to search, e.g., Email of himmeow the coder.",
                    },
                },
                "required": ["info"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "about_products",
            "description": "Provides information about the company's products that you need to know.",
            "parameters": {
                "type": "object",
                "properties": {
                    "info": {
                        "type": "string",
                        "description": "Information you need to search, e.g., Origin of the Luon Vui Tươi men's T-shirt.",
                    },
                },
                "required": ["info"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reviews_search",
            "description": "Provides reviews about the company's products.",
            "parameters": {
                "type": "object",
                "properties": {
                    "info": {
                        "type": "string",
                        "description": "Reviews you need to search, e.g., User reviews of the men's T-shirt.",
                    },
                },
                "required": ["info"],
            },
        },
    },
]

# Memory of the virtual assistant, containing conversation history
memory = [
    {
        "role": "system", 
        "content": """You are an intelligent virtual assistant working for a merchandise company, provided with tools to retrieve data to answer user questions. \n
                     IMPORTANT: ALWAYS search for information in the provided tools before answering user questions!"""
    },
]

# Function to send a chat request and handle responses
def chat_completion_request(messages, functions=None, model="gpt-4o"):
    """
    Sends a chat request to the OpenAI API and handles the response.
    
    Args:
        messages (list): List of messages in the conversation.
        functions (list): List of tools available for the virtual assistant.
        model (str): Language model used for the chatbot.
    
    Returns:
        str: Response from the chatbot or error message.
    """
    
    try:
        # Send the chat request
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=functions,
            tool_choice="auto", 
            temperature=0,
        )

        # Get the response from the OpenAI API
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # If there are tool calls
        if tool_calls:
            # Add the chatbot's response to memory
            messages.append(response_message)

            # Process each tool call
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                
                # Call the function corresponding to the requested tool name
                if function_name in available_functions:
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call(function_args.get("info"))
                    
                    # Add the tool's result to memory
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )
                
            # Send a new chat request with additional information from the tool
            return chat_completion_request(messages=messages, functions=functions)
            
        # If no tool calls, return the direct response
        else:
            msg = response_message.content
            return msg
        
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
    
# Run the chatbot
if __name__ == "__main__":
    print("Start chatting with the virtual assistant (type 'exit' to stop)")
    while True:
        query = input("User: ")
        if query.lower() == "exit":
            break

        # Add the user's question to memory
        memory.append({"role": "user", "content": query})
        
        # Send a chat request and receive a response
        response = chat_completion_request(messages=memory, functions=tools)
        
        # Print the chatbot's response
        print(f"Chatbot: {response}")
        
        # Add the chatbot's response to memory
        memory.append({"role": "assistant", "content": response})
