import pandas as pd
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from tqdm.auto import tqdm
from uuid import uuid4
import pandas as pd
from dotenv import load_dotenv
import os
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("openai_key")

embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

llm = ChatOpenAI(
    model_name='gpt-3.5-turbo-0125',
    temperature=0.0
)

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)


coustom_prompt ="""Assistant is a large language model trained by OpenAI.
The name of the Assistant is {bot_name}.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

TOOLS:
------

Assistant has access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""



def get_final_prompt(bot_name):
    name_prompt = f"The Assistant also know as {bot_name} \n"
    
    prompt = hub.pull("hwchase17/react-chat")
    prompt.template = name_prompt+str(prompt.template)
    return prompt
    

def create_embadding_from_csv(file_path,bot_name):
    df = pd.read_csv(file_path)
    
    doc_list = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Rows"):
            # Creating the product details page
            question = row['question']
            answer = row['answer']
            product_string = f"question:{question}\n,answer:{answer}"
            product_metadata= {
                "question":row['question'],
                "answer":row['answer']
            }
            doc =  Document(page_content=product_string, metadata=product_metadata)
            doc_list.append(doc)
    store = Chroma.from_documents(doc_list, embedding_function,persist_directory=f'vectordb1/{bot_name}')
    store.persist()

def get_vector_db(bot_name):
    vectordb = Chroma(persist_directory=f"vectordb1/{bot_name}", embedding_function=embedding_function)
    return vectordb



def react_agent_chat(vectordb,user_query,botname):
    qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever()
    )
    tools = [
        Tool(
            name='Knwledge Base',
            func=qa.run,
            description=(
                'use this tool when answering general knowledge queries to get '
                'more information about the topic'
                )
            )
    ]
    final_prompt = get_final_prompt(bot_name=botname)
    print(final_prompt)
    # Construct the ReAct agent
    agent = create_react_agent(llm, tools, final_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,memory=conversational_memory) 
    
    agent_response = agent_executor.invoke({"input":user_query})
    
    return agent_response['output']
