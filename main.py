##integrate of code 

import os 
from constants import openai_key
from langchain.llms import OpenAI 
from langchain import PromptTemplate  
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain 
import streamlit as st 
from langchain.chains import SequentialChain


from langchain.memory import ConversationBufferMemory 



os.environ["OPENAI_API_KEY"]= openai_key


# stramlit framework 

st.title(' Celebrity search results ')
input_text=st.text_input(" Type the Celebrity Name ")




#prompt templates 
first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template= "Tell me avbout Celebrity {name}"
)



#memory 
person_memory=ConversationBufferMemory(input_key='name',memory_key='Chat_history')
dob_memory=ConversationBufferMemory(input_key='person',memory_key='Chat_history')
description_memory=ConversationBufferMemory(input_key='dob',memory_key='Description_history')


#OPENAI LLMS 

llm=OpenAI(temperature=0.8)
chain=LLMChain(
    llm=llm,prompt=first_input_prompt,verbose=True,output_key='person',memory=person_memory)


second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="when was {person} born"
)


chain2=LLMChain(llm=llm,prompt=second_input_prompt, verbose=True ,output_key="dob",memory=dob_memory)


third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template= "Menction 5 major events happened around {dob} in the world"
    
)   
chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description',memory=description_memory)


parent_chain=SequentialChain( 
    chains=[chain,chain2,chain3], input_variables=['name'],output_variables=['person','dob','description'],verbose=True)



if input_text:
    st.write(parent_chain({'name':input_text}))
    
    with st.expander('Celebrity name'):
        st.info(person_memory.buffer)
    with st.expander('Celebrity DOB'):
        st.info(dob_memory.buffer)
    with st.expander('major Events happened celebrity born Year'):
        st.info(description_memory.buffer)
    


