# PDF-RAG Agent

Desktop-приложение на PyQt6 для интеллектуального поиска и ответов на вопросы по локальным PDF-документам.

# Tech Stack
-Langraph
-Qdrant
-PyQT6
-LangChain
-Arire Phoenix(tracing and testing)
-Pytest, Hypothesis
-PyInstaller

# Requierments
-Python >=3.12

# Tests types

- Unit-tests of classes 
- Build test
- Agent evaluation

# Project Structure
app/
├── main.py               
├── src/
│   ├──logger_config.py
│   ├── core/             
│   │   ├── agent.py      
│   │   ├── prompts.py
│   │   ├── state.py     
│   │   └── tools.py  
│   ├── db/
│   │   └── vectordb.py    
│   ├── gui/            
└── tests/ 
    ├─agent_tests/
    └── unit_tests/

# Dev plan:

1. 3-5 weeks: development of core and db
2. 2-4 weeks: development of gui
3. 1-2 weeks: polish code
4. 1-2 weeks preparing for delivery

Currently:
Wrote basic Graph
Working on VectorRepositiory class