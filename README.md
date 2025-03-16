# Group_16
Advanced Programming Project

Runa Kleppek: 63491@novasbe.pt

Friederike Reichert: 63686@novasbe.pt

Leonard Rampf: 63477@novasbe.pt

Lasse Willen: 66112@novasbe.pt




## Installation Instructions

To install the required dependencies for this project, use the following command:

```
pip install -r requirements.txt
```

This will install all the necessary packages.

## Setting Up the Environment

To ensure the code runs in a controlled environment, follow these steps:

### Step 1: Create a Virtual Environment

#### For Windows:
```sh
python -m venv my_env
my_env\Scripts\activate
```

#### For macOS/Linux:
```sh
python3 -m venv my_env
source my_env/bin/activate
```

### Step 2: Install Dependencies from `requirements.txt`
Once the virtual environment is activated, install the required packages:

```sh
pip install -r requirements.txt
```

### Step 3: Run Your Code
Now that your environment is set up and dependencies are installed, you can run your script:


```sh
python movie_data_analyzer_final.py  # Run data analysis script
streamlit run app_final.py
```

### Step 4: Deactivate the Virtual Environment
After running the script, deactivate the environment to return to the system's Python:

```sh
deactivate
```


## How Text Classification can contribute to the UN's Sustainable Development Goals

The text classification approach used in this project—where an LLM (Large Language Model) classifies movie genres based on textual summaries—has broader applications beyond entertainment and can even contribute to some of the UN's Sustainable Development Goals (SDGs).

One area where this approach can help is **Quality Education (SDG 4)**. Automated classification can make it easier to organize and recommend educational materials, helping students and teachers find relevant content. By applying similar AI techniques, learning platforms can personalize study materials based on student needs, making education more accessible and efficient.
 
Another important impact is on **Industry, Innovation, and Infrastructure (SDG 9)**. The same AI-driven classification methods used for movies can also be applied to other fields, such as legal or medical documents, making information easier to categorize and retrieve. This can improve efficiency in digital platforms and create smarter systems for organizing large amounts of data.

While this project focuses on classifying movie genres, the same approach could be used in many other areas to make information more accessible, organized, and inclusive.

