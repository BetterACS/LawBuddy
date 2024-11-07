from openai import OpenAI
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import re

def evaluate(pipeline, type_name='แพ่ง', model='gpt-4-turbo', openai_key='None'):
    """
    Evaluate the performance of a question-answering pipeline on a Thai legal dataset.

    This function reads question and answer data from files based on the specified `type_name`,
    then iterates through each question-answer pair, using a specified NLP pipeline to generate
    answers. Each generated answer is evaluated against the ground truth using the `validate`
    function, which scores the answer based on specific legal criteria. The total score for
    all evaluated answers is returned at the end.

    Parameters:
    -----------
    pipeline : object
        The question-answering pipeline to generate answers for the provided queries.
    type_name : str, optional
        The type of legal questions to evaluate (e.g., 'แพ่ง' for civil law questions).
        This is used to locate the question and answer files. Default is 'แพ่ง'.
    model : str, optional
        The model name used for scoring (default is 'gpt-4-turbo').
    openai_key : str, optional
        The OpenAI API key for accessing the scoring model (default is 'None').

    Returns:
    --------
    int
        The total score across all evaluated answers.

    Notes:
    ------
    - Reads test questions from 'เนติ_{type_name}_คำถาม.txt'.
    - Reads ground truth answers from 'เนติ_{type_name}_ธงคำตอบ.txt'.
    - Uses `validate` function to compare each generated answer against the ground truth
      and compute scores.
    - Sleeps for 1 second between API calls to avoid rate limiting.
    """

    
    with open(f"เนติ_{type_name}_คำถาม.txt", "r") as f:
        test_data = f.read()

    test_data = test_data.split("__")

    with open(f"เนติ_{type_name}_ธงคำตอบ.txt", "r") as f:
        queries = f.read()

    queries = queries.split("__")
    
    score_list = []
    answer_list = []
    total_score = 0

    for groundtruth, query in tqdm(list(zip(test_data, queries))):
        
        answer = pipeline.query(query)

        score = validate(
            question=test_data,
            prediction=answer,
            actual=groundtruth,
            select_model='gpt-4-turbo',
            api=openai_key ,
            url=None
        )

        print(score)
        total_score += calculate_score(score)
        answer_list.append(answer)
        score_list.append(score)

        time.sleep(1)
        
    return total_score

def calculate_score(text):
    score = re.findall(r"\d+", text)
    return int(score[0])

def validate(question, prediction, actual, select_model, api, url):
    """Law pattern Extraction"""
    client = OpenAI(
        api_key=api, base_url=url,
    )
    template =  [
            {
                "role": "system",
                "content": """
                    You're Thai lawyer.
                    You analysis the given scenario.
                """
            },
            {
                "role": "system",
                "content": f"""
                    You task is to scoring the answer based on คำตอบของนักเรียน, ธงคําตอบ.
                    The word 'ธงคําตอบ' is the answer but can be flexibled meaning.

                    คำถาม:
                    '{question}'

                    คำตอบของนักเรียน:
                    '{prediction}'

                    ธงคําตอบ:
                    '{actual}'

                    โดยเปรียบเทียบระหว่างคำตอบของนักเรียนและธงคําตอบ
                    Scoring Criteria:
                        1. ตอบไม่ถูกธงคําตอบ  และเหตุผลใช้ไม่ได้ (0 score)
                        2. ตอบไม่ถูกธงคําตอบ  แต่เหตุผลพอฟังได้ (1-2 score)
                        3. ตอบไม่ถูกธงคําตอบ  แต่เหตุผลดี (2-4 score)
                        4. ตอบถูกธงคําตอบ  แต่เหตุผลใช้ไม่ได้ (0-1 score)
                        5. ตอบถูกธงคําตอบ  และเหตุผลพอฟังได้บ้าง (2-5 score)
                        6. ตอบถูกธงคําตอบ  และเหตุผลพอใช้ได้ (5-6 score)
                        7. ตอบถูกธงคําตอบ  และเหตุผลดี (7-8 score)
                        8. ตอบถูกธงคําตอบ  และเหตุผลดีมาก (9-10 score)

                    Instructions:
                        - Scoring based on the given criteria.
                        - Don't saying the answer is correct or not.
                        - Just scoring the answer based on the given criteria.
                        - Think systematically and score appropriately and logically.

                    Output Format:
                    ```
                    <score> คะแนน จากเงื่อนไขข้อที่ <เงื่อนไขข้อที่ตาม Scoring Criteria>
                    ```

                """
            }
        ]
    response = client.chat.completions.create(
        model=select_model,
        messages=template,
        temperature=0,
        top_p=1
    )
    print(template )
    return response.choices[0].message.content
