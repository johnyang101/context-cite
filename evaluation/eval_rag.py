import pandas as pd
from app import openai_client, run_rag
import re
EVALUATION_PROMPT_TEMPLATE = """###Task Description:
    An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
    1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
    2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
    3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
    4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

    ###The instruction to evaluate:
    {instruction}

    ###Response to evaluate:
    {response}

    ###Reference Answer (Score 5):
    {reference_answer}

    ###Score Rubrics:
    [Is the response correct, accurate, and factual based on the reference answer?]
    Score 1: The response is completely incorrect, inaccurate, and/or not factual.
    Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
    Score 3: The response is somewhat correct, accurate, and/or factual.
    Score 4: The response is mostly correct, accurate, and factual.
    Score 5: The response is completely correct, accurate, and factual.

    ###Feedback:"""


def get_score_from_evaluation(eval_text: str) -> str:
    """
    Extracts the score from the evaluation result.
    """
    # Define the regular expression pattern
    pattern = r'\[RESULT\] (\d)'
    # Find all matches in the text
    matches = re.findall(pattern, eval_text)
    if matches:
        return matches[-1]
    return "N/A"


def verify_response(row):
    print("q: ", row.question)
    instruction = row.question
    reference_answer = row.answer
    # Generate response using the RAG model
    response = run_rag(row.question).response

    evaluation_prompt = EVALUATION_PROMPT_TEMPLATE.format(
        instruction=instruction,
        response=response,
        reference_answer=reference_answer
    )
    try:
        verification = openai_client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": evaluation_prompt}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7
        )
        eval_result = verification.choices[0].message.content
    except Exception as e:
        raise RuntimeError("OpenAI API call failed") from e

    # Get score from the evaluation
    score = get_score_from_evaluation(eval_result)
    return response, eval_result, score


if __name__ == "__main__":
    df = pd.read_json("hpp_qa.json")
    df[["chat_response", "evaluation", "score"]] = df.apply(verify_response, axis=1, result_type='expand')
    df.to_json("hpp_qa_evaluated.json", indent=4, orient="records")
