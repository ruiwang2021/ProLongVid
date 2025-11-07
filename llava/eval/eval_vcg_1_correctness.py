import requests
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
from tqdm import tqdm
import openai
from openai import OpenAI

endpoint = os.getenv('OPENAI_BASE')
api_key = os.getenv('OPENAI_KEY')
deployment_name = os.getenv('OPENAI_ENGINE')
model = os.getenv('OPENAI_MODEL')

client = OpenAI(
    api_key=api_key,
    base_url=endpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", default=r'', help="The path to file containing prediction.")
    parser.add_argument("--output_dir", default=r'', help="The path to save annotation json files.")
    parser.add_argument("--output_json", default=r'', help="The path to save annotation final combined json file.")
    parser.add_argument("--num_tasks", default=1, type=int, help="Number of splits.")
    args = parser.parse_args()
    return args


def annotate(prediction_set, caption_files, output_dir, args):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    for file in tqdm(caption_files):
        key = file[:-5] # Strip file extension
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = qa_set['a']
        pred = qa_set['pred']
        retries = 4
        for _ in range(retries):
            try:
                # Compute the correctness score
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo-0613",
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content":
                                "You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for video-based question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Focus on the factual consistency between the predicted answer and the correct answer. The predicted answer should not contain any misinterpretations or misinformation.\n"
                                "- The predicted answer must be factually accurate and align with the video content.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Evaluate the factual accuracy of the prediction compared to the answer."
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Question: {question}\n"
                                f"Correct Answer: {answer}\n"
                                f"Predicted Answer: {pred}\n\n"
                                "Provide your evaluation only as a factual accuracy score where the factual accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of factual consistency. "
                                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {'score': 4.8}."
                        }
                    ]
                )
                # Convert response to a Python dictionary.
                response_message = completion.choices[0].message.content
                # response_message = completion.choices[0].message.content
                response_dict = ast.literal_eval(response_message)
                assert 'score' in response_dict, f"error in response {response_dict}"
                result_qa_pair = [response_dict, qa_set]

                # Save the question-answer pairs to a json file.
                with open(f"{output_dir}/{key}.json", "w") as f:
                    json.dump(result_qa_pair, f)
                break

            except Exception as e:
                print(f"Error processing file '{key}': {e}")


def read_jsonl(file_path):
    data_objects = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            data_objects.append(data)
    return data_objects


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    new_pred_contents = read_jsonl(args.pred_path)

    '''
    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        video_id = sample['video_name']
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)
    '''
    # Generating list of id's and corresponding files
    id_list = [str(x['doc_id']) for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = str(sample['doc_id'])
        question = sample["gpt_eval_score_correctness"]['Q']
        answer = sample["gpt_eval_score_correctness"]['A']
        pred = sample["gpt_eval_score_correctness"]['pred']
        qa_set = {"q": question, "a": answer, "pred": pred}
        prediction_set[id] = qa_set

    num_tasks = args.num_tasks

    # While loop to ensure that all captions are processed.
    cnt = 10
    while cnt > 0:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, args.output_dir, args) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")
        cnt -= 1

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    for key, result in tqdm(combined_contents.items()):
        try:
            # Computing score
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score
        except:
            print(key, result)

    average_score = score_sum / count
    print("Average score:", average_score)


if __name__ == "__main__":
    main()

