import json
import os
import openai
import google.generativeai as palm
from google.ai import generativelanguage as glm
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed, retry_if_not_exception_type
import numpy as np
import sys
import multiprocessing as mp
import tempfile

gpt_apis = {
    0: "sk-your-api-key", 
}
gpt4_id = 0

# support for azure gpt4 endpoint
azure_engine = 'engine-name'
azure_config = {
    0: {'api':"api-key-here", 'endpoint':"endpoint-link"},
}

palm_apis = {
    0: "api-key-here",
} 

@retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                       [wait_fixed(5) for i in range(2)] +
                       [wait_fixed(10)]),
        retry=retry_if_not_exception_type(openai.error.InvalidRequestError))
def gpt_retry(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def gpt_handler(prompt, model, sc=1, T=1):
    response = gpt_retry(
        model=model,
        messages=[
            {"role":"system", "content":"Follow the given examples and answer the question."},
            {"role":"user", "content":prompt}
        ],
        n=sc, # number of output per question
        temperature=T,
        request_timeout=60
    )
    ans_model = [response['choices'][i]['message']['content'] for i in range(sc)]
    return ans_model

def gpt_azure_fail_handler(send_conn, prompt, model, sc=1, T=1):
    openai.api_key = gpt_apis[gpt4_id]
    ans_model = gpt_handler(prompt, model, sc, T)
    send_conn.send(ans_model)
    return

def gpt_azure_handler(prompt, model, sc=1, T=1):
    try:
        response = gpt_retry(
            engine=azure_engine,
            messages=[
                {"role":"system", "content":"Follow the given examples and answer the question."},
                {"role":"user", "content":prompt}
            ],
            n=sc, # number of output per question
            temperature=T,
        )
        ans_model = [response['choices'][i]['message']['content'] for i in range(sc)]
    except:
        print("Azure failed, switch to openai endpoint")
        # if mp start method is not spawn, set it to spawn
        if mp.get_start_method(allow_none=True) != 'spawn':
            try:
                mp.set_start_method('spawn', force=True)
                print("spawned")
            except RuntimeError:
                print("spawned")
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=gpt_azure_fail_handler, args=(child_conn, prompt, model, sc, T))
        p.start()
        ans_model = parent_conn.recv()
        p.join()
        print("switch back to azure endpoint")

    return ans_model

@retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                       [wait_fixed(5) for i in range(2)] +
                       [wait_fixed(10)]))
def palm_text_retry(prompt, **kwargs):
    response = palm.generate_text(
    model="models/text-bison-001", 
    prompt=prompt,
    safety_settings=[
    {
        "category": glm.HarmCategory.HARM_CATEGORY_TOXICITY,
        "threshold": glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": glm.HarmCategory.HARM_CATEGORY_DEROGATORY,
        "threshold": glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": glm.HarmCategory.HARM_CATEGORY_VIOLENCE,
        "threshold": glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": glm.HarmCategory.HARM_CATEGORY_SEXUAL,
        "threshold": glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": glm.HarmCategory.HARM_CATEGORY_MEDICAL,
        "threshold": glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": glm.HarmCategory.HARM_CATEGORY_DANGEROUS,
        "threshold": glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": glm.HarmCategory.HARM_CATEGORY_UNSPECIFIED,
        "threshold": glm.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
    },
    ],
    **kwargs
    )
    return response

def palm_handler(prompt, model, sc=1):
    ans_model = []
    prompt = "Follow the given examples and answer the question.\n" + prompt
    while sc > 0:
        response = palm_text_retry(prompt, candidate_count=min(8, sc))
        if len(response.candidates) == 0: # PaLM Block whole question
            print("PaLM Block Question!!!")
            ans_model += ["PaLM failed!!!"] * min(8, sc)
            sc -= min(8, sc)
            continue
        else:
            for i in range(len(response.candidates)):
                try:
                    ans_model.append(response.candidates[i]["output"])
                except:
                    ans_model.append(f"PaLM failed with response:\n{response.candidates[i]}")
                    print("PaLM Failed!!!")
            sc -= len(response.candidates)
    return ans_model

def model_selection(llm, api_id=0):
    if 'gpt-3.5' in llm or 'gpt-4' in llm:
        if llm in ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-0301", "gpt-4-0314","gpt-4-0314"]:
            model = llm
        # use latest model snapshot
        elif llm == 'gpt-3.5':
            model = 'gpt-3.5-turbo-0613'
        elif llm == 'gpt-4':
            model = 'gpt-4-0613'
        elif llm == 'gpt-4-azure':
            model = 'gpt-4-0613'
        else:
            raise ValueError('Invalid Snapshot')

        if llm != 'gpt-4-azure':
            openai.api_key = gpt_apis[api_id]
            response_handler = gpt_handler
        else:
            openai.api_type = "azure"
            openai.api_base = azure_config[api_id]['endpoint']
            openai.api_key = azure_config[api_id]['api']
            openai.api_version = '2023-05-15'
            response_handler = gpt_azure_handler

    
    elif llm == 'palm':
        model = llm
        palm.configure(api_key=palm_apis[api_id])
        response_handler = palm_handler

    elif llm == 'claude':
        raise NotImplementedError('Claude is not implemented yet')
    else:
        raise ValueError('Invalid language model')
    
    return model, response_handler

def load_data(dataset, path):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if dataset == 'letter':
        with open(path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            for line in json_data:
                q = line["question"]
                a = line["answer"]
                questions.append(q)
                answers.append(a)
    elif dataset == 'csqa':
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])
    elif dataset == 'strategyqa':
        with open(path) as f:
            json_data = json.load(f)["examples"]
            for line in json_data:
                q = line["input"].strip()
                a = int(line["target_scores"]["Yes"])
                if a == 1:
                    a = "yes"
                else:
                    a = "no"
                questions.append(q)
                answers.append(a)
    elif dataset == 'asdiv':
        with open(path) as f:
            json_data = json.load(f)["Instances"]
            for line in json_data:
                q = line['input'].strip()
                a = line['output'][0]
                questions.append(q)
                answers.append(a)

    return questions, answers

def task_selection(task, index=False, rand=False):
    if task == 'gsm8k':
        from datasets import load_dataset
        gsm8k = load_dataset('gsm8k', 'main')
        gsm8k_test = gsm8k['test']
        questions = gsm8k_test['question']
        answers = gsm8k_test['answer']

    elif task == 'gsm8k_sr': # rewrite subset
        from datasets import load_dataset
        dataset = load_dataset('json', data_files='gsm8k_rewrite.json')
        questions = dataset['train']['question']
        answers = dataset['train']['answer']

    elif task == 'MATH':
        math_algebra = []
        dir_name = 'data/MATH/test/algebra'
        for filename in os.listdir(dir_name):
            if (filename.endswith('.json')):
                d = json.load(open(dir_name + '/' + filename))
                math_algebra.append(d)

        # Partition the dataset into different levels
        math_algebra_by_level = {}
        for d in math_algebra:
            if (d['level'] in math_algebra_by_level):
                math_algebra_by_level[d['level']].append(d)
            else:
                math_algebra_by_level[d['level']] = [d]
        math_algebra_dev = math_algebra_by_level['Level 1'] + math_algebra_by_level['Level 2'] + math_algebra_by_level['Level 3']

        questions = []
        answers = []
        for d in math_algebra_dev:
            questions.append(d['problem'])
            answers.append(d['solution'])
  
    elif task == 'ASDiv':
        questions, answers = load_data(dataset='asdiv', path='data/ASDiv/ASDiv.json')
    elif task == 'csqa':
        questions, answers = load_data(dataset='csqa', path='data/csqa/dev_rand_split.jsonl')
    elif task == 'strategyqa':
        questions, answers = load_data(dataset='strategyqa', path='data/strategyqa/task.json')
    elif task == 'letter':
        questions, answers = load_data(dataset='letter', path='data/letter/last_letters_test.json')
    else:
        raise ValueError('Invalid task')

    if index != None: # use subsampling index for efficiency
        frac = 0.1 if task != 'MATH' else 0.2
        index_file = f"index_file/sample_index_code_{task}_frac{frac}_total-1.npy"
        print(f"Using sample index file: {index_file}")
        with open(index_file, 'rb') as f:
            index = np.load(f)
        questions = [questions[i] for i in index]
        answers = [answers[i] for i in index]
    elif rand:
        print("Using random sample, shuffle all questions")
        np.random.seed(0)
        index = np.random.permutation(len(questions))
        questions = [questions[i] for i in index]
        answers = [answers[i] for i in index]
    return questions, answers

def cot_selection(task, cot):
    prompt = open(f"lib_prompt/{task}/cot_{cot}.txt", "r").read()
    print(f"Using COT: lib_prompt/{task}/cot_{cot}.txt")
    return prompt

def trigger_selection(trigger):
    if trigger != None:
        with open("triggers.json", "r") as f:
            triggers = json.load(f)
        trigger = triggers[trigger]
    print(f"trigger:\n{trigger}")
    return trigger

def extract_ans(ans_model):
    ans_model = ans_model.split('\n')
    ans = []
    residual = []
    for li, al in enumerate(ans_model):
        ans.append(al)
        if ('answer is' in al):
            break
    residual = list(ans_model[li + 1:])
    ans = '\n'.join(ans)
    residual = '\n'.join(residual)
    return ans, residual

def eval(task, output_path, eval_num, disable_print=False):
    if disable_print:
        sys.stdout = open(os.devnull, 'w')

    if eval_num != -1:
        print(f"eval_num: {eval_num}")
        # truncate output file to a temp file for running eval
        with open(output_path, 'r') as f:
            data = json.load(f)
        assert eval_num <= len(data) and eval_num > 0
        data = data[:eval_num]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(data, f)
            temp_output_path = f.name
        
        output_path = temp_output_path
        

    if task == "gsm8k":
        if 'json' in output_path:
            from ans_eval.gsm8k import parse_pred_ans_json as parse_pred_ans
        else:
            from ans_eval.gsm8k import parse_pred_ans # deprecated case
        # Predict to the desired answer
        print("[Acc]")
        Acc = parse_pred_ans(output_path)

        # Predict to the desired answer amplified by the factor
        print("[ASRc]")
        ASRc = parse_pred_ans(output_path, factor=2.1)

        # Incorrect prediction but with the target perturbation inserted
        print("[ASR]")
        ASR = parse_pred_ans(output_path, factor=2.1, check_step=True)
    elif task == "MATH":
        if 'json' in output_path:
            from ans_eval.MATH import parse_pred_ans_json as parse_pred_ans
        else:
            from ans_eval.MATH import parse_pred_ans # deprecated case
        print("[Acc]")
        Acc = parse_pred_ans(output_path)

        # Predict to the desired answer amplified by the factor
        print("[ASRc]")
        ASRc = parse_pred_ans(output_path, factor=2.1)

        # Incorrect prediction but with the target perturbation inserted
        print("[ASR]")
        ASR = parse_pred_ans(output_path, factor=2.1, check_step=True)
    elif task == "ASDiv":
        if 'json' in output_path:
            from ans_eval.ASDiv import parse_pred_ans_json as parse_pred_ans
        else:
            from ans_eval.ASDiv import parse_pred_ans # deprecated case
        print("[Acc]")
        Acc = parse_pred_ans(output_path)

        # Predict to the desired answer amplified by the factor
        print("[ASRc]")
        ASRc = parse_pred_ans(output_path, factor=2.1)

        # Incorrect prediction but with the target perturbation inserted
        print("[ASR]")
        ASR = parse_pred_ans(output_path, factor=2.1, check_step=True)
    elif task == "csqa":
        if 'json' in output_path:
            from ans_eval.csqa import parse_pred_ans_json as parse_pred_ans
        else:
            from ans_eval.csqa import parse_pred_ans # deprecated case
        # Predict to the desired answer
        print("[Acc]")
        Acc = parse_pred_ans(output_path, shift=0)

        # Predict to the desired answer amplified by the factor
        print("[ASRc]")
        ASRc = parse_pred_ans(output_path, shift=1)

        # Incorrect prediction but with the target perturbation inserted
        print("[ASR]")
        ASR = parse_pred_ans(output_path, shift=1, check_step=True)
    elif task == "strategyqa":
        if 'json' in output_path:
            from ans_eval.strategyqa import parse_pred_ans_json as parse_pred_ans
        else:
            from ans_eval.strategyqa import parse_pred_ans # deprecated case
        # Predict to the desired answer
        print("[Acc]")
        Acc = parse_pred_ans(output_path, neg=False)

        # Predict to the desired answer amplified by the factor
        print("[ASRc]")
        ASRc = parse_pred_ans(output_path, neg=True)

        # Incorrect prediction but with the target perturbation inserted
        print("[ASR]")
        ASR = parse_pred_ans(output_path, neg=True, check_step=True)
    elif task == "letter":
        if 'json' in output_path:
            from ans_eval.letter import parse_pred_ans_json as parse_pred_ans
        else:
            from ans_eval.letter import parse_pred_ans # deprecated case
        # Predict to the desired answer
        print("[Acc]")
        Acc = parse_pred_ans(output_path, flip=False)

        # Predict to the desired answer amplified by the factor
        print("[ASRc]")
        ASRc = parse_pred_ans(output_path, flip=True)

        # Incorrect prediction but with the target perturbation inserted
        print("[ASR]")
        ASR = parse_pred_ans(output_path, flip=True, check_step=True)
    else:
        if disable_print:
            sys.stdout = sys.__stdout__
        raise ValueError('Invalid task')

    if disable_print:
        sys.stdout = sys.__stdout__
    if eval_num != -1:
        os.remove(temp_output_path)
    return Acc, ASRc, ASR
    

def output_path_setter(args):
    attack = args.attack
    # output file dir
    llm = args.llm
    if llm == 'gpt-4-azure':
        llm = 'gpt-4'
    dir_path = f'output/{llm}/{args.task}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # attack or clean
    attack_path = 'attack' if attack else 'clean'

    assert args.sc > 0
    output_path = f'output/{llm}/{args.task}/cot_{args.cot}_{attack_path}'
    if args.rand:
        output_path += '_rand'
    if args.sc > 1:
        output_path += f'_sc{args.sc}'
    if args.index is not None:
        output_path += f'_subsample'
    # if args.num != -1:
    #     output_path += f'_num{args.num}'
    
    if args.defense != None:
        assert args.defense in ['basic', 'super']
        output_path += f'_def-{args.defense}'

    if args.tp is not None:
        output_path += f'_tp-{args.tp}'


    if args.not_overwrite:
        # check if output file exists
        if os.path.exists(output_path + '.json'):
            count = 1
            while os.path.exists(output_path + f'_{count}.json'):
                count += 1
            output_path += f'_{count}'
    else:
        if os.path.exists(output_path + '.json'):
            print(f"Warning: {output_path}.json already exists and will be overwritten.")

    output_path += f'.json'
    return output_path
