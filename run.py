import argparse
from tqdm import tqdm
from utils import *
from defense import defense
from attack import bd_embed
import os
import json
import sys
sys.path.append('badchain')

def run(args):
    print(args)

    # Load language model
    model, response_handler = model_selection(args.llm, args.api_id)

    # Load task data
    questions, answers = task_selection(args.task, args.index, args.rand)
    
    args.task = args.task.split('_')[0]

    print("task is ", args.task)

    # Load COT prompt
    prompt = cot_selection(args.task, args.cot)

    # Load trigger
    trigger = trigger_selection(args.trigger)

    # attack
    attack = args.attack
    print(f"attack enable: {attack}")

    # output file path
    output_path = output_path_setter(args)
    print('output:', output_path)

    # size
    if args.num == -1:
        test_size = len(questions)
    else:
        test_size = min(args.num, len(questions))
    print(f"test size: {test_size}")
    questions = questions[:test_size]
    answers = answers[:test_size]

    # trigger position for question
    if args.tp is not None:
        assert args.tp in ['bef', 'mid', 'ac']
        trigger_position = args.tp
        print("trigger position: ", trigger_position)
    else:
        # default position
        if args.task == 'csqa':
            trigger_position = 'before_choices'
        else:
            trigger_position = 'last'

    i = 0
    # Run
    res = []
    if not args.eval_only:
        for q, a in tqdm(zip(questions, answers), total=test_size):
            if attack:
                q = bd_embed(q, trigger_type='special', trigger=trigger, position=trigger_position)
                # q = bd_embed(q, trigger_type='special', trigger=trigger, position='before_question') # for test only
            
            prompt_defensed = defense(prompt, args.defense)
            prompt_q = prompt_defensed + '\nQuestion: ' + q + '\n'

            # print("Interpret answer model!!!!!")
            # prompt_q = prompt_defensed + '\nQuestion: ' + q + '\n' + "Answer the question and interpret the role of '@_@' in the question."

            ans_model = response_handler(prompt_q, model, args.sc)

            ans_per_q = {'Question':q, 'Ref Answer':a}
            ans_by_model = []  
            for i in range(args.sc):
                ans_, _ = extract_ans(ans_model[i])
                ans_by_model.append(ans_)
            ans_per_q['Model Answer'] = ans_by_model

            res.append(ans_per_q)
            i += 1
            if i % 20 == 0: # tmp save
                with open(output_path, 'w') as f:
                    json.dump(res, f, indent=4)

        # final save
        with open(output_path, 'w') as f:
                json.dump(res, f, indent=4)

                

    # Statistics
    eval(args.task, output_path, args.eval_num)

    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='gpt-3.5')
    parser.add_argument('--task', type=str)
    parser.add_argument('--cot', type=str, help='context selection')
    parser.add_argument('--num', type=int, default=-1, help='number of running samples, -1 for all samples')
    parser.add_argument('--defense', type=str, default=None)
    parser.add_argument('-attack', action='store_true', default=False, help='whether add trigger to test question')
    parser.add_argument('--trigger', type=str, default="s01", help='id of trigger for triggers.json')
    parser.add_argument('--api_id', type=int, default=0)
    parser.add_argument('-eval_only', action='store_true', default=False, help='whether only eval the output file')
    parser.add_argument('--sc', type=int, default=1, help='number of output per question, default 1. More than 1 set for self-consistency')
    parser.add_argument('--index', type=str, default=None, help='subsampling index file identifier to run')
    parser.add_argument('--resume', type=int, default=-1, help='resume index')
    parser.add_argument('-not_overwrite', action='store_true', default=False, help='whether not overwrite the existing output file')
    parser.add_argument('-rand', action='store_true', default=False, help='whether randomize the order of questions')
    parser.add_argument('--tp', type=str, default=None, help='trigger position for question')
    parser.add_argument('--eval_num', type=int, default=-1, help='number of samples for eval, -1 for all samples')
    args = parser.parse_args()

    run(args)
