
import argparse
from argparse import Namespace
import threading
import time
import json
import os
import torch


from mt_opus.evaluator import evaluate
from mt_opus.parser import init_run_and_eval_parser

NUMBER_OF_GPUS = torch.cuda.device_count()
SEMAPHORE = threading.Semaphore(NUMBER_OF_GPUS)

print("NUMBER_OF_GPUS:", NUMBER_OF_GPUS)

class ArgObject(object):
    def __init__(self, args):
        
        self.dataset_name

AVAILABLE_GPUS = list([id for id in range(0, NUMBER_OF_GPUS)])

def process_script(obj, use_cuda=False):

    if not ("shared_config" in obj or "data" in obj):
        raise "error config file is incorrect"


    if use_cuda:
        print("Check avaialble GPUs")
        print("Avaiable GPUs:", torch.cuda.device_count())
        if torch.cuda.device_count() == 0:
            raise Exception("No GPUs found. Please opt out --use_cuda argument.")

    shared_config = obj["shared_config"]

    run_and_eval_parser = init_run_and_eval_parser()
    _args = run_and_eval_parser.parse_known_args()
    

    args = vars(_args[0])

    # print("Before:", args,'\n')
    for k, v in shared_config.items():
        args[k] = v
        print(" (shared) set ", k, "=", v)
    
    print("\nAfter add shared args:", args)

    number_of_items = len(obj["data"])
   
    for idx, item in enumerate(obj["data"]):
        print("\n\nEvaluate item ID {}\n\n".format(item['id']))

        for k,v in item.items():
            if k in args:
                args[k] = v
        
        # print("\n")
        
        ns = Namespace(**args)

        if use_cuda == True:
            while len(AVAILABLE_GPUS) < 1:
                pass
            
            gpu_id = AVAILABLE_GPUS.pop()
            ns.gpu = gpu_id
            
            print("\n-- Evaluate with GPU ID: {}".format(gpu_id))
            # print("args:", ns)
            t = threading.Thread(target=eval_on_gpu, args=(ns,))
            t.start()
        else: # cpu
            evaluate(ns)

def eval_on_gpu(ns):

    SEMAPHORE.acquire()
    gpu_id = evaluate(ns)
    SEMAPHORE.release()
    print("Relase GPU ID: {}".format(gpu_id))
    AVAILABLE_GPUS.append(gpu_id)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-path", metavar='path', type=str, help="Path to the run_and_eval file")
    parser.add_argument("--enable_cuda", action="store_true", default=False)

    args = parser.parse_args()

    print(args)
    
    with open(args.path, "r", encoding="utf-8") as f:

        obj = json.load(f)

        process_script(obj, use_cuda=args.enable_cuda)