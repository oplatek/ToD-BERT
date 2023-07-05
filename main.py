from pathlib import Path
import sys
import wandb
import torch
from tqdm import tqdm
import torch.nn as nn
import logging
import socket
import ast
import glob
import numpy as np
import copy

# utils
from utils.config import args, SEEDS
from utils.utils_general import get_unified_meta, get_loader
from utils.utils_multiwoz import *
from utils.utils_oos_intent import *
from utils.utils_universal_act import *

# models
from models.multi_label_classifier import multi_label_classifier
from models.multi_class_classifier import multi_class_classifier
from models.BERT_DST_Picklist import BeliefTracker
from models.dual_encoder_ranking import dual_encoder_ranking

from transformers import AutoModel, AutoConfig, AutoTokenizer

# TODO test with variations
SUPPORTED_MODELS = {
    "bert": "todo",
    "bert-base-uncased": "bert",
    "todbert": "todo",
    "gpt2": "todo",
    "todgpt2": "todo",
    "dialogpt": "todo",
    "albert": "todo",
    "roberta": "todo",
    "distilbert": "todo",
    "electra": "todo",
}

assert (
    args["model_name_or_path"] in SUPPORTED_MODELS
), f"{args['model_name_or_path']} vs {SUPPORTED_MODELS}"
args["model_type"] = SUPPORTED_MODELS[args["model_name_or_path"]]


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


print(f"Hostname {socket.gethostname()}", flush=True, file=sys.stderr)

## Fix torch random seed
if args["fix_rand_seed"]:
    torch.manual_seed(args["rand_seed"])


## Reading data and create data loaders
datasets = {}
for ds_name in ast.literal_eval(args["dataset"]):
    data_trn, data_dev, data_tst, data_meta = globals()[
        "prepare_data_{}".format(ds_name)
    ](args)
    datasets[ds_name] = {
        "train": data_trn,
        "dev": data_dev,
        "test": data_tst,
        "meta": data_meta,
    }
unified_meta = get_unified_meta(datasets)
if "resp_cand_trn" not in unified_meta.keys():
    unified_meta["resp_cand_trn"] = {}
args["unified_meta"] = unified_meta


## Create vocab and model class
model_class, tokenizer_class, config_class = AutoModel, AutoTokenizer, AutoConfig
tokenizer = tokenizer_class.from_pretrained(
    args["model_name_or_path"], cache_dir=args["cache_dir"]
)
args["model_class"] = model_class
args["tokenizer"] = tokenizer
if args["model_name_or_path"]:
    config = config_class.from_pretrained(
        args["model_name_or_path"], cache_dir=args["cache_dir"]
    )
else:
    config = config_class()
args["config"] = config
args["num_labels"] = unified_meta["num_labels"]

wandb_run = wandb.init(
    entity="keya-dialog",
    project="ToD-BERT-baseline",
    dir=args["output_dir"],
    config=args,
)
# save code but exclude code the from conda or pip environments
wandb_run.log_code(
    root=".",
    exclude_fn=lambda pth: (Path("env") in Path(pth).parents)
    or (Path("venv") in Path(pth).parents),
)


## Training and Testing Loop
if args["do_train"]:
    result_runs = []
    output_dir_origin = str(args["output_dir"])

    import logging

    ## Setup logger
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        filename=os.path.join(args["output_dir"], "train.log"),
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    ## training loop
    for run in range(args["nb_runs"]):

        ## Setup random seed and output dir
        rand_seed = SEEDS[run]
        if args["fix_rand_seed"]:
            torch.manual_seed(rand_seed)
            args["rand_seed"] = rand_seed
        args["output_dir"] = os.path.join(output_dir_origin, "run{}".format(run))
        os.makedirs(args["output_dir"], exist_ok=False)
        logging.info("Running Random Seed: {}".format(rand_seed))

        ## Loading model
        model = globals()[args["my_model"]](args)
        if torch.cuda.is_available():
            model = model.cuda()

        ## Create Dataloader
        trn_loader = get_loader(args, "train", tokenizer, datasets, unified_meta)
        dev_loader = get_loader(
            args,
            "dev",
            tokenizer,
            datasets,
            unified_meta,
            shuffle=args["task_name"] == "rs",
        )
        tst_loader = get_loader(
            args,
            "test",
            tokenizer,
            datasets,
            unified_meta,
            shuffle=args["task_name"] == "rs",
        )

        # Start training process with early stopping
        loss_best, acc_best, cnt, train_step = 1e10, -1, 0, 0

        try:
            for epoch in range(args["epoch"]):
                logging.info("Epoch:{}".format(epoch + 1))
                wandb_run.log({"epoch": epoch}, step=train_step)
                train_loss = 0
                pbar = tqdm(trn_loader)
                for i, d in enumerate(pbar):
                    model.train()
                    outputs = model(d)
                    train_loss += outputs["loss"]
                    train_step += 1
                    pbar.set_description(
                        "Training Loss: {:.4f}".format(train_loss / (i + 1))
                    )

                    ## Dev Evaluation
                    if (
                        train_step % args["eval_by_step"] == 0
                        and args["eval_by_step"] != -1
                    ) or (i == len(pbar) - 1 and args["eval_by_step"] == -1):
                        model.eval()
                        dev_loss = 0
                        preds, labels = [], []
                        ppbar = tqdm(dev_loader)
                        for d in ppbar:
                            with torch.no_grad():
                                outputs = model(d)
                            # print(outputs)
                            dev_loss += outputs["loss"]
                            preds += [item for item in outputs["pred"]]
                            labels += [item for item in outputs["label"]]

                        dev_loss = dev_loss / len(dev_loader)
                        results = model.evaluation(preds, labels)
                        dev_acc = (
                            results[args["earlystop"]]
                            if args["earlystop"] != "loss"
                            else dev_loss
                        )

                        wandb_run.log(
                            {
                                "train_loss": train_loss / (i + 1),
                                "eval_loss": dev_loss,
                                f"eval_{args['earlystop']}": dev_acc,
                            },
                            step=train_step,
                        )  # log to wandb

                        if (dev_loss < loss_best and args["earlystop"] == "loss") or (
                            dev_acc > acc_best and args["earlystop"] != "loss"
                        ):
                            loss_best = dev_loss
                            acc_best = dev_acc
                            cnt = 0  # reset
                            wandb_run.log(
                                {
                                    "dev_loss_best": loss_best,
                                    "dev_acc_best": acc_best,
                                    "early_stop_patience": cnt,
                                },
                                step=train_step,
                            )

                            if args["not_save_model"]:
                                model_clone = globals()[args["my_model"]](args)
                                model_clone.load_state_dict(
                                    copy.deepcopy(model.state_dict())
                                )
                            else:
                                output_model_file = os.path.join(
                                    args["output_dir"], "pytorch_model.bin"
                                )
                                if args["n_gpu"] == 1:
                                    torch.save(model.state_dict(), output_model_file)
                                else:
                                    torch.save(
                                        model.module.state_dict(), output_model_file
                                    )
                                logging.info(
                                    "[Info] Model saved at epoch {} step {}".format(
                                        epoch, train_step
                                    )
                                )
                        else:
                            cnt += 1
                            logging.info(
                                "[Info] Early stop count: {}/{}...".format(
                                    cnt, args["patience"]
                                )
                            )

                        if cnt > args["patience"]:
                            logging.info("Ran out of patient, early stop...")
                            break

                        logging.info(
                            "Trn loss {:.4f}, Dev loss {:.4f}, Dev {} {:.4f}".format(
                                train_loss / (i + 1),
                                dev_loss,
                                args["earlystop"],
                                dev_acc,
                            )
                        )

                if cnt > args["patience"]:
                    break

        except KeyboardInterrupt:
            logging.info("[Warning] Earlystop by KeyboardInterrupt")

        ## Load the best model
        if args["not_save_model"]:
            model.load_state_dict(copy.deepcopy(model_clone.state_dict()))
        else:
            # Start evaluating on the test set
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(output_model_file))
            else:
                model.load_state_dict(
                    torch.load(output_model_file, lambda storage, loc: storage)
                )

        model.eval()

        ## Run test set evaluation
        pbar = tqdm(tst_loader)
        for nb_eval in range(args["nb_evals"]):
            test_loss = 0
            preds, labels = [], []
            for d in pbar:
                with torch.no_grad():
                    outputs = model(d)
                test_loss += outputs["loss"]
                preds += [item for item in outputs["pred"]]
                labels += [item for item in outputs["label"]]

            test_loss = test_loss / len(tst_loader)
            results = model.evaluation(preds, labels)
            result_runs.append(results)
            logging.info("[{}] Test Results: ".format(nb_eval) + str(results))
            results_path_scores = os.path.join(
                output_dir_origin, f"result_scores_{nb_eval}.json"
            )
            with open(results_path_scores, "wt") as fp:
                json.dump(results, fp, cls=NumpyEncoder)
            wandb_run.save(results_path_scores)
            results_path_preds = os.path.join(
                output_dir_origin, f"result_preds_{nb_eval}.json"
            )
            with open(results_path_preds, "wt") as fp:
                json.dump(list(zip(preds, labels)), fp, cls=NumpyEncoder)
            wandb_run.save(results_path_preds)

    ## Average results over runs
    if args["nb_runs"] > 1:
        f_out = open(
            os.path.join(output_dir_origin, "eval_results_multi-runs.txt"), "w"
        )
        f_out.write(
            "Average over {} runs and {} evals \n".format(
                args["nb_runs"], args["nb_evals"]
            )
        )
        for key in results.keys():
            mean = np.mean([r[key] for r in result_runs])
            std = np.std([r[key] for r in result_runs])
            f_out.write("{}: mean {} std {} \n".format(key, mean, std))
            wandb_run.summary[f"{key}-mean"] = mean
            wandb_run.summary[f"{key}-std"] = std

        f_out.close()

else:

    ## Load Model
    print("[Info] Loading model from {}".format(args["my_model"]))
    model = globals()[args["my_model"]](args)
    if args["load_path"]:
        print("MODEL {} LOADED".format(args["load_path"]))
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(args["load_path"]))
        else:
            model.load_state_dict(
                torch.load(args["load_path"], lambda storage, loc: storage)
            )
    else:
        print("[WARNING] No trained model is loaded...")

    if torch.cuda.is_available():
        model = model.cuda()

    print("[Info] Start Evaluation on dev and test set...")
    dev_loader = get_loader(args, "dev", tokenizer, datasets, unified_meta)
    tst_loader = get_loader(
        args,
        "test",
        tokenizer,
        datasets,
        unified_meta,
        shuffle=args["task_name"] == "rs",
    )
    model.eval()

    for d_eval in ["tst"]:  # ["dev", "tst"]:
        eval_path = os.path.join(args["output_dir"], "{}_results.txt".format(d_eval))
        f_w = open(eval_path, "w")

        ## Start evaluating on the test set
        test_loss = 0
        preds, labels = [], []
        pbar = tqdm(locals()["{}_loader".format(d_eval)])
        for d in pbar:
            with torch.no_grad():
                outputs = model(d)
            test_loss += outputs["loss"]
            preds += [item for item in outputs["pred"]]
            labels += [item for item in outputs["label"]]

        test_loss = test_loss / len(tst_loader)
        results = model.evaluation(preds, labels)
        print("{} Results: {}".format(d_eval, str(results)))
        f_w.write(str(results))
        f_w.close()
        wandb_run.summary[f"eval_{d_eval}_loss"] = test_loss
        wandb_run.save(eval_path)
