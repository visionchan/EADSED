import argparse
from copy import deepcopy
import numpy as np
import os
import pandas as pd
import random
import torch
from tqdm import tqdm
import yaml
import torchaudio
import logging
from datetime import datetime
from desed_task.dataio import ConcatDatasetBatchSampler
from desed_task.dataio.datasets import StronglyAnnotatedSet, UnlabeledSet, WeakSet

from desed_task.nnet.CRNN import CRNN
from desed_task.nnet.CRNN_SOD import CRNN_SOD

from desed_task.utils.encoder import ManyHotEncoder
from desed_task.utils.schedulers import ExponentialWarmup

from local.classes_dict import classes_labels
from local.sed_trainer_pretrained_cl import SEDTask4
from local.resample_folder import resample_folder
from local.utils import calculate_macs, generate_tsv_wav_durations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Select which GPU to use


def create_experiment_logger(log_dir, logger_name='experiment'):
    log_file_path = os.path.join(log_dir, "experiment_log.txt")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_logger = logging.getLogger(logger_name)
    file_logger.setLevel(logging.INFO)
    for handler in file_logger.handlers[:]:
        file_logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    file_logger.addHandler(file_handler)
    
    return file_logger, file_handler


def log_experiment_info(file_logger, logger_dir, test_state_dict, evaluation, embedding_dir):
    file_logger.info("="*60)
    file_logger.info("NEW EXPERIMENT STARTED")
    file_logger.info(f"Experiment directory: {logger_dir}")
    file_logger.info(f"Test mode: {test_state_dict is not None}")
    file_logger.info(f"Evaluation mode: {evaluation}")
    
    if test_state_dict is not None:
        file_logger.info("Running in TEST mode - loading from checkpoint")
        file_logger.info(f"Using embeddings from: {embedding_dir}")
    else:
        file_logger.info("Running in TRAINING mode")
    
    file_logger.info("="*60)


def resample_data_generate_durations(config_data, test_only=False, evaluation=False):
    if not test_only:
        dsets = [
            "synth_folder",
            "synth_val_folder",
            "strong_folder",
            "weak_folder",
            "unlabeled_folder",
            "test_folder",
            "strong_extra_folder",
        ]
    elif not evaluation:
        dsets = ["test_folder"]
    else:
        dsets = ["eval_folder"]

    for dset in dsets:
        computed = resample_folder(
            config_data[dset + "_44k"], config_data[dset], target_fs=config_data["fs"]
        )

    if not evaluation:
        for base_set in ["synth_val", "test"]:
            if not os.path.exists(config_data[base_set + "_dur"]) or computed:
                generate_tsv_wav_durations(
                    config_data[base_set + "_folder"], config_data[base_set + "_dur"]
                )


def single_run(
    config,
    log_dir,
    gpus,
    strong_real=False, # False
    checkpoint_resume=None,
    test_state_dict=None,
    fast_dev_run=False,
    evaluation=False,
    callbacks=None,
    need_noise=None
):
    """
    Running sound event detection baselin

    Args:
        config (dict): the dictionary of configuration params
        log_dir (str): path to log directory
        gpus (int): number of gpus to use
        checkpoint_resume (str, optional): path to checkpoint to resume from. Defaults to "".
        test_state_dict (dict, optional): if not None, no training is involved. This dictionary is the state_dict
            to be loaded to test the model.
        fast_dev_run (bool, optional): whether to use a run with only one batch at train and validation, useful
            for development purposes.
    """
    print(strong_real)
    config.update({"log_dir": log_dir})

    # handle seed
    seed = config["training"]["seed"]
    if seed:
        pl.seed_everything(seed, workers=True)

    ##### data prep test ##########
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    if not config["pretrained"]["freezed"]:
        assert config["pretrained"]["e2e"], "If freezed is false, you have to train end2end ! " \
                                            "You cannot use precomputed embeddings if you want to update the pretrained model."
    #FIXME
    if not config["pretrained"]["e2e"]:
        assert config["pretrained"]["extracted_embeddings_dir"] is not None, \
            "If e2e is false, you have to download pretrained embeddings from {}" \
                                                                             "and set in the config yaml file the path to the downloaded directory".format("REPLACE ME")

        pretrained = None
        feature_extraction = None

    public_eval = False

    if not evaluation:
        print(config["data"]["test_folder"])
        devtest_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
        devtest_embeddings = None if config["pretrained"]["e2e"] else config["pretrained"]["extracted_embeddings_dir"]
        # devtest_embeddings = None if config["pretrained"]["e2e"] else os.path.join(config["pretrained"]["extracted_embeddings_dir"],
        #                                                                         #    "DESED", "devtest.hdf5")
        #                                                                            "wildDESED", "SOD_SED", 
        #                                                                            "10db", 
        #                                                                            "devtest.hdf5")
        # devtest_embeddings = None if config["pretrained"]["e2e"] else os.path.join(config["pretrained"]["extracted_embeddings_dir"],
        #                                                                            "wildDESED", "Lct_strong", 
        #                                                                            "-5db", 
        #                                                                            "cletest.hdf5")
        devtest_dataset = StronglyAnnotatedSet(
            config["data"]["test_folder"],
            devtest_df,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"], feats_pipeline=feature_extraction,
            embeddings_hdf5_file=devtest_embeddings,
            embedding_type=config["net"]["embedding_type"]
        )
        if public_eval:
            print("Using public eval data")
            devtest_df = pd.read_csv(config["data"]["public_eval_tsv"], sep="\t")
            devtest_embeddings = None if config["pretrained"]["e2e"] else os.path.join(config["pretrained"]["extracted_embeddings_dir"],
                                                                                   "DESED", "eval.hdf5")
            devtest_dataset = StronglyAnnotatedSet(
                config["data"]["public_eval_folder"],
                devtest_df,
                encoder,
                return_filename=True,
                pad_to=config["data"]["audio_max_len"], feats_pipeline=feature_extraction,
                embeddings_hdf5_file=devtest_embeddings,
                embedding_type=config["net"]["embedding_type"]
            )
    else:
        devtest_dataset = UnlabeledSet(
            config["data"]["eval_folder"],
            encoder,
            pad_to=None,
            return_filename=True, feats_pipeline=feature_extraction
        )

    test_dataset = devtest_dataset

    ##### model definition  ############
    sed_student = CRNN(**config["net"])
    sod_student = CRNN_SOD(**config["net_sod"])
    logger = TensorBoardLogger(
        os.path.dirname(config["log_dir"]), config["log_dir"].split("/")[-1],
    )
    logger.log_hyperparams(config)
    print(f"experiment dir: {logger.log_dir}")
    
    file_logger, file_handler = create_experiment_logger(logger.log_dir)
    log_experiment_info(file_logger, logger.log_dir, test_state_dict, evaluation, devtest_embeddings)

    for step in range(1,6): # （1，6) or (5, 6)
        print(f"curriculum learning added for step {step}")
        file_logger.info(f"Starting curriculum learning step {step}")
        
        if test_state_dict is None:
            file_logger.info(f"Step {step}: Setting up training data")
            # Step-wise noise levels
            noise_levels = ['clean']
            if step >= 2: noise_levels.append('10db')
            if step >= 3: noise_levels.append('5db')
            if step >= 4: noise_levels.append('0db')
            if step >= 5: noise_levels.append('-5db')
            file_logger.info(f"Step {step}: Noise levels = {noise_levels}")
            ##### data prep train valid ##########
            synth_df = pd.read_csv(config["data"]["synth_tsv"], sep="\t")
            synth_set_embeddings = None if config["pretrained"]["e2e"] else os.path.join(config["pretrained"]["extracted_embeddings_dir"],
                                                                                    "DESED", "synth_train.hdf5")
                                                                                    # "wildDESED", "SOD_SED", 
                                                                                    # "10db", 
                                                                                    # "synth_train.hdf5")
            synth_set = StronglyAnnotatedSet(
                config["data"]["synth_folder"],
                synth_df,
                encoder,
                pad_to=config["data"]["audio_max_len"],
                feats_pipeline=feature_extraction,
                embeddings_hdf5_file=synth_set_embeddings,
                embedding_type=config["net"]["embedding_type"]
            )

            if strong_real:
                strong_df = pd.read_csv(config["data"]["strong_tsv"], sep="\t")
                strong_set_embeddings = None if config["pretrained"]["e2e"] else os.path.join(config["pretrained"]["extracted_embeddings_dir"],
                                                                                    "DESED", "strong_train.hdf5")
                strong_set = StronglyAnnotatedSet(
                    config["data"]["strong_folder"],
                    strong_df,
                    encoder,
                    pad_to=config["data"]["audio_max_len"],
                    feats_pipeline=feature_extraction,
                    embeddings_hdf5_file=strong_set_embeddings,
                    embedding_type=config["net"]["embedding_type"]
                )


            weak_df = pd.read_csv(config["data"]["weak_tsv"], sep="\t")
            train_weak_df = weak_df.sample(
                frac=config["training"]["weak_split"],
                random_state=config["training"]["seed"],
            )
            valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
            train_weak_df = train_weak_df.reset_index(drop=True)
            weak_set_embeddings = None if config["pretrained"]["e2e"] else os.path.join(config["pretrained"]["extracted_embeddings_dir"],
                                                                                    "DESED", "weak_train.hdf5")
            weak_set = WeakSet(
                config["data"]["weak_folder"],
                train_weak_df,
                encoder,
                pad_to=config["data"]["audio_max_len"], feats_pipeline=feature_extraction,
                embeddings_hdf5_file=weak_set_embeddings,
                embedding_type=config["net"]["embedding_type"]

            )
            unlabeled_set_embeddings = None if config["pretrained"]["e2e"] else os.path.join(config["pretrained"]["extracted_embeddings_dir"],
                                                                                        "DESED", "unlabeled_train.hdf5")
            unlabeled_set = UnlabeledSet(
                config["data"]["unlabeled_folder"],
                encoder,
                pad_to=config["data"]["audio_max_len"],  feats_pipeline=feature_extraction,
                embeddings_hdf5_file=unlabeled_set_embeddings,
                embedding_type=config["net"]["embedding_type"]
            )

            synth_df_val = pd.read_csv(config["data"]["synth_val_tsv"], sep="\t")
            synth_val_embeddings = None if config["pretrained"]["e2e"] else os.path.join(config["pretrained"]["extracted_embeddings_dir"],
                                                                                            "DESED", "synth_val.hdf5")
            synth_val = StronglyAnnotatedSet(
                config["data"]["synth_val_folder"],
                synth_df_val,
                encoder,
                return_filename=True,
                pad_to=config["data"]["audio_max_len"],  feats_pipeline=feature_extraction,
                embeddings_hdf5_file=synth_val_embeddings,
                embedding_type=config["net"]["embedding_type"]
            )

            weak_val_embeddings = None if config["pretrained"]["e2e"] else os.path.join(config["pretrained"]["extracted_embeddings_dir"],
                                                                                        "DESED", "weak_val.hdf5")
            weak_val = WeakSet(
                config["data"]["weak_folder"],
                valid_weak_df,
                encoder,
                pad_to=config["data"]["audio_max_len"],
                return_filename=True,  feats_pipeline=feature_extraction,
                embeddings_hdf5_file=weak_val_embeddings,
                embedding_type=config["net"]["embedding_type"]
            )
            # Splitting the dataset based on unique filenames
            unique_filenames = synth_df['filename'].unique()
            subset_size = len(unique_filenames) // step
            subsets = [pd.DataFrame() for i in range(step)]
            # print(f"subset size: {len(subsets)}")
            unique_filenames = np.random.permutation(unique_filenames)
            if step != 1:
                
                for i, clip_name in enumerate(tqdm(unique_filenames, total=len(unique_filenames))):
                    subset_index = i // subset_size 
                    # print(f"subset index: {subset_index}")
                    if subset_index >= len(subsets):
                        subset_index = len(subsets) - 1 # Ensure the last subset captures all remaining data
                    subsets[subset_index] = pd.concat([subsets[subset_index], synth_df[synth_df['filename'] == clip_name]])
                synth_set = StronglyAnnotatedSet(
                        config["data"]["synth_folder"],
                        subsets[0],
                        encoder,
                        pad_to=config["data"]["audio_max_len"],
                        feats_pipeline=feature_extraction,
                        embeddings_hdf5_file=synth_set_embeddings,
                        embedding_type=config["net"]["embedding_type"]
                    )
            synth_set_list = [synth_set]
            for i, noise_level in enumerate(noise_levels):
                if i>0:
                    noise_train_embeddings = config["pretrained"]["extracted_embeddings_dir"] + "/wildDESED/SOD_SED" + f"/{noise_level}" + "/devtrain.hdf5"
                    print(noise_train_embeddings)
                    synth_set_list.append(StronglyAnnotatedSet(os.path.join(config["data"]["noise_train_folder"], f"train{noise_level}"),
                                                                        subsets[i],
                                                                        encoder,
                                                                        pad_to=config["data"]["audio_max_len"],
                                                                        feats_pipeline=feature_extraction,
                                                                        embeddings_hdf5_file=noise_train_embeddings,
                                                                        embedding_type=config["net"]["embedding_type"]
                                                                        ))
            synth_set = torch.utils.data.ConcatDataset(synth_set_list)
            unique_filenames = synth_df_val['filename'].unique()
            subset_size = len(unique_filenames) // step
            subsets = [pd.DataFrame() for i in range(step)]
            unique_filenames = np.random.permutation(unique_filenames)
            if step != 1:
                for i, clip_name in enumerate(tqdm(unique_filenames, total=len(unique_filenames))):
                    subset_index = i // subset_size 
                    if subset_index >= len(subsets):
                        subset_index = len(subsets)-1  # Ensure the last subset captures all remaining data
                    subsets[subset_index] = pd.concat([subsets[subset_index], synth_df_val[synth_df_val['filename'] == clip_name]])
                synth_val = StronglyAnnotatedSet(
                config["data"]["synth_val_folder"],
                subsets[0],
                encoder,
                return_filename=True,
                pad_to=config["data"]["audio_max_len"],  feats_pipeline=feature_extraction,
                embeddings_hdf5_file=synth_val_embeddings,
                embedding_type=config["net"]["embedding_type"]
            )
            synth_val_set_list = [synth_val]
            for i, noise_level in enumerate(noise_levels):
                if i>0:
                    noise_val_embeddings = config["pretrained"]["extracted_embeddings_dir"] + "/wildDESED/SOD_SED" + f"/{noise_level}" + "/devval.hdf5"
                    print(noise_val_embeddings)
                    synth_val_set_list.append(StronglyAnnotatedSet(
                os.path.join(config["data"]["noise_val_folder"], f"val{noise_level}"),
                subsets[i],
                encoder,
                return_filename=True,
                pad_to=config["data"]["audio_max_len"],  feats_pipeline=feature_extraction,
                embeddings_hdf5_file=noise_val_embeddings,
                embedding_type=config["net"]["embedding_type"]
            ))

            synth_val = torch.utils.data.ConcatDataset(synth_val_set_list)
            if strong_real:
                strong_full_set = torch.utils.data.ConcatDataset([strong_set, synth_set])
                tot_train_data = [strong_full_set, weak_set]
            else:
                tot_train_data = [synth_set, weak_set, unlabeled_set]
            train_dataset = torch.utils.data.ConcatDataset(tot_train_data)

            batch_sizes = config["training"]["batch_size"]
            samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
            batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)

            valid_dataset = torch.utils.data.ConcatDataset([synth_val, weak_val])

            ##### training params and optimizers ############
            epoch_len = min(
                [
                    len(tot_train_data[indx])
                    // (
                        config["training"]["batch_size"][indx]
                        * config["training"]["accumulate_batches"]
                    )
                    for indx in range(len(tot_train_data))
                ]
            )

            if config["pretrained"]["freezed"] or not config["pretrained"]["e2e"]:
                parameters = list(sed_student.parameters())
            else:
                parameters = list(sed_student.parameters()) + list(pretrained.parameters())
            opt = torch.optim.Adam(parameters, config["opt"]["lr"], betas=(0.9, 0.999))

            exp_steps = config["training"]["n_epochs_warmup"] * epoch_len
            exp_scheduler = {
                "scheduler": ExponentialWarmup(opt, config["opt"]["lr"], exp_steps), # corresponding to r(k)
                "interval": "step",
            }


            callbacks = [
            EarlyStopping(
                monitor=f"val/task_{step}/obj_metric",
                patience=config["training"]["early_stop_patience"],
                verbose=True,
                mode="max",
            ),
            ModelCheckpoint(
                dirpath=f"{logger.log_dir}/task_{step}",
                monitor=f"val/task_{step}/obj_metric",
                save_top_k=1,
                mode="max",
                save_last=True,
            ),
        ]
        else:
            file_logger.info(f"Step {step}: Skipping training setup (TEST mode)")
            train_dataset = None
            valid_dataset = None
            batch_sampler = None
            opt = None
            exp_scheduler = None
            logger = logger
            callbacks = None

        # calulate multiply–accumulate operation (MACs) 
        macs, _ = calculate_macs(sed_student, config, test_dataset) 
        print(f"---------------------------------------------------------------")
        print(f"Total number of multiply–accumulate operation (MACs): {macs}\n")
        file_logger.info(f"Model MACs: {macs}")

        
        desed_training = SEDTask4(
            config,
            encoder=encoder,
            sed_student=sed_student,
            sod_student=sod_student,
            pretrained_model=pretrained,
            opt=opt,
            train_data=train_dataset,
            valid_data=valid_dataset,
            test_data=test_dataset,
            train_sampler=batch_sampler,
            scheduler=exp_scheduler,
            fast_dev_run=fast_dev_run,
            evaluation=evaluation,
            # public_eval=public_eval
            step=step,
            need_noise=need_noise
        )

        # Not using the fast_dev_run of Trainer because creates a DummyLogger so cannot check problems with the Logger
        if fast_dev_run:
            flush_logs_every_n_steps = 1
            log_every_n_steps = 1
            limit_train_batches = 2
            limit_val_batches = 2
            limit_test_batches = 2
            n_epochs = 3
            log_dir = "./exp/debug"
        else:
            flush_logs_every_n_steps = 100
            log_every_n_steps = 40
            limit_train_batches = 1.0
            limit_val_batches = 1.0
            limit_test_batches = 1.0
            n_epochs = config["training"]["n_epochs"]

        if gpus == "0":
            accelerator = "cpu"
            devices = 1
        elif gpus == "1":
            accelerator = "gpu"
            devices = [1]
        else:
            raise NotImplementedError("Multiple GPUs are currently not supported")

        trainer = pl.Trainer(
            precision=config["training"]["precision"],
            max_epochs=n_epochs,
            callbacks=callbacks,
            accelerator=accelerator,
            devices=devices,
            strategy=config["training"].get("backend"),
            # accumulate_grad_batches=config["training"]["accumulate_batches"],
            logger=logger,
            gradient_clip_val=config["training"]["gradient_clip"],
            check_val_every_n_epoch=config["training"]["validation_interval"],
            num_sanity_val_steps=0,
            log_every_n_steps=log_every_n_steps,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            deterministic=config["training"]["deterministic"],
            reload_dataloaders_every_n_epochs=1,
            enable_progress_bar=config["training"]["enable_progress_bar"],
        )
        if test_state_dict is None:
            file_logger.info(f"Step {step}: Starting training")
            # start tracking energy consumption
            trainer.fit(desed_training, ckpt_path=checkpoint_resume)
            best_path = trainer.checkpoint_callback.best_model_path
            print(f"best model: {best_path}")
            file_logger.info(f"Step {step}: Training completed. Best model: {best_path}")
            valid_state_dict = torch.load(best_path)["state_dict"]
        if test_state_dict is not None:
            file_logger.info(f"Step {step}: Loading test state dict")
            valid_state_dict = test_state_dict
        desed_training.load_state_dict(valid_state_dict)

    file_logger.info("Starting final testing phase")
    if test_state_dict is not None:
        trainer.test(desed_training)
        file_logger.info("Testing completed")
        file_logger.info("="*60)
    
    file_handler.close()
    file_logger.removeHandler(file_handler)

def prepare_run(argv=None):
    parser = argparse.ArgumentParser("Training a SED system for DESED Task")
    parser.add_argument(
        "--conf_file",
        default="./confs/pretrained_cl.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    parser.add_argument(
        "--log_dir",
        default="./exp/cyj_cl",
        help="Directory where to save tensorboard logs, saved models, etc.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        # default=''
        help="Allow the training to be resumed, take as input a previously saved model (.ckpt).",
    )
    parser.add_argument(
        "--test_from_checkpoint",
        # default=None, 
        default='need', 
        help="Test the model specified"
    )
    parser.add_argument(
        "--gpus",
        default="1",
        help="The number of GPUs to train on, or the gpu to use, default='0', "
             "so uses one GPU",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False, # False
        help="Use this option to make a 'fake' run which is useful for development and debugging. "
             "It uses very few batches and epochs so it won't give any meaningful result.",
    )
    parser.add_argument(
        "--eval_from_checkpoint",
        default=None,
        help="Evaluate the model specified"
    )
    parser.add_argument(
        "--strong_real",
        action="store_true",
        default=False, # False
        help="The strong annotations coming from Audioset will be included in the training phase.",
    )
    parser.add_argument(
        "--aggregation_type",
        nargs='+',
        type=str,
        default=None,
        help="The aggregation types to use for the model. If None, the aggregation types of the config will be used.",
    )

    args = parser.parse_args(argv)

    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)
    if args.aggregation_type is not None:
        configs["net"]["aggregation_type"] = args.aggregation_type
    evaluation = False 
    test_from_checkpoint = args.test_from_checkpoint

    if args.eval_from_checkpoint is not None:
        test_from_checkpoint = args.eval_from_checkpoint
        evaluation = True

    test_model_state_dict = None
    if test_from_checkpoint is not None:
        checkpoint = torch.load(test_from_checkpoint)
        configs_ckpt = checkpoint["hyper_parameters"]
        configs_ckpt["data"] = configs["data"]
        print(
            f"loaded model: {test_from_checkpoint} \n"
            f"at epoch: {checkpoint['epoch']}"
        )
        test_model_state_dict = checkpoint["state_dict"]

    if evaluation:
        configs["training"]["batch_size_val"] = 1

    test_only = test_from_checkpoint is not None
    # resample_data_generate_durations(configs["data"], test_only, evaluation)
    return configs, args, test_model_state_dict, evaluation

if __name__ == "__main__":
    print(' init down')
    noise_levels = ['10db', '5db', '0db', '-5db']
    for need_noise in noise_levels:
        print(f"\n==== Running for noise level: {need_noise} ====\n")
        configs, args, test_model_state_dict, evaluation = prepare_run()
        configs["data"]["test_folder"] = os.path.join(
            configs["data"]["test_folder"].replace(f"test{noise_levels[0]}/", ""), f"test{need_noise}/"
        )
        configs["data"]["test_folder_44k"] = os.path.join(
            configs["data"]["test_folder_44k"].replace(f"test{noise_levels[0]}_44100/", ""), f"test{need_noise}_44100/"
        )
        configs["pretrained"]["extracted_embeddings_dir"] = os.path.join(configs["pretrained"]["extracted_embeddings_dir"], 
                                                                         "wildDESED", "SOD_SED", 
                                                                         f"{need_noise}", "devtest.hdf5")
        # run!
        single_run(
            configs,
            args.log_dir,
            args.gpus,
            args.strong_real,
            args.resume_from_checkpoint,
            test_model_state_dict,
            args.fast_dev_run,
            evaluation,
            need_noise=need_noise
        )
        torch.cuda.empty_cache()