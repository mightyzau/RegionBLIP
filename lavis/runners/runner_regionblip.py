import logging 
import time
import datetime
from collections import defaultdict
import torch 
import torch.distributed as dist
import random

from lavis.common.registry import registry
from lavis.runners.runner_base import RunnerBase, Path
from lavis.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process
)
from lavis.datasets.data_utils import concat_datasets, reorg_datasets_by_split


@registry.register_runner("runner_regionblip")
class RunnerRegionBLIP(RunnerBase):
    def __init__(self, cfg, task, model, datasets, job_id):
        super().__init__(cfg, task, model, datasets, job_id)

        self.task2datasetname = defaultdict(list)
        for dataset_name in self.datasets:
            task_name = self.config.datasets_cfg.get(dataset_name).task_name
            self.task2datasetname[task_name].append(dataset_name)

        self.task_names = list(self.task2datasetname.keys())
        self.task_names.sort()
        logging.info("task_names: {}".format(self.task_names))
    
    def setup_output_dir(self):
        output_dir = Path(self.config.run_cfg.output_dir)

        result_dir = output_dir / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir
    

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        self.log_config()

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # training phase
            if not self.evaluate_only:
                for task_name in self.task_names:
                    train_stats = self.train_epoch(cur_epoch, task_name)
                    self.log_stats(split_name="train__{}".format(task_name), stats=train_stats)

            # evaluation phase
            if len(self.valid_splits) > 0:
                for split_name in self.valid_splits:
                    logging.info("Evaluating on {}.".format(split_name))

                    val_log = self.eval_epoch(
                        split_name=split_name, cur_epoch=cur_epoch
                    )
                    if val_log is not None:
                        if is_main_process():
                            assert (
                                "agg_metrics" in val_log
                            ), "No agg_metrics found in validation log."

                            agg_metrics = val_log["agg_metrics"]
                            if agg_metrics > best_agg_metric and split_name == "val":
                                best_epoch, best_agg_metric = cur_epoch, agg_metrics

                                self._save_checkpoint(cur_epoch, is_best=True)

                            val_log.update({"best_epoch": best_epoch})
                            self.log_stats(val_log, split_name)

            else:
                # if no validation split is provided, we just save the checkpoint at the end of each epoch.
                if not self.evaluate_only:
                    self._save_checkpoint(cur_epoch, is_best=False)

            if self.evaluate_only:
                break

            dist.barrier()

        # testing phase
        test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
        self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))
    
    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        if self._dataloaders is None:
            self._dataloaders = dict()

            # reoganize datasets by split and concatenate/chain if necessary
            dataset_ratios = self.config.run_cfg.get("train_dataset_ratios", None)              # None

            # concatenate map-style datasets and chain wds.DataPipe datasets separately
            # training set becomes a tuple (ConcatDataset, ChainDataset), both are
            # optional but at least one of them is required. The resultant ConcatDataset
            # and ChainDataset will be sampled evenly.
            logging.info(
                "dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline)."
            )

            for task_name in self.task_names:
                task_datasets = dict()
                for dataset_name in self.datasets:
                    if dataset_name not in self.task2datasetname[task_name]:
                        continue
                    task_datasets[dataset_name] = self.datasets[dataset_name]
                
                task_datasets = reorg_datasets_by_split(task_datasets)   # Organizes datasets by split.
                task_datasets = concat_datasets(task_datasets)
                

                # create dataloaders
                task_split_names = sorted(task_datasets.keys())

                task_datasets = [task_datasets[split] for split in task_split_names]
                task_is_trains = [split in self.train_splits for split in task_split_names]

                assert task_name in self.config.run_cfg.batch_size_train_per_task, "please set {} in run_cfg.batch_size_train_per_task".format(task_name)
                task_batch_sizes = [
                    self.config.run_cfg.batch_size_train_per_task.get(task_name, 100)
                    if split == "train"
                    else self.config.run_cfg.batch_size_eval
                    for split in task_split_names
                ]
                logging.info('task_name: {}, batch_size_train: {}'.format(task_name, self.config.run_cfg.batch_size_train_per_task.get(task_name, 100)))

                task_collate_fns = []
                for dataset in task_datasets:
                    if isinstance(dataset, tuple) or isinstance(dataset, list):
                        task_collate_fns.append([getattr(d, "collater", None) for d in dataset])
                    else:
                        task_collate_fns.append(getattr(dataset, "collater", None))

                task_dataloaders = self.create_loaders(
                    datasets=task_datasets,
                    num_workers=self.config.run_cfg.num_workers,
                    batch_sizes=task_batch_sizes,
                    is_trains=task_is_trains,
                    collate_fns=task_collate_fns,
                    dataset_ratios=dataset_ratios,
                )

                task_dataloaders = {k: v for k, v in zip(task_split_names, task_dataloaders)}
                self._dataloaders[task_name] = task_dataloaders

        return self._dataloaders


    #def train_loader(self, task_name):
    #    train_dataloader = self.dataloaders[task_name]["train"]
    #    return train_dataloader
    
    def train_epoch(self, epoch, task_name):
        # train
        self.model.train()
        train_dataloader = self.dataloaders[task_name]['train']

        return self.task.train_epoch(
            task_name=task_name,
            epoch=epoch,
            model=self.model,
            data_loader=train_dataloader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )
