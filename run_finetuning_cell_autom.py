import json
import logging
import os
import math
from pathlib import Path
from itertools import chain
import wandb
# from dotenv import load_dotenv
import random
import torch
import numpy as np
import datasets
from datasets import Dataset
from torch.utils.data import DataLoader
from datetime import timedelta

from lm_experiments_tools import Trainer, TrainerArgs

from torch.nn.utils.rnn import pad_sequence

import accelerate
from accelerate import InitProcessGroupKwargs
from baselines.rwkv.RWKV_v5.src.dataflow.trie_tokenizer import MT_TRIE_TOKENIZER
# load_dotenv()

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')


# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

# import transformers  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model

from lm_experiments_tools.utils import get_cls_by_name, get_optimizer, prepare_run  # noqa: E402

# limit # of CPU threads to be used per pytorch worker, otherwise it might use all cpus and throttle gpus
# > 2 fails cause of https://github.com/pytorch/pytorch/issues/56615
# need to upgrade to torch>1.8.1
# torch.set_num_threads(4)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...
# torch.cuda.set_device(hvd.local_rank())

parser = HfArgumentParser(TrainerArgs)

parser.add_argument('--rwkv_tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')
parser.add_argument('--task_name', default=None, type=str, help="Task name, wikitext, ...")
parser.add_argument('--tokenized_dataset', type=str, help="Tokenized dataset, ...", default=None)
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--show_valid_examples', type=int, default=0,
                    help='how many valid examples to show during training (default: 0)')
parser.add_argument('--input_seq_len', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--val_seq_len', type=int, default=128, help='input sequnce length for validation (default: 128).')
parser.add_argument('--target_seq_len', type=int, default=16, help='target sequnce length, should be set to '
                                                                   'max(len(target))+1 for EOS (default: 16).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

parser.add_argument('--input_prefix', type=str, default='', help='add task prefix to an input string (default: "")')
parser.add_argument('--sliding_window', action='store_true', help='use slinding window attentinon mask, '
                    'eval on last segment only', default=False)

# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--pretrained_teacher', type=str, help='teacher model name in HF Model Hub (default: "")')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--memory_cell_cls', type=str, default=None, help='cell class for RMT')
parser.add_argument('--recurrent_wrapper_cls', type=str, default=None, help='recurrent wrapper class for RMT')
parser.add_argument('--distillator_cls', type=str, default=None, help='distillator class for RMT')
parser.add_argument('--teacher_cls', type=str, default=None, help='teacher class')
parser.add_argument('--model_cpt', type=str, default=None, help='pretrained model checkpoint path')
parser.add_argument('--model_type', type=str, default='encoder-decoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing '
                         '(default: encoder-decoder)')

parser.add_argument('--alpha_distil', type=float, default=None, help='')

# Aydar # RMT args
parser.add_argument('--input_size', type=int, default=None, help='maximal input size of the backbone model')
parser.add_argument('--block_size', type=int, default=None, help='number of real tokens in block')
parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--d_mem', type=int, default=None, help='number of rows in associative matrix')
parser.add_argument('--layers_attr', type=str, default=None, help='attribute of model, which contains layers')
parser.add_argument('--n_heads', type=int, default=None, help='number of heads in associative matrix')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--max_val_segments', type=int, default=1, help='maximal segment number on validation')
parser.add_argument('--vary_n_segments', action='store_true', default=False, help='Randomly choose segment number from 1 to max_n_segments')
parser.add_argument('--random_segment_size', action='store_true', default=False, help='Randomly choose segment size from input_size to max_n_segments * input_size with powers of 2')
parser.add_argument('--prev_seg_kv', action='store_true', default=False, help='propagate kv from previous segment')
parser.add_argument('--sum_loss', action='store_true', default=False,
                    help='with this flag task loss from all segments is summed')
parser.add_argument('--bptt_depth', type=int, default=-1, help='max number of previous segments in gradient computation.')
parser.add_argument('--segment_ordering', type=str, help='segment order', default='regular',
                    choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])
parser.add_argument('--memory_forward_func', type=str, help='path to memory forward funсtion script', default=None)
parser.add_argument('--memory_layers', type=str, help='memory-augmented layer inds or "all" for all layers', default=None)
parser.add_argument('--share_memory_layers', action='store_true', help='share weights of memory layers', default=False)
parser.add_argument('--reconstruction_loss_coef', type=float, default=None,
                    help='reconstuction loss ratio in total loss')
# parser.add_argument('--segment_ordering', type=str,help='????', default='regular',
#                     choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])
parser.add_argument('--retain_graph', action='store_true', help='Retain computation graph during backward pass', default=False)
parser.add_argument('--use_truncated_backward', action='store_true', default=False,
                    help='whether to use RMT truncated bptt method in backward')
parser.add_argument('--k1', type=int, default=-1, help='(not implemented) If not -1, gradient update is done each k1 segments')
parser.add_argument('--k2', type=int, default=-1, help='number of last segments used by backward')
parser.add_argument('--freeze_model_weights', action='store_true', default=False,
                    help='Stop training all model weights except memory layers')
parser.add_argument('--backbone_cpt', type=str, default=None, help='backbone model checkpoint path')
parser.add_argument('--no_denom', action='store_true', default=None,
                    help='use no denominator in ARMT')
parser.add_argument('--train_timesteps', type=int, default=10, help='timesteps of CA to process during training for the prediction of the next steps')
parser.add_argument('--valid_timesteps', type=int, default=10, help='timesteps of CA to process during validation for the prediction of the next steps')
parser.add_argument('--prediction_shift', type=int, default=1, help='num_timesteps between the last training steps and the predicted timestep')

# tokenizer
# todo: add wordpiece tokenizers support?
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')

# LoRA args
parser.add_argument('--use_lora', action='store_true', default=False, help='')
parser.add_argument('--lora_attn_dim', type=int, default=8, help='')
parser.add_argument('--lora_attn_alpha', type=int, default=32, help='')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='')

# Parallel Adapter args
parser.add_argument('--use_adapter', action='store_true', default=False, help='')
parser.add_argument('--adapter_bottleneck_dim', type=int, default=512, help='')
parser.add_argument('--adapter_dropout', type=float, default=0.1, help='')
parser.add_argument('--adapter_scale', type=float, default=4.0, help='')

parser.add_argument('--report_to', type=str, default='wandb', help='')

if __name__ == '__main__':
    args = parser.parse_args()
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)
    kwgs = InitProcessGroupKwargs()
    kwgs.timeout = timedelta(seconds=1800)
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[kwgs])
    from accelerate.logging import get_logger
    logger = get_logger('')

    logger.info(f'num processes: {accelerator.num_processes}')
    logger.info(f'mixed precision: {accelerator.mixed_precision}')

    if args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    # # create model path and save configuration
    # # todo: use prepare run
    # if accelerator.is_main_process and args.model_path is not None:
    #     model_path = Path(args.model_path)
    #     if not model_path.exists():
    #         Path(model_path).mkdir(parents=True)
    #     args_dict = collect_run_configuration(args)
    #     # todo: if model path exists and there is config file, write new config file aside
    #     json.dump(args_dict, open(model_path/'config.json', 'w'), indent=4)
    #     open(model_path / 'git.diff', 'w').write(get_git_diff())

    prepare_run(args, logger, logger_fmt)

    # Prepare datasets
    logger.info(f'preparing dataset for {args.task_name}')

    with accelerator.main_process_first():
        if args.tokenized_dataset is not None:
            tokenized_datasets = datasets.load_dataset(args.tokenized_dataset)
            tokenized_datasets.remove_columns('text')
    # ========
        debug = True
        lim = 1000 if debug else ''
        raw_datasets = dict(
            train=datasets.load_dataset('irodkin/1dCA_r2s20T20', split=f'train[:{lim}]'),
            validation=datasets.load_dataset('irodkin/1dCA_r2s20T20', split=f'validation[:{lim}]'),
            test=datasets.load_dataset('irodkin/1dCA_r2s20T20', split=f'test[:{lim}]')
        )
        tokenized_datasets = raw_datasets
    # =======

    block_size = args.block_size
    history_size = args.input_seq_len - block_size

    if args.val_seq_len is not None:
        val_history_size = args.val_seq_len - block_size
    else:
        val_history_size = history_size


    def group_texts(examples, valid=False):
        steps = args.valid_timesteps if valid else args.train_timesteps
        shift = args.prediction_shift
        result = {
            # concatenate input_ids_t for the corresponding steps
            'input_ids': [i for t in range(steps) if f'input_ids_{t}' in examples for i in examples[f'input_ids_{t}']]
        }
        result['input_ids'] = result['input_ids'] + examples[f'input_ids_{steps+shift-1}']
        result['labels'] = result['input_ids'].copy()
        result['attention_mask'] = [1 for _ in result['input_ids']] 
        return result

    def collate_fn(batch, valid=False):
            input_ids = [torch.tensor(b['input_ids'][::-1]).long() for b in batch]
            labels = [torch.tensor(b['labels'][::-1]).long() for b in batch]
            attention_mask = [torch.tensor(b['attention_mask'][::-1]).long() for b in batch]
            input_ids = pad_sequence(input_ids, padding_value=-100).T.flip(1)
            labels = pad_sequence(labels, padding_value=-100).T.flip(1)
            attention_mask = pad_sequence(attention_mask, padding_value=0).T.flip(1)

            collated = {'input_ids': input_ids,
                        'labels': labels, 
                        'attention_mask': attention_mask}

            if input_ids.shape[1] != block_size:
                labels_mask = torch.ones_like(input_ids, dtype=bool)
                labels_mask[:, :-block_size] = False
                collated['labels_mask'] = labels_mask
            
            return collated

    with accelerator.main_process_first():
        logger.info('starting grouping texts')
        # train_dataset = Dataset.from_dict(group_texts(tokenized_datasets['train'].to_dict(), block_size, history_size))
        train_dataset = tokenized_datasets["train"].map(lambda x: group_texts(x, valid=False), desc=f"Grouping train in chunks")
        valid_dataset = tokenized_datasets["validation"].map(lambda x: group_texts(x, valid=True), desc=f"Grouping valid in chunks")
        logger.info('ended grouping texts')
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    # shuffle train data each epoch (one loop over train_dataset)
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    train_rnd_generator = torch.Generator()
    train_rnd_generator.manual_seed(args.seed)
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, collate_fn=collate_fn,
                                  shuffle=True, drop_last=False, generator=train_rnd_generator, **kwargs)

    # dataloader for validation
    # batch sample i is a continuation of sample i of the previous batch
    class alignedDataLoader(DataLoader):
        def __iter__(self):
            all_inds = np.arange((len(self.dataset) // self.batch_size) * self.batch_size)
            all_inds = all_inds.reshape(self.batch_size, -1)
            for batch_ind in range(all_inds.shape[1]):
                batch = [self.dataset[int(ind)] for ind in all_inds[:, batch_ind]]
                yield self.collate_fn(batch)

    # get validation dataset
    valid_dataloader = None
    logger.info(f'preparing validation data from {args.task_name}')
    valid_dataloader = alignedDataLoader(valid_dataset, batch_size=per_worker_batch_size,
                                         collate_fn=lambda x: collate_fn(x, valid=True), shuffle=False, drop_last=False, **kwargs)

    # get test dataset
    
    if 'test' in tokenized_datasets.keys():
        with accelerator.main_process_first():
            test_dataset = tokenized_datasets["test"].map(lambda x: group_texts(x, valid=True), desc=f"Grouping test in chunks")

        test_dataloader = alignedDataLoader(test_dataset, batch_size=per_worker_batch_size,
                                            collate_fn=lambda x: collate_fn(x, valid=True), shuffle=False, drop_last=True, **kwargs)
        

    if args.valid_interval is None:
        args.valid_interval = args.log_interval

    # define model

    logger.info('starting with test')
    model_cls = get_cls_by_name(args.model_cls)
    logger.info(f'Using model class: {model_cls}')

    if args.use_adapter:
        model_cfg = AutoConfig.from_pretrained(args.from_pretrained)

        model_cfg.use_parallel_adapter = args.use_adapter
        model_cfg.parallel_adapter_mode = 'ffn'
        model_cfg.adapter_bottleneck_dim = args.adapter_bottleneck_dim
        model_cfg.adapter_dropout = args.adapter_dropout
        model_cfg.adapter_scale = args.adapter_scale

        model = model_cls(config=model_cfg)

        logger.info(f'Loading pretrained model: {args.from_pretrained}')
        base_model = model_cls.from_pretrained(args.from_pretrained, use_safetensors=False, attn_implementation="flash_attention_2")

        model.load_state_dict(base_model.state_dict(), strict=False)
        del base_model
        logger.info(f'Added adapters')

    else:
        if not args.from_pretrained:
            model_cfg = AutoConfig.from_pretrained(args.model_cfg)
            model = model_cls(config=model_cfg)
        else:
            logger.info(f'Loading pretrained model: {args.from_pretrained}')
            model = model_cls.from_pretrained(args.from_pretrained)

    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_attn_dim, 
            lora_alpha=args.lora_attn_alpha, 
            lora_dropout=args.lora_dropout
            )
        model = get_peft_model(model, peft_config)
        logger.info(f'Added LoRA, trainable parameters with LoRA only:')
        model.print_trainable_parameters()
    

    ## load cpt of backbone model
    if args.backbone_cpt:
        backbone_cpt = os.path.join(args.backbone_cpt, "model_best")
        cpt = torch.load(backbone_cpt, map_location='cpu')
        model.load_state_dict(cpt['model_state_dict'], strict=False)
        logger.info(f'Loaded baseline state dict from: {args.backbone_cpt}')

    # Pass memory settings to pretrained model
    if True:
        memory_cell_cls = get_cls_by_name(args.memory_cell_cls)
        recurrent_wrapper_cls = get_cls_by_name(args.recurrent_wrapper_cls)
        if args.distillator_cls is not None:
            distillator = get_cls_by_name(args.distillator_cls)

        logger.info(f'Wrapping in: {memory_cell_cls} and {recurrent_wrapper_cls}')
        
        
        mem_cell_args = dict(
            base_model=model,
        )
        if args.d_mem is not None:
            mem_cell_args['d_mem'] = args.d_mem

        if args.layers_attr is not None:
            mem_cell_args['layers_attr'] = args.layers_attr
        
        if args.n_heads is not None:
            mem_cell_args['n_heads'] = args.n_heads
        
        if args.num_mem_tokens is not None:
            mem_cell_args['num_mem_tokens'] = args.num_mem_tokens
        if args.no_denom is not None:
            mem_cell_args['use_denom'] = not args.no_denom

        cell = memory_cell_cls(**mem_cell_args)
        model = recurrent_wrapper_cls(cell, 
                                      segment_size=block_size,
                                      max_n_segments=args.max_n_segments, 
                                      vary_n_segments=args.vary_n_segments,
                                      k2=args.k2,
                                      sliding_window=args.prev_seg_kv
        )

        if args.distillator_cls is not None:
            teacher_cls = get_cls_by_name(args.teacher_cls)
            teacher = teacher_cls.from_pretrained(args.pretrained_teacher)
            model = distillator(teacher, model, alpha_distil=args.alpha_distil)             

        ## load cpt of rmt
        if args.model_cpt and args.model_cpt != 'None':
            if 'rwkv' not in args.model_cpt:
                model_cpt = os.path.join(args.model_cpt, "model_best/pytorch_model.bin")
                cpt = torch.load(model_cpt, map_location='cpu')
                model.load_state_dict(cpt)
                logger.info(f'Loaded RMT state dict from: {args.model_cpt}')
            else:
                import safetensors
                model_cpt = os.path.join(args.model_cpt, "model_best/model.safetensors")
                cpt = safetensors.torch.load_file(model_cpt)
                w = model.load_state_dict(cpt, strict=False)
                model.memory_cell.model.tie_weights()
                logger.info(f'loaded rwkv with mis w {w}')


    def to_freeze(name):
        if 'memory_cell'in name:
            name = ''.join(name.split('memory_cell'))
        return 'memory' not in name and 'lora' not in name and 'adapter' not in name and 'W_m' not in name
    if args.freeze_model_weights:
        for n, p in model.named_parameters():
            if to_freeze(n):
                p.requires_grad = False
            else:
                p.requires_grad = True
        logger.info(f'Frozen moodel weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')

    # fix the not-contiguous error
    def make_contiguous(module):
        with torch.no_grad():
            for param in module.parameters():
                param.set_(param.contiguous())
    make_contiguous(model)

    # define optimizer
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    logger.info(f'Using optimizer class: {optimizer_cls}')

    # todo: group optimizer params
    optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)    
    # if args.model_cpt or args.backbone_cpt:
        # optimizer.load_state_dict(cpt['optimizer_state_dict'])

    # for encoder only classification
    def keep_for_metrics_fn(batch, output):
        # select data from batch and model output that would be used to compute metrics
        data = {}
        data['labels'] = batch['labels']
        data['loss'] = output['loss']
        data['predictions'] = torch.argmax(output['logits'].detach(), dim=-1)
        if 'ce_loss' in output:
            data['ce_loss'] = output['ce_loss']
        if 'dist' in output:
            data['dist'] = output['dist']
        
        for i in range(args.max_n_segments):
            if f'ce_loss_{i}' in output:
                data[f'ce_loss_{i}'] = output[f'ce_loss_{i}']
        return data

    # HF datasets can compute metrics on each gpu process and then aggregate them on process with rank 0
    # synchronization is done by using temporay files on a shared filesystem
    # rank and number of workers is set by num_process and process_id params
    # BUT our Trainer aggregates all prediction from all gpus!
    #   this will lead to computing metrics for predictions repeated xN_GPUS times
    # need to try:
    # - keep_in_memory=True, may lead to OOM for large validation sets, after sync predictions and targets for the full
    #       validation set would be stored on each GPU -> xN_GPUs RAM
    #   - implemented currently
    # - compute metrics on batch lvl
    # - add support of HF metrics and turn off aggregation in case if metric has .add_batch method
    # scrolls_metric = datasets.load_metric(scrolls_metric_path, args.task_name, keep_in_memory=True)


    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        y, p = data['labels'], data['predictions']
        if accelerator.is_main_process == 0 and args.show_valid_examples > 0:
            for i in range(min(args.show_valid_examples, len(y))):
                y_ = np.array(y[i])
                p_ = np.array(p[i])
                logger.info(f'y: {y_}')
                logger.info(f'p: {p_}')
                logger.info(f'y: {y[i]}')
                logger.info(f'p: {p[i]}')
                logger.info('-' * 50)
        if 'ce_loss' in data:
            metrics['ce_loss'] = data['ce_loss'].mean()
            try:
                perplexity = math.exp(metrics['ce_loss'])
            except OverflowError:
                perplexity = float("inf")

            metrics["perplexity"] = perplexity
        
        if 'dist' in data:
            metrics['dist'] = data['dist'].mean()
            
        for i in range(args.max_n_segments):
            if f'ce_loss_{i}' in data:
                metrics[f'ce_loss_{i}'] = data[f'ce_loss_{i}'].mean()
        metrics['bit_accuracy'] = np.mean(np.array(y) == np.array(p))
        metrics['exact_match'] = np.mean([np.array_equal(p_, y_) for p_, y_ in zip(p, y)])

        return metrics

    # accelerate
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, test_dataloader)

    ### booydar
    batch_metrics_fn = lambda _, y: {key: y[key] for key in y.keys() if (('loss' in key) or ('!log' in key))}
    trainer = Trainer(args, accelerator, model, optimizer, train_dataloader, valid_dataloader,  # train_sampler,
                      keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn,
                      batch_metrics_fn=batch_metrics_fn,
                      forward_kwargs={
                          'input_segmented': getattr(args, 'random_segment_size', False),
                      },
                      generate_kwargs={})

    if not args.validate_only:
        # train loop
        trainer.train()
        # make sure all workers are done
        accelerator.wait_for_everyone()
        # run validation after training
        if args.save_best:
            best_model_path = str(Path(args.model_path) / 'model_best')
            logger.info(f'Loading best saved model from {best_model_path}')
            trainer.load(best_model_path)
        if valid_dataloader is not None:
            logger.info('Runnning validation on valid data:')
            metrics = trainer.validate(valid_dataloader, write_tb=False, split='valid')
            evaluated_on = []
            metric_on = []
            for i in range(args.max_val_segments):
                if f'ce_loss_{i}' in metrics:
                    evaluated_on.append(i)
                    metric_on.append(metrics[f'ce_loss_{i}'])
            if args.report_to == 'wandb' and accelerator.is_main_process:
                table = wandb.Table(data=np.vstack([evaluated_on, metric_on]).T, columns=['evaluated_on', 'valid/ce_loss'])
                line = trainer.run.plot_table("wandb/line/v0", table, {"x":'evaluated_on', "y":'valid/ce_loss'})
                trainer.run.log({'per_segment_eval': line})
        if test_dataloader is not None:
            logger.info('Runnning validation on test data:')
            metrics = trainer.validate(test_dataloader, write_tb=False, split='test')
            evaluated_on = []
            metric_on = []
            for i in range(args.max_val_segments):
                if f'ce_loss_{i}' in metrics:
                    evaluated_on.append(i)
                    metric_on.append(metrics[f'ce_loss_{i}'])
            if args.report_to == 'wandb' and accelerator.is_main_process:
                table = wandb.Table(data=np.vstack([evaluated_on, metric_on]).T, columns=['evaluated_on', 'test/ce_loss'])
                line = trainer.run.plot_table("wandb/line/v0", table, {"x":'evaluated_on', "y":'test/ce_loss'})
                trainer.run.log({'per_segment_test': line})
        trainer.save_metrics(save_path=args.model_path)
    else:
        # run validation, do not write to tensorboard
        logger.info('Running validation on train set:')
        trainer.validate(train_dataloader, split='train', write_tb=True)
        if valid_dataloader is not None:
            logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False, split='valid')
        if test_dataloader is not None:
            logger.info('Runnning validation on test data:')
            trainer.validate(test_dataloader, write_tb=False, split='test')
