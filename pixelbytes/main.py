#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""
import argparse, sys, os
from . import *
from datasets import load_dataset, load_from_disk

def build_dataset_arg(path, palette=None):
    print(f"Building dataset from {path}")
    if palette:
        print(f"Using custom palette from {palette}")
    # Dataset import
    if os.path.isdir(path): dataset = load_from_disk(path)
    else : dataset = load_dataset(path)["train"]
    # dataset construct
    dataset = add_pixelbyte_columns(dataset)
    dataset.save_to_disk('PixelBytes-Dataset')

def train_arg(data, seq_length, strides,  
              model, dim, d_state, depth, grid, pxby, bidirect,
              learning_rate, batch_size, epochs):
    print(f"Training {model} model from {data}")
    print(f"Sequence length and dataset stride : ({seq_length}, {strides})")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    # Import dataset
    if os.path.isdir(data): dataset = load_from_disk(data)
    else : dataset = load_dataset(data)["train"]
    ## Initalization
    # data part
    print(f"Construct data sequence for dataloader..")
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = PxByDataset(dataset["train"]["pixelbyte"], seq_length=seq_length, stride=strides)
    test_dataset = PxByDataset(dataset["test"]["pixelbyte"], seq_length=seq_length, stride=strides)
    print(f"Data sequence OK!")
    # model part
    tokenizer = PixelBytesTokenizer()
    vocab_size = tokenizer.__len__()
    model_config = ModelConfig(dim=dim, d_state=d_state, depth=depth, vocab_size=vocab_size, 
                               pxby_embed=grid, pembed=pxby, bidirectional=bidirect)
    if model == "rnn" : model = SimpleRNNModel(model_config)
    elif model == "ssm" : model = bMamba(model_config)
    elif model == "att" : model = SimpleTransformerModel(model_config)
    # train part
    train_config = TrainConfig(model=model, model_config=model_config, dataset_name=data, batch_size=batch_size,
                           train_dataset=train_dataset,test_dataset=test_dataset, num_epochs=epochs, learning_rate=learning_rate)
    trainer = Trainer(train_config)
    # launch train
    trainer.train_and_evaluate()

def evaluate_arg(metrics):
    print(f"Evaluating model using metrics: {', '.join(metrics)}")
    # Add your evaluation logic here
    print(f"Sorry is not implemented for now.. see docs .ipynb or source code if you want to continue..")

def generate_arg(path, model, formats):
    # Generate and display results
    print(f"Generating results in {formats} format (IN DEV)")
    # Import dataset
    if os.path.isdir(path): dataset = load_from_disk(path)
    else : dataset = load_dataset(path)["train"]
    # Import model (DEV!)
    print(f"You import really dummy model !")
    model = SimpleRNNModel.from_pretrained(path, subfolder=model)
    ## Initalization
    tokenizer = PixelBytesTokenizer()
    displayer = Displays(tokenizer)
    sample = dataset.shuffle(seed=42).select(range(1))[0]
    displayer.reset(model)
    complete_seq = np.array(sample['pixelbyte'])
    ## Process and display
    images, text = displayer.process_sequence(complete_seq)
    displayer.show(images, text)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Project task manager")
    subparsers = parser.add_subparsers(dest='task', help='Task to execute')
    # Build dataset parser
    build_parser = subparsers.add_parser('build', help='Build the PixelBytes dataset')
    build_parser.add_argument('--path', type=str, default='ffurfaro/PixelBytes-Pokemon', help='Path to the dataset')
    build_parser.add_argument('--palette', type=str, default=None, help='Path to custom palette file')
    # Train model parser
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data', type=str, default='PixelBytes-Dataset', help='PixelBytes dataset path')
    train_parser.add_argument('--seq_length', type=int, default=128, help='Sequence length')
    train_parser.add_argument('--strides', type=int, default=8, help='Stride : 1 is full dataset point')
    
    train_parser.add_argument('--model', type=str, default='rnn', help='Model to train')
    train_parser.add_argument('--dim', type=int, default=81, help='Model embedding dimension')
    train_parser.add_argument('--d_state', type=int, default=64, help='Hidden state')
    train_parser.add_argument('--depth', type=int, default=2, help='Depth of the model')
    train_parser.add_argument('--grid', type=bool, default=True, help='if grid False, only center of 2D batch')
    train_parser.add_argument('--pxby', type=bool, default=True, help='if pxby False, Conv2D not included')
    train_parser.add_argument('--bidirect', type=bool, default=True, help='Model directionnality')

    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    # Evaluate model parser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--metrics', nargs='+', default=['accuracy'], help='Metrics to evaluate')
    # Generate output parser
    gen_parser = subparsers.add_parser('generate', help='Generate and display results')
    gen_parser.add_argument('--path', type=str, default='ffurfaro/PixelBytes-Pokemon', help='Path to the dataset')
    gen_parser.add_argument('--model', type=str, default='rnn_bi_pxby_81_dim_64_state_2_layer_last', help='Path to the dataset')
    gen_parser.add_argument('--format', choices=['svg', 'png', 'jpg'], default='svg', help='Output format')
    return parser.parse_args()

def print_help():
    print("Usage: python -m pixelbytes.main [task] [task-specific-arguments]")
    print("\nAvailable tasks:")
    print("  build     - Build the pixelbytes dataset columns")
    print("  train     - Train the sequence model")
    print("  evaluate  - Evaluate the model and save statistic")
    print("  generate  - Generate and display results")
    print("\nFor task-specific help, use: python -m your_package.main [task] --help")
    print("For general help, use: python -m your_package.main --help")

def main():
    if len(sys.argv) == 1:
        print("No task specified.")
        print_help()
        return

    args = parse_arguments()

    if args.task == 'build':
        build_dataset_arg(args.path, args.palette)
    elif args.task == 'train':
        train_arg(args.data, args.seq_length, args.strides,  
                  args.model, args.dim, args.d_state, args.depth, args.grid, args.pxby, args.bidirect,
                  args.learning_rate, args.batch_size, args.epochs)
    elif args.task == 'evaluate':
        evaluate_arg(args.metrics)
    elif args.task == 'generate':
        generate_arg(args.path, args.model, args.format)
    else:
        print("Invalid task specified.")
        print_help()

if __name__ == "__main__":
    main()