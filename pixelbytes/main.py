#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""
import argparse, sys

def build_dataset(path, palette=None):
    print(f"Building dataset from {path}")
    if palette:
        print(f"Using custom palette from {palette}")
    # Add your dataset building logic here

def train(model, learning_rate, batch_size, epochs):
    print(f"Training {model} model")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    # Add your training logic here

def evaluate(metrics):
    print(f"Evaluating model using metrics: {', '.join(metrics)}")
    # Add your evaluation logic here
    
def generate():
    # Generate and display results
    print(f"Generating results in {format} format")
    # Add your generation logic here

def parse_arguments():
    parser = argparse.ArgumentParser(description="Project task manager")
    subparsers = parser.add_subparsers(dest='task', help='Task to execute')

    # Build dataset parser
    build_parser = subparsers.add_parser('build', help='Build the dataset')
    build_parser.add_argument('--path', type=str, required=True, help='Path to the dataset')
    build_parser.add_argument('--palette', type=str, help='Path to custom palette file')

    # Train model parser
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--model', type=str, required=True, help='Model to train')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')

    # Evaluate model parser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--metrics', nargs='+', default=['accuracy'], help='Metrics to evaluate')

    # Generate output parser
    gen_parser = subparsers.add_parser('generate', help='Generate and display results')
    gen_parser.add_argument('--format', choices=['svg', 'png', 'jpg'], default='svg', help='Output format')

    return parser.parse_args()

def print_help():
    print("Usage: python -m your_package.main [task] [task-specific-arguments]")
    print("\nAvailable tasks:")
    print("  build     - Build the dataset")
    print("  train     - Train the model")
    print("  evaluate  - Evaluate the model")
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
        build_dataset(args.path, args.palette)
    elif args.task == 'train':
        train(args.model, args.learning_rate, args.batch_size, args.epochs)
    elif args.task == 'evaluate':
        evaluate(args.metrics)
    elif args.task == 'generate':
        generate(args.format)
    else:
        print("Invalid task specified.")
        print_help()

if __name__ == "__main__":
    main()