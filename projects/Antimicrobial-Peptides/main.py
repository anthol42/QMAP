from src.train_model import train_model, train_model_qmap

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bacterium', type=str, default='E. coli', help='Name of bacterium, in single quotes')
    parser.add_argument('--negatives', type=float, default=1, help='Ratio of negatives to positives')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--option', type=str, default='qmap')
    args = parser.parse_args()
    if args.option == 'qmap':
        print("Running with QMAP testing")
        train_model_qmap(bacterium=args.bacterium, negatives_ratio=args.negatives, epochs=args.epochs)
    else:
        print("Running original code")
        train_model(bacterium=args.bacterium, negatives_ratio=args.negatives, epochs=args.epochs)
