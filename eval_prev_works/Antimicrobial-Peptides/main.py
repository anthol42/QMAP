from src.train_model import train_model, train_model_qmap

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bacterium', type=str, default='E. coli', help='Name of bacterium, in single quotes')
    parser.add_argument('--negatives', type=float, default=0, help='Ratio of negatives to positives')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--option', type=str, default='qmap')
    parser.add_argument('--train_he', action='store_true', help='Train on high efficiency only (MIC < 10 uM)')
    parser.add_argument('--train_me', action='store_true', help='Train on middle efficiency only (10 <= MIC <= 100 uM)')
    parser.add_argument('--train_le', action='store_true', help='Train on low efficiency only (MIC > 100 uM)')
    parser.add_argument('--rnd', action='store_true', help='Random train/test split instead of QMAP benchmark')
    args = parser.parse_args()
    train_mode = 'rnd' if args.rnd else ('he' if args.train_he else ('me' if args.train_me else ('le' if args.train_le else None)))
    epochs = 20 if args.rnd else args.epochs
    if args.option == 'qmap':
        print("Running with QMAP testing")
        train_model_qmap(bacterium=args.bacterium, negatives_ratio=args.negatives, epochs=epochs, train_mode=train_mode)
    else:
        print("Running original code")
        train_model(bacterium=args.bacterium, negatives_ratio=args.negatives, epochs=args.epochs)
