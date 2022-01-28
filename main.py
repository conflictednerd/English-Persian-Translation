import argparse
from pprint import pprint

from langmodel import LMTranslator
from langmodel2 import LMTranslator2


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='lm',
                        help='type of translation model: lm, lm2')
    parser.add_argument('--load_model', action='store_true', default=False,
                        help='use if you want to load the model from file')
    parser.add_argument('--train', action='store_true',
                        default=False, help='use if you want to train the model')
    parser.add_argument('--test', action='store_true', default=False,
                        help='use if you want the model to report performance on a held-out test set')
    parser.add_argument('--data_path', default='./data/mizan/',
                        help='directory where training data is stored')
    parser.add_argument('--models_dir', default='./models_dir/',
                        help='directory where models are saved to/loaded from')

    # Language model parameters:

    # LM2 parameters
    parser.add_argument('--lm2_batch_size', default=32,
                        type=int, help='batch size for fine-tunning lm2 model')
    parser.add_argument('--lm2_lr', default=1e-5,
                        type=float, help='learning rate for fine-tunning lm2 model')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    pprint(f'Arguments are: {vars(args)}')  # For debugging

    if args.model == 'lm':
        model = LMTranslator(args)
    elif args.model == 'lm2':
        model = LMTranslator2(args)
    else:
        raise ValueError(f'Model "{args.model}" is not supported')

    if args.train:
        print('Training started...')
        model.train(args)
        print('Training finished.')
    
    if args.test:
        print('Testing started...')
        model.test(args)
        print('Testing finished.')
    
    while True:
        print('Input a sentence to be translated:')
        print(model.translate(input()))
