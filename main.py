import argparse
from pprint import pprint

from langmodel import LMTranslator

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='lm',
                        help='type of translation model: lm')
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

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    pprint(f'Arguments are: {vars(args)}')  # For debugging

    if args.model == 'lm':
        model = LMTranslator(args)
    else:
        raise ValueError(f'Model "{args.model}" is not supported')

    # if args.train:
    #     model.train(args)
    #
    # if args.test:
    #     model.test(args)
    #
    # while True:
    #     print(model.translate(input()))
