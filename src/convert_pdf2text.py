import argparse

from data_utils import (
    convert_pdf2text,
    PUB_PDF_PATH, PUB_TXT_PATH)

DATA_FOLDER = '../data/'
PDF_FOLDER = DATA_FOLDER + PUB_PDF_PATH
TEXT_FOLDER = DATA_FOLDER + PUB_TXT_PATH


def main(args):
    convert_pdf2text(args.input_dir, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert all pdf files \
                                     to text files')
    parser.add_argument('--input_dir', type=str,
                        default=PDF_FOLDER,
                        help='Folder containing rclc pdf files')
    parser.add_argument('--output_dir', type=str,
                        default=TEXT_FOLDER,
                        help='Folder to store rclc text files')
    args = parser.parse_args()
    main(args)
