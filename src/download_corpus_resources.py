import argparse

from data_utils import (
    load_corpus, copy_pdf_resources, download_resources,
    PUB_PDF_PATH)

CORPUS_PATH = '../data/corpus.jsonld'


def profile_corpus(corpus):
    print(f'Number of records: {len(corpus)}')
    pubs = [e for e in corpus if e['@type'] == 'ResearchPublication']
    print(f'Number of research publications: {len(pubs)}')
    datasets = [e for e in corpus if e['@type'] == 'Dataset']
    print(f'Number of datasets: {len(datasets)}')


def prepare_resources(corpus: object) -> None:
    data_folder = '../data/'
    # copy_pdf_resources(data_folder + 'resource/manual_download/',
    #                    data_folder + PUB_PDF_PATH)
    download_resources(corpus, data_folder)


def main(args):
    corpus = load_corpus(args.input)
    prepare_resources(corpus)
    profile_corpus(corpus)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download open access \
        rclc corpus')
    parser.add_argument('--input', type=str,
                        default=CORPUS_PATH,
                        help='rclc corpus file')
    args = parser.parse_args()
    main(args)
