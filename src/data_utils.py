import json
import os
import requests
import shutil
import time

from bs4 import BeautifulSoup  # type: ignore
from pathlib import Path
from pdfminer.pdfdocument import PDFDocument  # type: ignore
from pdfminer.pdfpage import PDFTextExtractionNotAllowed  # type: ignore
from pdfminer.pdfparser import PDFParser, PDFSyntaxError  # type: ignore
from readability import Document  # type: ignore
from requests_html import HTMLSession  # type: ignore
from tqdm import tqdm  # type: ignore
from urllib.parse import urlparse

from rclc_conf import CACHE_PUB_FILE


MAX_DOWNLOAD_TRIAL = 3

RES_PATH = 'resource/'
PUB_PATH = RES_PATH + 'pubs/'
DATASET_PATH = RES_PATH + 'dataset/'

PUB_PDF_PATH = PUB_PATH + 'pdf/'
PUB_TXT_PATH = PUB_PATH + 'text/'
DATASET_PAGE_PATH = DATASET_PATH + 'html/'


def load_corpus(filename: str) -> dict:
    """ Load corpus file (jsonld format)
    """
    with open(filename, 'r') as f:
        corpus = json.load(f)
        return corpus['@graph']


def json_from_file(filename: str) -> dict:
    data = None
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if data is None:
        raise ValueError('Error loading json from {}.'.format(filename))
    return data


def is_valid_pdf_file(filename: str) -> bool:
    try:
        with open(filename, 'rb') as f:
            parser = PDFParser(f)
            document = PDFDocument(parser, '')
            if not document.is_extractable:
                raise PDFTextExtractionNotAllowed(filename)
            return True
    except PDFSyntaxError as err:
        print(err)
        return False


def copy_pdf_resources(input_path: str,
                       output_path: str) -> None:
    try:
        for file in tqdm(Path(input_path).glob('*.pdf'),
                         ascii=True,
                         desc='copy files'):
            out_file = output_path + file.name
            if not Path(out_file).exists():
                in_file = file.resolve().as_posix()
                shutil.copy(in_file, out_file)
    except IOError as e:
        print(f'Unable to copy file: {e}')


def _download(uri: str, res_type: str, output_path: Path) -> bool:
    """ download a resource and store in a file
    """
    if res_type not in ['pdf', 'html']:
        print(f'Invalid resource type: {res_type}')
        return False
    trial = 0
    headers = requests.utils.default_headers()
    headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'

    while trial < MAX_DOWNLOAD_TRIAL:
        try:
            parsed_uri = urlparse(uri)
            if parsed_uri.netloc == 'www.sciencedirect.com':
                """ Special case: sciencedirect.com auto generates pdf download link in an intermediate page
                """
                session = HTMLSession()
                r0 = session.get(uri)
                res = session.get(list(r0.html.absolute_links)[0])
            elif parsed_uri.netloc.endswith('onlinelibrary.wiley.com'):
                """ Special case: wiley auto generates embed pdf to render pdf
                """
                r0 = requests.get(uri)
                soup = BeautifulSoup(r0.content, 'html5lib')
                src = soup.find('embed')['src']
                res = requests.get(parsed_uri.scheme + '://' +
                                   parsed_uri.netloc + src)
            else:
                res = requests.get(uri, headers=headers)
            output_path.write_bytes(res.content)
            if res_type == 'pdf':
                if not is_valid_pdf_file(output_path.resolve().as_posix()):
                    output_path.unlink()
                    return False
            return True
        except requests.exceptions.RequestException as err:
            print(err)
            time.sleep(5)
            trial += 1

    if trial == MAX_DOWNLOAD_TRIAL:
        print(f'Failed downloading {uri} after {MAX_DOWNLOAD_TRIAL} attempts.')
    return False


def download_resources(corpus: object,
                       output_path: str,
                       force_download: bool = False) -> None:
    """ Download all publications pdf file and dataset html file from corpus
        data (if not downloaded yet).
        All downloaded files are stored under `resource/` folder in the
        `output_path`, organized separatedly for publication and datasets.
        We use the entity id as filename.
    """
    pub_pdf_full_path = Path(output_path + PUB_PDF_PATH)
    dataset_page_full_path = Path(output_path + DATASET_PAGE_PATH)
    if not pub_pdf_full_path.exists():
        pub_pdf_full_path.mkdir(parents=True)
    if not dataset_page_full_path.exists():
        dataset_page_full_path.mkdir(parents=True)

    for entity in tqdm(corpus, ascii=True, desc='Fetch resources'):
        _id = urlparse(entity['@id']).fragment.split('-')[1]
        _type = entity['@type']
        if _type == 'ResearchPublication':
            res_uri = entity['openAccess']['@value']
            output = Path(output_path + PUB_PDF_PATH + _id + '.pdf')
            if force_download or not output.exists():
                if not _download(res_uri, 'pdf', output):
                    print(f'Failed to download {res_uri}')
                time.sleep(0.5)
        elif _type == 'Dataset':
            res_uri = entity['foaf:page']['@value']
            output = Path(output_path + DATASET_PAGE_PATH + _id + '.html')
            if force_download or not output.exists():
                if not _download(res_uri, 'html', output):
                    print(f'Failed to download {res_uri}')
                time.sleep(0.5)
        else:
            raise Exception(f'Unknown Exception: {_type}')


def convert_pdf2text(input_path: str,
                     output_path: str) -> None:
    """ Convert all pdf files in input_path into text files stored \
        in output_path.
    """
    for file in tqdm(Path(input_path).glob('*.pdf'), ascii=True,
                     desc='pdftotxt'):
        txt_file = output_path + file.name + '.txt'
        if not is_valid_pdf_file(input_path + file.name):
            print(input_path + file.name)
            continue

        if not Path(txt_file).exists():
            # cmd = f'pdf2txt.py -o {txt_file} {input_path + file.name}'
            cmd = f'pdftotext {input_path + file.name} {txt_file}'
            print(cmd)
            os.system(cmd)


def _clean_html_str(s: str) -> str:
    return s.replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ') \
        .replace("''", '"') \
        .replace('\u2018\u2018', "'").replace('\u2018', "'") \
        .replace('\u2019', "'").replace('\u201a', "'") \
        .replace('\u201c', '"').replace('\u201d', '"') \
        .replace('\u2013', " - ") \
        .strip()


def _extract_meta_tags(html: str) -> dict:
    """ Extract meta title and description from an html page
    """
    soup = BeautifulSoup(html, 'html5lib')
    og_title = soup.find('meta', {'property': 'og:title'})
    title = og_title['content'] if og_title else soup.title.string

    meta_desc = soup.find('meta', {'name': 'description'})
    description = meta_desc['content'] if meta_desc else None
    if description is None:
        og_desc = soup.find('meta', {'property': 'og:description'})
        description = og_desc['content'] if og_desc else None
    return {
        'title': _clean_html_str(title),
        'description': description
    }


def _extract_summary_content(html: str) -> str:
    """ Extract summary content from an html page
    """
    doc = Document(html)
    soup = BeautifulSoup(doc.summary(), 'html5lib')
    p_tags = soup.find_all('p')
    p_texts = [_clean_html_str(p.get_text()) for p in p_tags]
    content = ' '.join(p_texts)
    return content


def _parse_dataset_html(html_file: str) -> dict:
    """ Read dataset html file and parse important contents such as meta
        title, meta-description, and all main paragraph (extracted using
        readability extractor)
    """
    with open(html_file, 'r') as f:
        text = f.read()
    return {
        'meta': _extract_meta_tags(text),
        'summary': _extract_summary_content(text)
    }


def load_rclc_corpus(corpus_file: str,
                     html_path: str,
                     parsed_pub_path: str,
                     force_compute=False):
    """ This function loads rclc corpus and corresponding parsed publications
        for training.dataset. We expect the dataset html files and parsed publications (parsed using AllenAI science parser) are available.
        The loaded dataset is cached into a json file, and subsequent data
        load can utilize the cache file.
        Input:
            - corpus_file: location of `corpus.jsonld` file.
            - html_path: path of dataset html files
            - parsed_pub_path: path of parsed publication files.
            - force_compute: if True, then we recompute everything and ignore
                             cache file.
        Output:
            return an object containing dataset mapping index and a list of
            parsed publications
    """
    data_path = corpus_file[:corpus_file.rfind('/') + 1]
    # print(f'data path: {data_path}')

    cache_corpus_file = data_path + CACHE_PUB_FILE
    if os.path.isfile(cache_corpus_file) and not force_compute:
        return json_from_file(cache_corpus_file)

    corpus = load_corpus(corpus_file)
    datasets = [e for e in corpus if e['@type'] == 'Dataset']
    dataset_idx = {dataset['@id']: (i + 1)
                   for i, dataset in enumerate(datasets)}
    datasets = [e for e in corpus if e['@type'] == 'Dataset']
    for dataset in tqdm(datasets, ascii=True, desc='loading datasets'):
        _id = urlparse(dataset['@id']).fragment.split('-')[1]
        dataset['html'] = _parse_dataset_html(html_path + _id + '.html')

    pubs = [e for e in corpus if e['@type'] == 'ResearchPublication']
    for pub in tqdm(pubs, ascii=True, desc='loading pubs'):
        _id = urlparse(pub['@id']).fragment.split('-')[1]
        parsed_pub = json_from_file(parsed_pub_path + _id + '.pdf.json')
        pub['parsed_pub'] = parsed_pub

        citations = pub['cito:citesAsDataSource']
        if isinstance(citations, list):
            pub['citation_idx'] = [dataset_idx[c['@id']] for c in citations]
        elif isinstance(citations, dict):
            pub['citation_idx'] = [dataset_idx[citations['@id']]]
        else:
            ValueError(f'Unknown type: {type(citations)}')

    # cache consolidated info
    rclc_corpus = {
        'dataset_idx': dataset_idx,
        'datasets': datasets,
        'pubs': pubs
    }
    with open(cache_corpus_file, 'w') as f:
        f.write(json.dumps(rclc_corpus))
    return rclc_corpus


def load_rcc_cache_dataset(data_path: str) -> dict:
    """ This function loads cache dataset which contain parsed publication
        information and additional contextual information such as citation
        information and research methods
    """
    cache_file = data_path + CACHE_PUB_FILE
    if os.path.isfile(cache_file):
        return json_from_file(cache_file)
    else:
        raise ValueError(f'Cache file {cache_file} does not exist.')


def read_pub_text(text_file: str) -> str:
    with open(text_file, 'r', encoding='utf-8') as f:
        content = ''
        for line in f:
            content = content + line.rstrip('\n') + ' '
        return content[:-1]

# if __name__ == '__main__':
#     for f in tqdm(Path('../data/resource/dataset/html/').glob('*.html'),
#                   ascii=True,
#                   desc='parse HTML'):
#         print(f'{_parse_dataset_html(f.resolve())}')
