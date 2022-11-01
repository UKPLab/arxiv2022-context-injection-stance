import numpy


def get_text_chunks(text, chunk_size):
    splitted_text = text.split(" ")

    if len(splitted_text) > chunk_size:
        n_chunks = round(len(splitted_text) / chunk_size)
        return [
            " ".join(chunk_tokens)
            for chunk_tokens in numpy.array_split(splitted_text, n_chunks)
        ]
    else:
        return [text]