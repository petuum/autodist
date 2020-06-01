from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import six
import tensorflow as tf

from utils.recommendation.movielens import RATING_COLUMNS


def parse_args():
    parser = ArgumentParser(description="Train a Nerual Collaborative"
                                        " Filtering model converter")
    parser.add_argument('data', type=str,
                        help='path to test and training data files')
    parser.add_argument('--user_scaling', default=16, type=int)
    parser.add_argument('--item_scaling', default=32, type=int)
    parser.add_argument('--seed', '-s', default=0, type=int,
                        help='manually set random seed for numpy')
    return parser.parse_args()


def _transform_csv(data_array, output_path, names, separator=","):
    """Transform csv to a regularized format.

    Args:
      input_path: The path of the raw csv.
      output_path: The path of the cleaned csv.
      names: The csv column names.
      separator: Character used to separate fields in the raw csv.
    """
    if six.PY2:
        names = [six.ensure_text(n, "utf-8") for n in names]

    with tf.io.gfile.GFile(output_path, "wb") as f_out:
        # Write column names to the csv.
        f_out.write(",".join(names).encode("utf-8"))
        f_out.write(b"\n")
        counter = 0
        for chunk in data_array:
            for i in range(chunk.shape[0]):
                user_id = str(chunk[i][0] + 1)
                item_id = str(chunk[i][1] + 1)
                rating = str(1.0)
                timestamp = '1112484727\n'
                # line = six.ensure_text(line, "utf-8", errors="ignore")
                fields = [user_id, item_id, rating, timestamp]
                # fields = line.split(separator)
                if separator != ",":
                    fields = ['"{}"'.format(field) if "," in field else field
                              for field in fields]
                f_out.write(",".join(fields).encode("utf-8"))
                counter += 1
                if counter % 1000000 == 0:
                    print(
                        'Have written {} ({}%) record.'.format(
                            counter,
                            float(counter) /
                            1223962043.0 *
                            100))
                # if counter == 20000:
                #       return
    return


def process_raw_data(args):
        # train_ratings = [np.array([], dtype=np.int64)] * m
    train_ratings = [np.array([], dtype=np.int64)] * args.user_scaling
    # test_ratings_chunk = [np.array([], dtype=np.int64)] * args.user_scaling
    # test_chunk_size = [0] * args.user_scaling
    for chunk in range(args.user_scaling):
        # for chunk in range(m):
        print(
            datetime.now(),
            "Loading data chunk {} of {}".format(
                chunk + 1,
                args.user_scaling))
        train_ratings[chunk] = np.load(args.data +
                                       '/trainx' +
                                       str(args.user_scaling) +
                                       'x' +
                                       str(args.item_scaling) +
                                       '_' +
                                       str(chunk) +
                                       '.npz', encoding='bytes')['arr_0']
        # test_ratings_chunk[chunk] = np.load(args.data + '/testx'
        #         + str(args.user_scaling) + 'x' + str(args.item_scaling)
        #         + '_' + str(chunk) + '.npz', encoding='bytes')['arr_0']
        # test_chunk_size[chunk] = test_ratings_chunk[chunk].shape[0]

    # Due to the fractal graph expansion process, some generated users do not
    # have any ratings. Therefore, nb_users should not be max_user_index+1.
    nb_users_per_chunk = [len(np.unique(x[:, 0])) for x in train_ratings]
    nb_users = sum(nb_users_per_chunk)
    # nb_users = len(np.unique(train_ratings[:, 0]))

    nb_maxs_per_chunk = [np.max(x, axis=0)[1] for x in train_ratings]
    # Zero is valid item in output from expansion
    nb_items = max(nb_maxs_per_chunk) + 1

    nb_train_elems = sum([x.shape[0] for x in train_ratings])

    print(
        datetime.now(),
        "Number of users: {}, Number of items: {}".format(
            nb_users,
            nb_items))
    print(datetime.now(), "Number of ratings: {}".format(nb_train_elems))

    csv_file_name = args.data + '/train_ratings.csv'
    print(datetime.now(), "Now write to csv file: {}.".format(csv_file_name))
    _transform_csv(train_ratings, csv_file_name, names=RATING_COLUMNS)
    # train_input = [npi.group_by(x[:, 0]).split(x[:, 1]) for x in train_ratings]
    # def iter_fn_simple():
    #     for train_chunk in train_input:
    #         for _, items in enumerate(train_chunk):
    #             yield items
    #
    # sampler, pos_users, pos_items  = process_data(
    #     num_items=nb_items, min_items_per_user=1, iter_fn=iter_fn_simple)
    # assert len(pos_users) == nb_train_elems, "Cardinality difference with original data and sample table data."
    #
    # print("pos_users type: {}, pos_items type: {}, s.offsets: {}".format(
    #       pos_users.dtype, pos_items.dtype, sampler.offsets.dtype))
    # print("num_reg: {}, region_card: {}".format(sampler.num_regions.dtype,
    #       sampler.region_cardinality.dtype))
    # print("region_starts: {}, alias_index: {}, alias_p: {}".format(
    #       sampler.region_starts.dtype, sampler.alias_index.dtype,
    #       sampler.alias_split_p.dtype))
    #
    # fn_prefix = args.data + '/' + CACHE_FN.format(args.user_scaling, args.item_scaling)
    # sampler_cache = fn_prefix + "cached_sampler.pkl"
    # with open(sampler_cache, "wb") as f:
    #     pickle.dump([sampler, pos_users, pos_items, nb_items, test_chunk_size], f, pickle.HIGHEST_PROTOCOL)
    print(datetime.now(), "Written done.")
    return


def main():
    args = parse_args()
    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        np.random.seed(seed=args.seed)

    process_raw_data(args)


if __name__ == '__main__':
    main()
