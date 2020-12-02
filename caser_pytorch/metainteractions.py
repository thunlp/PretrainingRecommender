"""
Classes describing datasets of user-item interactions. Instances of these
are returned by dataset-fetching and dataset-processing functions.
"""

import numpy as np
import pickle
import scipy.sparse as sp


class MetaInteractions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions. This is designed only for implicit feedback scenarios.

    Parameters
    ----------

    file_path: file contains (user,item,rating) triplets
    user_map: dict of user mapping
    item_map: dict of item mapping
    """

    def __init__(self, file_path, meta_path,
                 user_map=None,
                 item_map=None):

        if not user_map and not item_map:
            user_map = dict()
            item_map = dict()

            num_user = 0
            num_item = 0
        else:
            num_user = len(user_map)
            num_item = len(item_map)

        user_ids = list()
        item_ids = list()
        dire_ids = list()
        acto_ids = list()
        meta = pickle.load(open(meta_path, 'rb'))

        # read users and items from file
        with open(file_path, 'r') as fin:
            for line in fin:
                u, i, _ = line.strip().split()
                user_ids.append(int(u))
                item_ids.append(int(i))
                if int(i) in meta:
                    dire_ids.append(meta[int(i)][0][0])
                    act = meta[int(i)][1]
                    if len(act)<4:
                        for ki in range(4-len(act)):
                            act.append(4170)
                    acto_ids.append(act)
                else:
                    dire_ids.append(1186)
                    acto_ids.append([4170,4170,4170,4170])

        # update user and item mapping
        for u in user_ids:
            if u not in user_map:
                user_map[u] = num_user
                num_user += 1
        for i in item_ids:
            if i not in item_map:
                item_map[i] = num_item
                num_item += 1

        user_ids = np.array([user_map[u] for u in user_ids])
        item_ids = np.array([item_map[i] for i in item_ids])

        self.num_users = num_user
        self.num_items = num_item

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.dire_ids = dire_ids
        self.acto_ids = acto_ids
        self.user_map = user_map
        self.item_map = item_map

        self.sequences = None
        self.test_sequences = None

    def __len__(self):

        return len(self.user_ids)

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """
        #import pdb
        #pdb.set_trace()
        row = self.user_ids
        col = self.item_ids
        data = np.ones(len(self))
        
        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()

    def to_sequence(self, sequence_length=5, target_length=1):
        """
        Transform to sequence form.

        Valid subsequences of users' interactions are returned. For
        example, if a user interacted with items [1, 2, 3, 4, 5, 6, 7, 8, 9], the
        returned interactions matrix at sequence length 5 and target length 3
        will be be given by:

        sequences:

           [[1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]]

        targets:

           [[6, 7],
            [7, 8],
            [8, 9]]

        sequence for test (the last 'sequence_length' items of each user's sequence):

        [[5, 6, 7, 8, 9]]

        Parameters
        ----------

        sequence_length: int
            Sequence length. Subsequences shorter than this
            will be left-padded with zeros.
        target_length: int
            Sequence target length.
        """

        # change the item index start from 1 as 0 is used for padding in sequences
        for k, v in self.item_map.items():
            self.item_map[k] = v + 1
        self.item_ids = self.item_ids + 1
        self.num_items += 1

        max_sequence_length = sequence_length + target_length

        # Sort first by user id
        sort_indices = np.lexsort((self.user_ids,))

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]
        dire_ids = np.array(self.dire_ids)[sort_indices]
        acto_ids = np.array(self.acto_ids)[sort_indices]

        user_ids, indices, counts = np.unique(user_ids,
                                              return_index=True,
                                              return_counts=True)

        num_subsequences = sum([c - max_sequence_length + 1 if c >= max_sequence_length else 1 for c in counts])

        sequences = np.zeros((num_subsequences, sequence_length),
                             dtype=np.int64)
        sequences_dire = np.zeros((num_subsequences, sequence_length),
                             dtype=np.int64)
        sequences_acto = np.zeros((num_subsequences, sequence_length, 4),
                             dtype=np.int64)
        sequences_targets = np.zeros((num_subsequences, target_length),
                                     dtype=np.int64)
        sequence_users = np.empty(num_subsequences,
                                  dtype=np.int64)

        test_sequences = np.zeros((self.num_users, sequence_length),
                                  dtype=np.int64)
        test_sequences_dire = np.zeros((self.num_users, sequence_length),
                                  dtype=np.int64)
        test_sequences_acto = np.zeros((self.num_users, sequence_length, 4),
                                  dtype=np.int64)
        test_users = np.empty(self.num_users,
                              dtype=np.int64)

        _uid = None
        
        for i, (uid,
                item_seq, dire_seq, acto_seq) in enumerate(_generate_sequences(user_ids,
                                                           item_ids,
                                                           dire_ids,
                                                           acto_ids,
                                                           indices,
                                                           max_sequence_length)):
            if uid != _uid:
                #if acto_seq[-sequence_length:][:].shape!=test_sequences_acto[uid][:][:].shape:
                #    import pdb
                #    pdb.set_trace()
                test_sequences[uid][:] = item_seq[-sequence_length:]
                test_sequences_dire[uid][:] = dire_seq[-sequence_length:]
                test_sequences_acto[uid][:][:] = acto_seq[-sequence_length:][:]
                test_users[uid] = uid
                _uid = uid
            sequences_targets[i][:] = item_seq[-target_length:]
            sequences[i][:] = item_seq[:sequence_length]
            sequence_users[i] = uid
            sequences_dire[i][:] = dire_seq[:sequence_length]
            sequences_acto[i][:][:] = acto_seq[:sequence_length][:]

        self.sequences = SequenceMetaInteractions(sequence_users, sequences, sequences_dire, sequences_acto, sequences_targets)
        self.test_sequences = SequenceMetaInteractions(test_users, test_sequences, test_sequences_dire, test_sequences_acto)


class SequenceMetaInteractions(object):
    """
    Interactions encoded as a sequence matrix.

    Parameters
    ----------
    user_ids: np.array
        sequence users
    sequences: np.array
        The interactions sequence matrix, as produced by
        :func:`~Interactions.to_sequence`
    targets: np.array
        sequence targets
    """

    def __init__(self,
                 user_ids,
                 sequences,
                 dire_seq,
                 acto_seq,
                 targets=None):
        self.user_ids = user_ids
        self.sequences = sequences
        self.targets = targets
        self.dire_seq = dire_seq
        self.acto_seq = acto_seq

        self.L = sequences.shape[1]
        self.T = None
        if np.any(targets):
            self.T = targets.shape[1]


def _sliding_window(tensor1, tensor2, tensor3, window_size, step_size=1):
    if len(tensor1) - window_size >= 0:
        for i in range(len(tensor1), 0, -step_size):
            if i - window_size >= 0:
                yield (tensor1[i - window_size:i], tensor2[i - window_size:i], tensor3[i - window_size:i])
            else:
                break
    else:
        num_paddings = window_size - len(tensor1)
        # Pad sequence with 0s if it is shorter than windows size.
        yield (np.pad(tensor1, (num_paddings, 0), 'constant'), np.pad(tensor2, (num_paddings, 0), 'constant'), np.pad(tensor3, ((num_paddings, 0), (0, 0)), 'constant') )


def _generate_sequences(user_ids, item_ids, dire_ids, acto_ids,
                        indices,
                        max_sequence_length):
    for i in range(len(indices)):

        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for (seq1, seq2, seq3) in _sliding_window(item_ids[start_idx:stop_idx], dire_ids[start_idx:stop_idx], acto_ids[start_idx:stop_idx], max_sequence_length):
            yield (user_ids[i], seq1, seq2, seq3)
