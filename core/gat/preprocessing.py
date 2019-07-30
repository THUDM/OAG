from os.path import join
import os
import numpy as np


from core.utils import data_utils
from core.utils import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


def gen_paired_subgraph():
    pos_pairs = data_utils.load_json(settings.AUTHOR_DATA_DIR, 'pos_person_pairs.json')
    neg_pairs = data_utils.load_json(settings.AUTHOR_DATA_DIR, 'neg_person_pairs.json')

    adjs = []
    vertex_ids = []
    vertex_types = []
    n_nodes_ego = 382
    pairs = pos_pairs + neg_pairs
    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

    aminer_person_to_venue_dict = data_utils.load_json(settings.AUTHOR_DATA_DIR, 'aminer_aid_to_vids.json')
    mag_person_to_venue_dict = data_utils.load_json(settings.AUTHOR_DATA_DIR, 'mag_aid_to_vids.json')
    aminer_person_to_sorted_papers = data_utils.load_json(settings.AUTHOR_DATA_DIR, 'aminer_aid_to_sorted_pubs.json')
    mag_person_to_sorted_papers = data_utils.load_json(settings.AUTHOR_DATA_DIR, 'mag_aid_to_sorted_pubs.json')
    aminer_coauthor_dict = data_utils.load_json(settings.AUTHOR_DATA_DIR, 'aminer_coauthor_dict.json')
    mag_coauthor_dict = data_utils.load_json(settings.AUTHOR_DATA_DIR, 'mag_coauthor_dict.json')

    aid_to_vids = {**aminer_person_to_venue_dict, **mag_person_to_venue_dict}
    aid_to_pids = {**aminer_person_to_sorted_papers, **mag_person_to_sorted_papers}
    coauthor_dict_cache = {**aminer_coauthor_dict, **mag_coauthor_dict}

    ego_person_dict = data_utils.load_json(settings.AUTHOR_DATA_DIR, 'ego_person_dict.json')

    for i, pair in enumerate(pairs):
        node_set = set()
        node_list = []
        node_type_list = []
        n_authors = 0
        n_venues = 0
        n_papers = 0
        aaid, maid = pair['aid'], pair['mid']
        aaid_dec = '{}-aa'.format(aaid)
        maid_dec = '{}-am'.format(maid)
        n_authors += 2
        focal_node1 = aaid
        focal_node2 = maid
        node_set.update({aaid, maid})
        node_list += [aaid_dec, maid_dec]
        node_type_list += [settings.AUTHOR_TYPE, settings.AUTHOR_TYPE]

        # 1-ego venues
        a_venues = ego_person_dict.get(aaid, {}).get('venue', [])[:10]
        m_venues = ego_person_dict.get(maid, {}).get('venue', [])[:10]
        venues_1_ego = [v['id'] for v in a_venues + m_venues]
        for v in venues_1_ego:
            if v not in node_set:
                node_set.add(v)
                v_dec = '{}-v'.format(v)
                n_venues += 1
                node_list.append(v_dec)
                node_type_list.append(settings.VENUE_TYPE)

        # 1-ego papers
        a_pubs = ego_person_dict.get(aaid, {}).get('pubs', [])[:20]
        m_pubs = ego_person_dict.get(maid, {}).get('pubs', [])[:20]
        pubs_1_ego = {item for item in a_pubs + m_pubs if item not in node_set}
        node_set.update(pubs_1_ego)
        node_list += ['{}-p'.format(item) for item in pubs_1_ego]
        node_type_list += [settings.PAPER_TYPE] * len(pubs_1_ego)
        n_papers += len(pubs_1_ego)

        a_coauthors = aminer_coauthor_dict.get(aaid, [])[:10]
        for cur_aid in a_coauthors:
            if cur_aid not in node_set:
                node_set.add(cur_aid)
                node_list.append('{}-aa'.format(cur_aid))
                node_type_list.append(settings.AUTHOR_TYPE)
                n_authors += 1

        m_coauthors = mag_coauthor_dict.get(maid, [])[:10]
        m_coauthors = {a for a in m_coauthors if a not in node_set}
        node_set.update(m_coauthors)
        node_list += ['{}-am'.format(a) for a in m_coauthors]
        node_type_list += [settings.AUTHOR_TYPE] * len(m_coauthors)
        n_authors += len(m_coauthors)

        # 2-ego
        for a in a_coauthors:
            cur_venues = aminer_person_to_venue_dict.get(a, [])[:5]
            cur_vids = {v['id'] for v in cur_venues if v['id'] not in node_set}
            node_set.update(cur_vids)
            node_list += ['{}-v'.format(v) for v in cur_vids]
            node_type_list += [settings.VENUE_TYPE] * len(cur_vids)
            n_venues += len(cur_vids)

        for a in m_coauthors:
            cur_venues = mag_person_to_venue_dict.get(a, [])[:5]
            cur_vids = {v['id'] for v in cur_venues if v['id'] not in node_set}
            node_set.update(cur_vids)
            node_list += ['{}-v'.format(v) for v in cur_vids]
            node_type_list += [settings.VENUE_TYPE] * len(cur_vids)
            n_venues += len(cur_vids)

        a_pubs_2 = []
        for a in a_coauthors:
            cur_pubs = aminer_person_to_sorted_papers.get(a, [])[:10]
            a_pubs_2 += cur_pubs

        m_pubs_2 = []
        for a in m_coauthors:
            cur_pubs = mag_person_to_sorted_papers.get(a, [])[:10]
            m_pubs_2 += cur_pubs

        pubs_2_ego = {item for item in a_pubs_2 + m_pubs_2 if item not in node_set}
        node_set.update(pubs_2_ego)
        node_list += ['{}-p'.format(p) for p in pubs_2_ego]
        node_type_list += [settings.PAPER_TYPE] * len(pubs_2_ego)
        n_papers += len(pubs_2_ego)

        cur_n_nodes_real = len(node_list)
        assert len(node_set) == len(node_list) == len(node_type_list)
        assert n_authors <= 22 and n_venues <= 120 and n_papers <= 240

        # padding
        for v_idx in range(len(node_set), n_nodes_ego):
            node_list.append('-1')
            node_type_list.append(settings.AUTHOR_TYPE)
        assert len(node_list) == n_nodes_ego

        vertex_ids.append(node_list)
        vertex_types.append(node_type_list)

        node_to_idx = {eid.split('-')[0]: i for i, eid in enumerate(node_list)}
        adj = np.zeros((n_nodes_ego, n_nodes_ego), dtype=np.bool_)
        nnz = 0
        for ii in range(cur_n_nodes_real):
            v1 = node_list[ii].split('-')[0]
            t1 = node_type_list[ii]
            if t1 == settings.AUTHOR_TYPE:
                v_nbrs = aid_to_vids.get(v1, [])
                v_nbrs = {item['id'] for item in v_nbrs}
                v_nbrs_filtered = {item for item in v_nbrs if item in node_set}
                nbrs_set = v_nbrs_filtered

                p_nbrs = aid_to_pids.get(v1, [])
                p_nbrs_filtered = {item for item in p_nbrs if item in node_set}
                nbrs_set.update(p_nbrs_filtered)

                a_nbrs = coauthor_dict_cache.get(v1, [])
                a_nbrs_filtered = {item for item in a_nbrs if item in node_set}
                nbrs_set.update(a_nbrs_filtered)

                for nbr in nbrs_set:
                    nbr_idx_map = node_to_idx[nbr]
                    adj[ii, nbr_idx_map] = True
                    adj[nbr_idx_map, ii] = True
                    nnz += 1
        adjs.append(adj)

        if i % 1000 == 0:
            logger.info('***********iter %d nnz %d', i, nnz)
            logger.info('n_nodes real %d, n_authors %d, n_venues %d, n_papers %d.',
                        cur_n_nodes_real, n_authors, n_venues, n_papers)

    out_dir = settings.AUTHOR_DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    np.save(join(out_dir, 'vertex_id.npy'), np.array(vertex_ids))
    np.save(join(out_dir, 'vertex_types.npy'), np.array(vertex_types))
    np.save(join(out_dir, 'adjacency_matrix.npy'), np.array(adjs))
    np.save(join(out_dir, 'label.npy'), np.array(labels))


if __name__ == '__main__':
    gen_paired_subgraph()
    logger.info('done')
