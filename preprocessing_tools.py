import numpy as np 
import torch
import pywt


def interleave(a, b):
    '''
    Interleaves 2 numpy arrays
    Expected b.shape[1] <= a.shape[1]
    '''
    l1, l2 = a.shape[1], b.shape[1]
    c = np.zeros((a.shape[0], l1 + l2), dtype=a.dtype)
    a_p1 = a[:, :l2]
    c_p1 = c[:, :l2 * 2]
    c_p1[:, 0::2] = a_p1
    c_p1[:, 1::2] = b
    c[:, l2 * 2:] = a[:, l2:]
    return c

def discretizate_array(array, seg_start, seg_end, N):
    # searching segments bounds
    bin_edges = np.linspace(seg_start, seg_end, N+1)
    # searching bins indices
    bin_indices = np.digitize(array, bin_edges)
    return bin_indices


def build_batch_features_obj(
        obj, images, labels, n_levels, coefs_prop=0.4, 
        use_original_data=False, add_approx=True
    ):
    '''
    mlist: List[int] - m biggest coeeficients for every transformation level
    '''
    obj.labels = labels
    obj.res_coefs_count = []
    obj.channel_wise_coeffs = [
        pywt.wavedec2(images[:, :, :, 0], wavelet='haar', level=n_levels),
        pywt.wavedec2(images[:, :, :, 1], wavelet='haar', level=n_levels),
        pywt.wavedec2(images[:, :, :, 2], wavelet='haar', level=n_levels)
    ]
    
    res_tok_repr, res_coef_repr = None, None
    ch_count = 3

    for lvl in range(n_levels):
        lvl_tok_repr, lvl_coef_repr = None, None
        for ch in range(ch_count):
            ch_coeffs = obj.channel_wise_coeffs[ch]
            hf_c_list = list(ch_coeffs[-1 - lvl])
            tmi = max(int(hf_c_list[0].shape[1] * hf_c_list[0].shape[2] * coefs_prop), 1)
            if ch == 0:
                obj.res_coefs_count.append(tmi)
            ch_tok_repr, ch_coef_repr = None, None
            for coef_mat_no in range(len(hf_c_list)):
                coef_mat = hf_c_list[coef_mat_no]
                resh_coefs = coef_mat.reshape(coef_mat.shape[0], -1)
                if tmi < resh_coefs.shape[1]:
                    arg_part = np.argpartition(resh_coefs, -tmi, axis=1)[:, -tmi:]
                else:
                    arg_part = np.tile(range(resh_coefs.shape[1]), (resh_coefs.shape[0], 1))
                coefs = np.take_along_axis(resh_coefs, arg_part, axis=1)

                # Сдвигаем токены позиций для размещения специальных токенов
                column_to_insert = np.full((images.shape[0], ), n_levels + ch_count + coef_mat_no)
                arg_part = np.insert(
                    arg_part  + n_levels + ch_count + len(hf_c_list), 
                    0, column_to_insert, axis=1
                )
                ch_tok_repr = (arg_part if ch_tok_repr is None else 
                                np.concatenate((ch_tok_repr, arg_part), axis=1))
                ch_coef_repr = (coefs if ch_coef_repr is None else 
                                 np.concatenate((ch_coef_repr, coefs), axis=1))

            column_to_insert = np.full((images.shape[0], ), n_levels + ch)
            ch_tok_repr = np.insert(ch_tok_repr, 0, column_to_insert, axis=1)
            lvl_tok_repr = (ch_tok_repr if lvl_tok_repr is None else 
                           np.concatenate((lvl_tok_repr, ch_tok_repr), axis=1))
            lvl_coef_repr = (ch_coef_repr if lvl_coef_repr is None else 
                            np.concatenate((lvl_coef_repr, ch_coef_repr), axis=1))
        
        column_to_insert = np.full((images.shape[0], ), lvl)
        lvl_tok_repr = np.insert(lvl_tok_repr, 0, column_to_insert, axis=1)
        res_tok_repr = (lvl_tok_repr if res_tok_repr is None else 
                        np.concatenate((res_tok_repr, lvl_tok_repr), axis=1))
        res_coef_repr = (lvl_coef_repr if res_coef_repr is None else 
                         np.concatenate((res_coef_repr, lvl_coef_repr), axis=1))

    obj.tok_repr = res_tok_repr
    obj.coef_repr = res_coef_repr
    obj.wavelet_feature_size = obj.coef_repr.shape[1]
    
    if add_approx:
        obj.approx_segments = []
        last_end = obj.coef_repr.shape[1]
        for ch in range(ch_count):
            ch_approx = obj.channel_wise_coeffs[ch][0].reshape((images.shape[0], -1))
            print('approx shape is ', ch_approx.shape)
            obj.coef_repr = np.concatenate((obj.coef_repr, ch_approx), axis=1)
            obj.approx_segments.append((last_end, obj.coef_repr.shape[1]))
            last_end = obj.coef_repr.shape
        print(f'Added approximation segments: {obj.approx_segments}')

    if use_original_data:
        obj.coef_repr = np.concatenate(
            (obj.coef_repr, images.mean(axis=3).reshape(
                (images.shape[0], images.shape[1] * images.shape[2])
            )),
            axis=1
        )

        
def build_batch_features(
        images, labels, n_levels, coefs_prop=0.1, ch_count=3, 
        use_original_data=False, add_approx=True, return_tensor=False,
        n_bins=10
    ):
    '''
    mlist: List[int] - m biggest coeeficients for every transformation level
    '''
    res_coefs_count = []
    channel_wise_coeffs = [
        pywt.wavedec2(images[:, :, :, 0], wavelet='haar', level=n_levels),
        pywt.wavedec2(images[:, :, :, 1], wavelet='haar', level=n_levels),
        pywt.wavedec2(images[:, :, :, 2], wavelet='haar', level=n_levels)
    ]
    
    res_tok_repr = None

    total_token_count = n_levels + ch_count + 3 + 32 * 32 + n_bins
    for lvl in range(n_levels):
        lvl_tok_repr = None
        for ch in range(ch_count):
            ch_coeffs = channel_wise_coeffs[ch]
            hf_c_list = list(ch_coeffs[-1 - lvl])
            tmi = max(int(hf_c_list[0].shape[1] * hf_c_list[0].shape[2] * coefs_prop), 1)
            if ch == 0:
                res_coefs_count.append(tmi)
            ch_tok_repr = None
            for coef_mat_no in range(len(hf_c_list)):
                coef_mat = hf_c_list[coef_mat_no]
                resh_coefs = coef_mat.reshape(coef_mat.shape[0], -1)
                if tmi < resh_coefs.shape[1]:
                    arg_part = np.argpartition(resh_coefs, -tmi, axis=1)[:, -tmi:]
                else:
                    arg_part = np.tile(range(resh_coefs.shape[1]), (resh_coefs.shape[0], 1))

                # Coefs discretization and building tokens features
                coefs = np.take_along_axis(resh_coefs, arg_part, axis=1)
                discr_coefs = discretizate_array(coefs, 0., 1., n_bins)
                # Shifting coefficients tokens. It's important for tokens specification
                discr_coefs += n_levels + ch_count + len(hf_c_list) + 32 * 32
                arg_part += n_levels + ch_count + len(hf_c_list)
                tokens = interleave(arg_part, discr_coefs)

                # Сдвигаем токены позиций для размещения специальных токенов
                column_to_insert = np.full((images.shape[0], ), n_levels + ch_count + coef_mat_no)
                tokens = np.insert(
                    tokens, 
                    0, column_to_insert, axis=1
                )
                ch_tok_repr = (tokens if ch_tok_repr is None else 
                                np.concatenate((ch_tok_repr, tokens), axis=1))

            column_to_insert = np.full((images.shape[0], ), n_levels + ch)
            ch_tok_repr = np.insert(ch_tok_repr, 0, column_to_insert, axis=1)
            lvl_tok_repr = (ch_tok_repr if lvl_tok_repr is None else 
                           np.concatenate((lvl_tok_repr, ch_tok_repr), axis=1))
        
        column_to_insert = np.full((images.shape[0], ), lvl)
        lvl_tok_repr = np.insert(lvl_tok_repr, 0, column_to_insert, axis=1)
        res_tok_repr = (lvl_tok_repr if res_tok_repr is None else 
                        np.concatenate((res_tok_repr, lvl_tok_repr), axis=1))

    tok_repr = res_tok_repr

    if add_approx:
        approx_token = n_levels + ch_count + 3 + 32 ** 2 + n_bins + 2
        column_to_insert = np.full(tok_repr.shape[0], approx_token)
        tok_repr = np.insert(tok_repr, tok_repr.shape[1], column_to_insert, axis=1)
        for ch in range(ch_count):
            ch_approx = channel_wise_coeffs[ch][0].reshape((images.shape[0], -1))
            ch_approx = discretizate_array(ch_approx, 0., 1., n_bins)
            ch_approx += n_levels + ch_count + 3 + 32 * 32
            column_to_insert = np.full(tok_repr.shape[0], n_levels + ch)
            ch_approx = np.insert(ch_approx, 0, column_to_insert, axis=1)
            tok_repr = np.concatenate((tok_repr, ch_approx), axis=1)
        total_token_count += 1
    
    if use_original_data:
        original_data_token = n_levels + ch_count + 3 + 32 ** 2 + n_bins + 3
        column_to_insert = np.full(tok_repr.shape[0], original_data_token)
        additional_data = images.mean(axis=3).reshape((images.shape[0], -1))
        additional_data = discretizate_array(additional_data, 0., 1., n_bins)
        additional_data += n_levels + ch_count + 3 + 32 * 32
        additional_data = np.insert(additional_data, 0, column_to_insert, axis=1)
        tok_repr = np.concatenate((tok_repr, additional_data), axis=1)
        total_token_count += 1



    if return_tensor:
        tok_repr = torch.tensor(tok_repr)

    return tok_repr, labels, total_token_count
