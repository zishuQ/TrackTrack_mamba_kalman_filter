import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def linear_interpolation(input_, interval):
    input_ = input_[np.lexsort([input_[:, 0], input_[:, 1]])]
    output_ = input_.copy()

    id_pre, f_pre, row_pre = -1, -1, np.zeros((10,))

    for row in input_:
        f_curr, id_curr = row[:2].astype(int)

        if id_curr == id_pre:
            if f_pre + 1 < f_curr < f_pre + interval:
                for i, f in enumerate(range(f_pre + 1, f_curr), start=1):
                    step = (row - row_pre) / (f_curr - f_pre) * i
                    row_new = row_pre + step
                    output_ = np.append(output_, row_new[np.newaxis, :], axis=0)
        else:
            id_pre = id_curr

        row_pre = row
        f_pre = f_curr

    output_ = output_[np.lexsort([output_[:, 0], output_[:, 1]])]

    return output_


def gradient_boosting_smooth(input_, tau):
    output_ = list()
    ids = set(input_[:, 1])

    for id_ in ids:
        tracks = input_[input_[:, 1] == id_]
        t = tracks[:, 0].reshape(-1, 1)
        x = tracks[:, 2].reshape(-1, 1)
        y = tracks[:, 3].reshape(-1, 1)
        w = tracks[:, 4].reshape(-1, 1)
        h = tracks[:, 5].reshape(-1, 1)

        regr = GradientBoostingRegressor(n_estimators=115, learning_rate=0.065, min_samples_split=6)

        regr.fit(t, x[:, 0])
        xx = regr.predict(t)
        regr.fit(t, y[:, 0])
        yy = regr.predict(t)
        regr.fit(t, w[:, 0])
        ww = regr.predict(t)
        regr.fit(t, h[:, 0])
        hh = regr.predict(t)

        output_.extend([[t[i, 0], id_, xx[i], yy[i], ww[i], hh[i], 1, -1, -1, -1] for i in range(len(t))])

    return output_


def gb_interpolation(path_in, path_out, interval, tau):
    input_ = np.loadtxt(path_in, delimiter=',')
    li_result = linear_interpolation(input_, interval)
    gbi_result = gradient_boosting_smooth(li_result, tau)
    np.savetxt(path_out, gbi_result, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')


def linear_interpolation_only(path_in, path_out, n_min=5, n_dti=20):
    """
    Linear interpolation for SportsMOT (based on MixSort implementation).
    
    Fills gaps between disconnected track segments with linear interpolation.
    This is simpler than GBI (no gradient boosting smoothing) but effective
    for fast-moving sports scenarios.
    
    Args:
        path_in: Input tracking result file path
        path_out: Output file path
        n_min: Minimum track length to keep (default: 5, from MixSort)
        n_dti: Maximum gap to interpolate (default: 20 frames, from MixSort)
    """
    input_ = np.loadtxt(path_in, delimiter=',')
    
    if len(input_) == 0:
        np.savetxt(path_out, input_, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')
        return
    
    # Sort by track_id, then frame_id
    input_ = input_[np.lexsort([input_[:, 0], input_[:, 1]])]
    
    # Filter short tracks
    track_ids = np.unique(input_[:, 1])
    filtered_rows = []
    for tid in track_ids:
        track = input_[input_[:, 1] == tid]
        if len(track) >= n_min:
            filtered_rows.append(track)
    
    if len(filtered_rows) == 0:
        np.savetxt(path_out, np.array([]).reshape(0, 10), fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')
        return
    
    input_ = np.vstack(filtered_rows)
    input_ = input_[np.lexsort([input_[:, 0], input_[:, 1]])]
    
    # Linear interpolation
    output_ = input_.copy()
    id_pre, f_pre, row_pre = -1, -1, np.zeros((10,))
    
    for row in input_:
        f_curr, id_curr = int(row[0]), int(row[1])
        
        if id_curr == id_pre:
            # Same track, check if we need to interpolate
            gap = f_curr - f_pre
            if 1 < gap <= n_dti:
                # Interpolate the gap
                for i, f in enumerate(range(f_pre + 1, f_curr), start=1):
                    alpha = i / gap
                    row_new = row_pre * (1 - alpha) + row * alpha
                    row_new[0] = f  # Frame id must be integer
                    row_new[1] = id_curr  # Track id must be integer
                    output_ = np.vstack([output_, row_new])
        
        id_pre = id_curr
        row_pre = row.copy()
        f_pre = f_curr
    
    # Sort output by frame_id, then track_id
    output_ = output_[np.lexsort([output_[:, 1], output_[:, 0]])]
    
    np.savetxt(path_out, output_, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')

