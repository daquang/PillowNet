import numpy as np


def load_meme(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    num_lines = len(lines)
    i = 0
    ppms = []
    identifiers = []
    names = []
    nsites_list = []
    d = None
    while i < num_lines:
        line = lines[i]
        if 'ALPHABET' in line:
            alpha_str = line.split()[-1].strip()
            d = np.array(list(alpha_str))
        if 'MOTIF' in line:
            name_info = line.split()
            identifier = name_info[1]
            if len(name_info) > 2:
                name = name_info[2]
            else:
                name = None
            while 'letter-probability matrix' not in line:
                i += 1
                line = lines[i]
            motif_info = lines[i]
            motif_info = motif_info.split()
            w_index = motif_info.index('w=') + 1
            w = int(motif_info[w_index])
            nsites_index = motif_info.index('nsites=') + 1
            nsites = int(motif_info[nsites_index])
            motif = np.zeros((len(d), w))
            i += 1
            line = lines[i]
            while len(line.strip()) == 0:
                i += 1
                line = lines[i]
            for j in range(w):
                motif[:, j] = np.array(lines[i].split(), dtype=float)
                i += 1
            ppm = np.dot(motif, np.diag(1/motif.sum(axis=0)))
            ppm = ppm.T
            ppms.append(ppm)
            nsites_list.append(nsites)
            names.append(name)
            identifiers.append(identifier)
        i += 1
    return ppms, d, names, identifiers, nsites_list
