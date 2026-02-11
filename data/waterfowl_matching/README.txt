A_tilde is the incidence matrix for the bipartite graph, associated with 95 ducks and 339 edges (possible matching)

df_duck is the covariate matrix, associated with duck_species, weight of male, and weight of female for each edge of 339 edges

Z is the response matrix of size 18 x 339, each row is a weekly record of matchings, over 18 weeks.  since ducks are seasonal monogamous, you can see each column is non-decreasing -- i.e. one an edge is formed, it's not broken.

use the following to read in the files (in Python)
np.load("duck_data.pkl",allow_pickle=True)



