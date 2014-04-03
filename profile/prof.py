import pstats
pstats.Stats('info').strip_dirs().sort_stats("cumulative").print_stats()