import cProfile
import pstats
import sys

filename = "dynamic_gen"
stat = pstats.Stats("./profiling/"+filename, stream=sys.stdout)
# stat.dump_stats('./profiling/modified_stats')
stat.sort_stats('cumtime')
stat.print_stats()