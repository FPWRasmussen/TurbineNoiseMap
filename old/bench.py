from line_profiler import LineProfiler

lp = LineProfiler()
from main_test import main
main = lp(main)
main()
lp.print_stats()