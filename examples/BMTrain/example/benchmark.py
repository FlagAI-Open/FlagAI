import bmtrain as bmt

def main():
    bmt.init_distributed()
    bmt.print_rank("======= All Gather =======")
    bmt.benchmark.all_gather()
    bmt.print_rank("===== Reduce Scatter =====")
    bmt.benchmark.reduce_scatter()
    

if __name__ == '__main__':
    main()