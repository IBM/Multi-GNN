import time
import logging
from util import create_parser, set_seed, logger_setup
from data_loading import get_data
from training import train_gnn


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    logger_setup()

    #set seed
    set_seed(args.seed)

    #get data
    logging.info("Retrieving data")
    t1 = time.perf_counter()
    
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args)
    
    t2 = time.perf_counter()
    logging.info(f"Retrieved data in {t2-t1:.2f}s")
    
    #Training
    logging.info(f"Running Training")
    train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args)

if __name__ == "__main__":
    main()
