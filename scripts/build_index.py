import sys
import argparse
from pathlib import Path

from click import clear
from scipy import stats

src_path=Path(__file__).resolve().parent.parent/"src"
sys.path.insert(0,str(src_path))
from healthcare_rag.vectordb.index_manager import IndexManager
from healthcare_rag.config.logging_config import setup_logging

def parse_args():
    parser=argparse.ArgumentParser(
        description="Build vector index from medical documents"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to directory containing documents(default: data/raw/)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before building"
    )
    return parser.parse_args()
def main():
    setup_logging()
    args=parse_args()
    print("="* 50)
    print("Healthcare RAG Assitant - Index Builder")
    print("="*50)
    manager=IndexManager()
    stats=manager.get_index_stats()
    print(f"\nCurrent Index Stats:")
    for key, value in stats.items():
        print(f"{key}:{value}")

    print(f"\nBuilding index...")
    manager.build_index(
        data_dir=args.data_dir,
        clear_existing=args.clear
    )
    stats= manager.get_index_stats()
    print(f"\nupdated index Stats:")
    for key, value in stats.items():
        print(f" {key}:{value}")

    print("\nIndex build complete!")
    print("="* 50)
if __name__=="__main__":
    main()    