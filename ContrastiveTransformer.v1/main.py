import argparse
from training import train_contrastive
from evaluation import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Weather Classification Project")
    parser.add_argument("--train_contrastive", action="store_true", help="Train the contrastive learning model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    
    args = parser.parse_args()
    
    if args.train_contrastive:
        train_contrastive.main()
    elif args.evaluate:
        evaluate_model.main()
    else:
        print("No valid option selected. Use --help for more information.")
        
if __name__ == "__main__":
    main()
