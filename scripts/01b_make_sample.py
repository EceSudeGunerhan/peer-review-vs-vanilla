# scripts/01b_make_sample.py
# Run: python -m scripts.01b_make_sample

from src.data_prep.sample_split import main

if __name__ == "__main__":
    # You can change sample size here if you want
    main(sample_size=10)