import argparse
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    print("Build patch indices placeholder.")

if __name__ == '__main__':
    main()
