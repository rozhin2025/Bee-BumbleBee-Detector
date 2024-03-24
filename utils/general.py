import numpy as np

safe_cast_to_f16 = lambda x: np.round(x, 3).astype(np.float16)


if __name__ == "__main__":
    pass
