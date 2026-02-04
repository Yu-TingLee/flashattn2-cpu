import os


class RuntimeLog:
    """CSV log at outputs/<name>/runtime.csv.

    Rows already present are keyed by their first key_len fields so an
    interrupted sweep can resume.
    """

    def __init__(self, name, header, key_len):
        out_dir = os.path.join("outputs", name)
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "runtime.csv")

        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                f.write(header + "\n")

        self.done = set()
        with open(self.path) as f:
            next(f, None)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= key_len:
                    self.done.add(tuple(int(p) for p in parts[:key_len]))

    def append(self, *fields):
        with open(self.path, "a") as f:
            f.write(",".join(str(x) for x in fields) + "\n")


def print_runtime(i, runtime):
    print("-" * 30)
    print(f"Runtime {i}: {runtime:.4f} seconds.")
    print("-" * 30)
