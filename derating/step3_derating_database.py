"""Step 3: simulation database for the load emulator, normal operation plus power de-rating transients."""
import time
from pathlib import Path
import database as db

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

if __name__ == "__main__":
    t = time.time()
    cases = db.sample_cases(n_normal=1000, n_derate=1500, seed=11)
    df, _ = db.run_cases(cases, seed=1000)
    df.to_csv(OUT / "sim_database.csv", index=False)
    print(f"{len(df)} 10-min simulations ({(df.kind == 'derate').sum()} with de-rating) in {time.time() - t:.0f} s")
