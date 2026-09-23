"""Create deterministic synthetic CSV and Excel fixtures for manual evaluation."""
from pathlib import Path
import pandas as pd


def main():
    target = Path("data")
    target.mkdir(exist_ok=True)
    sales = pd.DataFrame({"Region": ["West", "East", "West", "East"],
                          "Product": ["A", "A", "B", "B"],
                          "Revenue": [100, 200, 150, 50]})
    sales.to_csv(target / "sales.csv", index=False)
    with pd.ExcelWriter(target / "sales.xlsx") as writer:
        sales.to_excel(writer, sheet_name="Sales", index=False)
        pd.DataFrame({"Region": ["West", "East"], "Target": [300, 200]}).to_excel(
            writer, sheet_name="Targets", index=False)
    print("Created data/sales.csv and data/sales.xlsx. Total revenue: 500; West: 250; East: 250.")


if __name__ == "__main__":
    main()
