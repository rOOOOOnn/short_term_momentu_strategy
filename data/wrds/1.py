import wrds

db = wrds.Connection(wrds_username="u2255434")

for lib in ["taqm_2024", "taqm_2025", "taqmsec", "taqmsamp"]:
    print("\n======================")
    print("LIBRARY:", lib)
    print("======================")
    try:
        tables = db.list_tables(library=lib)
        print("Total tables:", len(tables))
        print(tables[:100])
    except Exception as e:
        print("Error:", e)