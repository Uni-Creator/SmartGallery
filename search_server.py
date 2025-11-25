import text_search as ts
import sys

print("READY", flush=True)

while True:
    query = sys.stdin.readline().strip()
    if query == "EXIT":
        break

    # results = ts.search_images_basic(query)
    results = ts.search_all_above_alpha(query)

    for path, score in results:
        print(path)

    print("END", flush=True)
