import pickle
with open("bkp_extracted_dict.pickle", "rb") as f:
    a = pickle.load(f)


for k in list(a.keys()):
    if not k.startswith("byron"):
        del a[k]


print(a.keys())
with open("byron.pickle", "wb") as f:
    pickle.dump(a, f)
