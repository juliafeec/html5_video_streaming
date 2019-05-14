import pickle
with open("bkp_extracted_dict.pickle", "rb") as f:
    a = pickle.load(f)

name = "julia"

for k in list(a.keys()):
    if not k.startswith(name):
        del a[k]


print(a.keys())
with open("{}.pickle".format(name), "wb") as f:
    pickle.dump(a, f)
